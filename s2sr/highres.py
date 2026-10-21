"""High-resolution reference imagery aligned to a scene's grid.

Sources
- XYZ web tiles (default: Esri World Imagery). Fetched for the scene footprint,
  mosaicked in Web Mercator and warped onto the scene's own CRS/grid.
- A local GeoTIFF (e.g. a national orthophoto), warped the same way.

The result is an RGB uint8 image covering exactly the scene extent, at a pixel
size that is 10 m / 2^n so it nests in the 10 m grid. It is a visual reference
only: different date, sensor and radiometry from Sentinel-2.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import io
import math
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CACHE = Path.home() / ".cache" / "s2sr_bench" / "tiles"

PRESETS = {
    "Esri World Imagery": {
        "url": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        "attribution": "Esri, Maxar, Earthstar Geographics, and the GIS User Community",
        "max_zoom": 19,
    },
}
WEB_MERCATOR_RES0 = 156543.03392804097  # m/px at zoom 0 (256 px tiles)
ORIGIN = 20037508.342789244


@dataclass
class HighRes:
    image: np.ndarray        # (H, W, 3) uint8 on the scene grid
    res: float               # pixel size in metres
    label: str
    attribution: str = ""


def choose_res(scene_extent_m: float, max_px: int = 6144) -> float:
    """Finest of 10/2^n m (down to 0.625 m) that keeps the image <= max_px wide."""
    r = 0.625
    while scene_extent_m / r > max_px:
        r *= 2
    return r


def _target_grid(scene, res):
    from affine import Affine

    if scene.transform is None or scene.crs is None:
        raise ValueError("This scene has no georeferencing (e.g. the synthetic demo).")
    h10, w10 = scene.shape_at(10)
    k = 10.0 / res
    t = scene.transform
    tr = Affine(res, 0, t.c, 0, -res, t.f)
    return tr, int(round(h10 * k)), int(round(w10 * k))


def _tile_bytes(url: str) -> bytes | None:
    key = hashlib.sha1(url.encode()).hexdigest()
    p = CACHE / key[:2] / key
    if p.exists():
        return p.read_bytes()
    req = urllib.request.Request(url, headers={"User-Agent": "s2sr-bench/0.1 (research tool)"})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            data = r.read()
    except Exception:  # noqa: BLE001 - missing tiles are left blank
        return None
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(data)
    return data


def from_xyz(scene, url: str = PRESETS["Esri World Imagery"]["url"], label: str = "Esri World Imagery",
             attribution: str = PRESETS["Esri World Imagery"]["attribution"], max_zoom: int = 19,
             max_tiles: int = 1000, res: float | None = None, progress=None) -> HighRes:
    import rasterio
    from affine import Affine
    from PIL import Image
    from rasterio.warp import Resampling, reproject, transform_bounds

    h10, w10 = scene.shape_at(10)
    res = res or choose_res(max(h10, w10) * 10.0)
    dst_tr, H, W = _target_grid(scene, res)
    left, top = dst_tr.c, dst_tr.f
    right, bottom = left + W * res, top - H * res
    mx0, my0, mx1, my1 = transform_bounds(scene.crs, "EPSG:3857", left, bottom, right, top, densify_pts=21)
    lat = math.degrees(math.atan(math.sinh((my0 + my1) / 2 / 6378137.0)))
    # Zoom whose ground pixel is at least as fine as `res`
    z = int(math.ceil(math.log2(WEB_MERCATOR_RES0 * math.cos(math.radians(lat)) / res)))
    z = max(1, min(z, max_zoom))

    def tiles_at(z):
        n = 2 ** z
        span = 2 * ORIGIN / n
        x0, x1 = int((mx0 + ORIGIN) // span), int((mx1 + ORIGIN) // span)
        y0, y1 = int((ORIGIN - my1) // span), int((ORIGIN - my0) // span)
        return x0, x1, y0, y1, span

    x0, x1, y0, y1, span = tiles_at(z)
    while (x1 - x0 + 1) * (y1 - y0 + 1) > max_tiles and z > 1:
        z -= 1
        x0, x1, y0, y1, span = tiles_at(z)

    nx, ny = x1 - x0 + 1, y1 - y0 + 1
    mosaic = np.zeros((3, ny * 256, nx * 256), np.uint8)
    jobs = {(tx, ty): url.format(z=z, x=tx, y=ty) for tx in range(x0, x1 + 1) for ty in range(y0, y1 + 1)}
    done = 0
    with cf.ThreadPoolExecutor(8) as ex:
        futs = {ex.submit(_tile_bytes, u): xy for xy, u in jobs.items()}
        for f in cf.as_completed(futs):
            tx, ty = futs[f]
            data = f.result()
            done += 1
            if progress:
                progress(done / len(jobs), f"Tiles {done}/{len(jobs)} (zoom {z})")
            if not data:
                continue
            try:
                im = np.asarray(Image.open(io.BytesIO(data)).convert("RGB").resize((256, 256)))
            except Exception:  # noqa: BLE001
                continue
            r, c = (ty - y0) * 256, (tx - x0) * 256
            mosaic[:, r:r + 256, c:c + 256] = im.transpose(2, 0, 1)

    src_tr = Affine(span / 256, 0, x0 * span - ORIGIN, 0, -span / 256, ORIGIN - y0 * span)
    out = np.zeros((3, H, W), np.uint8)
    reproject(mosaic, out, src_transform=src_tr, src_crs="EPSG:3857", dst_transform=dst_tr,
              dst_crs=scene.crs, resampling=Resampling.bilinear)
    return HighRes(out.transpose(1, 2, 0), res, f"{label} (zoom {z})", attribution)


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    if arr.dtype == np.uint8:
        return arr
    a = arr.astype(np.float32)
    out = np.zeros(a.shape, np.uint8)
    for i in range(a.shape[0]):
        v = a[i][a[i] > 0]
        lo, hi = (np.percentile(v, [1, 99]) if v.size else (0, 1))
        out[i] = (np.clip((a[i] - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return out


def from_geotiff(scene, path, bands=(1, 2, 3), res: float | None = None) -> HighRes:
    """Warp RGB bands of any georeferenced raster onto the scene grid."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    h10, w10 = scene.shape_at(10)
    with rasterio.open(path) as ds:
        native = abs(ds.transform.a)
        is_byte = ds.dtypes[0] == "uint8"
        if res is None:
            res = choose_res(max(h10, w10) * 10.0)
            if ds.crs and ds.crs.is_projected:
                while res / 2 >= native and max(h10, w10) * 10 / (res / 2) <= 6144:
                    res /= 2
        dst_tr, H, W = _target_grid(scene, res)
        out = np.zeros((len(bands), H, W), np.float32)
        for i, b in enumerate(bands):
            reproject(rasterio.band(ds, b), out[i], dst_transform=dst_tr, dst_crs=scene.crs,
                      resampling=Resampling.bilinear)
    img = np.clip(np.round(out), 0, 255).astype(np.uint8) if is_byte else _to_uint8(out)
    return HighRes(img.transpose(1, 2, 0), res, f"Local: {Path(path).name}")


def save(hr: HighRes, scene, path) -> None:
    import rasterio

    tr, H, W = _target_grid(scene, hr.res)
    with rasterio.open(path, "w", driver="GTiff", width=W, height=H, count=3, dtype="uint8",
                       crs=scene.crs, transform=tr, compress="jpeg", photometric="ycbcr",
                       tiled=True, blockxsize=256, blockysize=256) as ds:
        ds.write(hr.image.transpose(2, 0, 1))
        ds.update_tags(source=hr.label, attribution=hr.attribution)
