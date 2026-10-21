"""Load Sentinel-2 data from local files.

Supported inputs
- A multi-band GeoTIFF (e.g. exported from GEE, cubo, SNAP) with a known band order.
- A folder with one file per band: a .SAFE product (L2A, JP2) or loose
  GeoTIFF/JP2 files whose names contain the band id (B02, B8A, ...).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from ..bands import ALL_BANDS, NATIVE_RES
from ..scene import Scene

DEFAULT_ORDER = ",".join(ALL_BANDS)


def parse_band_order(text: str) -> list[str]:
    names = [t.strip().upper() for t in re.split(r"[,\s]+", text) if t.strip()]
    fixed = []
    for n in names:
        if re.fullmatch(r"B\d", n):
            n = f"B0{n[1]}"
        fixed.append(n)
    return fixed


def load_geotiff(path, band_order: list[str] | str = DEFAULT_ORDER, scale: float = 10000.0,
                 offset: float = 0.0, window: tuple[int, int, int] | None = None) -> Scene:
    """Load a multi-band GeoTIFF where all bands share one grid.

    reflectance = (DN + offset) / scale. Use scale=1 if the file already holds reflectance.
    `window` = (row, col, size) in file pixels, to read only part of a large file.
    """
    import rasterio
    from rasterio.windows import Window

    if isinstance(band_order, str):
        band_order = parse_band_order(band_order)
    with rasterio.open(path) as ds:
        if ds.count < len(band_order):
            raise ValueError(f"File has {ds.count} bands but the order lists {len(band_order)}")
        win = Window(window[1], window[0], window[2], window[2]) if window else None
        arr = ds.read(list(range(1, len(band_order) + 1)), window=win).astype(np.float32)
        res = abs(ds.transform.a)
        transform = ds.window_transform(win) if win else ds.transform
        crs = ds.crs
        nodata = ds.nodata
    if nodata is not None:
        arr[arr == nodata] = np.nan
    refl = np.nan_to_num((arr + offset) / scale, nan=0.0)
    keep = [i for i, b in enumerate(band_order) if b in NATIVE_RES]
    names = [band_order[i] for i in keep]
    # Treat ~10 m files as 10 m; if the whole file is 20 m, the 10 m bands are coarser than native.
    nominal = 10.0 if res < 15 else 20.0
    if nominal == 10.0 and transform.a != 10.0:
        from affine import Affine
        transform = Affine(10.0, 0, transform.c, 0, -10.0, transform.f)
    sc = Scene.from_stack(refl[keep], names, nominal, crs=crs, transform=transform,
                          meta={"id": Path(path).name, "source": "local GeoTIFF",
                                "file_res": res, "offset_applied": offset})
    if nominal == 20.0:
        sc.meta["warning"] = "File is at 20 m; 10 m bands are coarser than native."
    return sc.crop_to_multiple(16)


def _find_band_files(folder: Path) -> dict[str, Path]:
    files = [p for p in folder.rglob("*") if p.suffix.lower() in {".jp2", ".tif", ".tiff"}]
    found: dict[str, Path] = {}
    for b in ALL_BANDS:
        pat = re.compile(rf"(^|[_\-.]){b}([_\-.]|$)", re.IGNORECASE)
        cands = [p for p in files if pat.search(p.stem)]
        if not cands:
            continue
        nat = NATIVE_RES[b]
        # In a .SAFE prefer the native-resolution folder (R10m / R20m).
        native = [p for p in cands if f"_{nat}m" in p.stem or f"R{nat}m" in str(p.parent)]
        found[b] = sorted(native or cands, key=lambda p: len(str(p)))[0]
    return found


def _safe_offset(folder: Path) -> float:
    for mtd in list(folder.rglob("MTD_MSIL2A.xml"))[:1]:
        txt = mtd.read_text(errors="ignore")
        m = re.search(r"<BOA_ADD_OFFSET[^>]*>\s*(-?\d+)\s*<", txt)
        if m:
            return float(m.group(1))
    return 0.0


def load_band_folder(folder, center_latlon: tuple[float, float] | None = None,
                     size_px: int = 512, scale: float = 10000.0, offset: float | None = None) -> Scene:
    """Load per-band files (e.g. a .SAFE). Reads a `size_px` window (10 m pixels)
    centred on `center_latlon`, or on the tile centre if not given."""
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import Window
    from affine import Affine

    folder = Path(folder)
    files = _find_band_files(folder)
    if not files:
        raise FileNotFoundError(f"No band files (B02, B03, ...) found under {folder}")
    if offset is None:
        offset = _safe_offset(folder)
    size_px = size_px // 16 * 16

    # Anchor on a 20 m band if available so windows nest.
    anchor_band = next((b for b in ["B05", "B11", "B8A"] if b in files), next(iter(files)))
    with rasterio.open(files[anchor_band]) as ds:
        anchor_res = abs(ds.transform.a)
        if center_latlon:
            xs, ys = warp_transform("EPSG:4326", ds.crs, [center_latlon[1]], [center_latlon[0]])
            cc, rc = ~ds.transform * (xs[0], ys[0])
        else:
            cc, rc = ds.width / 2, ds.height / 2
        n_anchor = int(size_px * 10 / anchor_res)
        col0 = int(np.clip(round(cc) - n_anchor // 2, 0, max(0, ds.width - n_anchor)))
        row0 = int(np.clip(round(rc) - n_anchor // 2, 0, max(0, ds.height - n_anchor)))
        x0, y0 = ds.transform * (col0, row0)
        crs = ds.crs

    out, res = {}, {}
    for b, p in files.items():
        with rasterio.open(p) as ds:
            r = abs(ds.transform.a)
            c, rr = ~ds.transform * (x0, y0)
            n = int(round(size_px * 10 / r))
            arr = ds.read(1, window=Window(int(round(c)), int(round(rr)), n, n),
                          boundless=True, fill_value=0).astype(np.float32)
        refl = np.where(arr == 0, 0, (arr + offset) / scale).astype(np.float32)
        out[b], res[b] = refl, float(r)
    meta = {"id": folder.name, "source": "local band files", "offset_applied": offset,
            "files": {b: str(p) for b, p in files.items()}}
    return Scene(out, res, crs, Affine(10.0, 0, x0, 0, -10.0, y0), meta).crop_to_multiple(16)
