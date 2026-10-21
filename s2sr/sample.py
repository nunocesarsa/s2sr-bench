"""Download self-contained sample scenes into the project's samples/ folder.

Each sample is a folder with
  s2_l2a.tif     10 Sentinel-2 L2A bands on a 10 m grid (uint16, reflectance x 10000,
                 offset already removed; 20 m bands stored by pixel replication so
                 their native values are preserved)
  highres.tif    RGB aerial/satellite basemap of the same footprint (visual reference)
  info.json      provenance

Once downloaded, samples load without network access.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np

from .bands import ALL_BANDS, NATIVE_RES

SAMPLES_DIR = Path(__file__).resolve().parents[1] / "samples"

PRESETS = {
    "amsterdam": dict(lat=52.3765, lon=4.9000, note="Historic centre, canals, IJ waterfront"),
    "rotterdam_port": dict(lat=51.9500, lon=4.0500, note="Maasvlakte port: containers, cranes, ships"),
    "flevoland_farmland": dict(lat=52.5300, lon=5.6200, note="Large rectangular polder fields"),
    "berlin": dict(lat=52.5163, lon=13.3777, note="Mitte and Tiergarten"),
}


def download_sample(name: str, lat: float, lon: float, start: str = "2025-04-01", end: str = "2025-09-30",
                    size: int = 512, max_cloud: float = 5, catalog: str = "Earth Search (AWS)",
                    highres: bool = True, log=print) -> Path:
    import rasterio
    from affine import Affine

    from . import highres as hrmod
    from .data import stac
    from .resample import upsample

    out = SAMPLES_DIR / name
    out.mkdir(parents=True, exist_ok=True)
    log(f"[{name}] searching {catalog} {start}..{end}, cloud < {max_cloud}% at {lat:.4f}, {lon:.4f}")
    hits = stac.search(lat, lon, date.fromisoformat(start), date.fromisoformat(end), max_cloud, catalog)
    if not hits:
        raise SystemExit(f"[{name}] no scenes found; try a wider date range or higher cloud limit")

    best = None
    for h in hits[:4]:   # tile-level cloud cover says little about the AOI: check the SCL band
        log(f"[{name}] reading {h.label}")
        sc = stac.read_subset(h, lat, lon, size, catalog)
        clear = sc.meta.get("aoi_clear_fraction", 1.0)
        log(f"[{name}]   clear over AOI: {100 * clear:.0f}%")
        if best is None or clear > best.meta.get("aoi_clear_fraction", 0):
            best = sc
        if clear >= 0.98:
            break
    sc = best

    h10, w10 = sc.shape_at(10)
    stack = np.stack([upsample(sc.bands[b], int(NATIVE_RES[b] // 10), "nearest") for b in ALL_BANDS])
    dn = np.clip(np.round(stack * 10000), 0, 65535).astype(np.uint16)
    tr = sc.transform
    with rasterio.open(out / "s2_l2a.tif", "w", driver="GTiff", width=w10, height=h10, count=len(ALL_BANDS),
                       dtype="uint16", crs=sc.crs, transform=Affine(10, 0, tr.c, 0, -10, tr.f),
                       compress="deflate", predictor=2, tiled=True, blockxsize=256, blockysize=256) as ds:
        ds.write(dn)
        for i, b in enumerate(ALL_BANDS, 1):
            ds.set_band_description(i, b)
        ds.update_tags(scene_id=sc.meta.get("id", ""), datetime=sc.meta.get("datetime", ""),
                       scale="10000", offset_already_removed="true")
    log(f"[{name}] wrote s2_l2a.tif ({h10}x{w10} px, {sc.meta.get('datetime', '')[:10]})")

    info = {k: v for k, v in sc.meta.items() if isinstance(v, (str, int, float))}
    info.update(name=name, lat=lat, lon=lon, bands=ALL_BANDS, scale=10000, offset=0,
                note=PRESETS.get(name, {}).get("note", ""))
    if highres:
        try:
            hr = hrmod.from_xyz(sc, progress=None)
            hrmod.save(hr, sc, out / "highres.tif")
            info["highres"] = {"source": hr.label, "pixel_m": hr.res, "attribution": hr.attribution,
                               "note": "Visual reference only; acquisition date differs from Sentinel-2."}
            log(f"[{name}] wrote highres.tif ({hr.label}, {hr.res} m)")
        except Exception as e:  # noqa: BLE001
            log(f"[{name}] high-res download failed ({e}); the Sentinel-2 sample is still usable")
    (out / "info.json").write_text(json.dumps(info, indent=2))
    return out


def list_samples() -> list[Path]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted(p for p in SAMPLES_DIR.iterdir() if (p / "s2_l2a.tif").exists())


def load_sample(folder):
    """Return (Scene, HighRes or None)."""
    import rasterio

    from .data.local import load_geotiff
    from .highres import HighRes

    folder = Path(folder)
    sc = load_geotiff(folder / "s2_l2a.tif", ",".join(ALL_BANDS), 10000.0, 0.0)
    info = {}
    if (folder / "info.json").exists():
        info = json.loads((folder / "info.json").read_text())
        for k in ("id", "datetime", "tile_cloud_cover", "aoi_clear_fraction", "source"):
            if k in info:
                sc.meta[k] = info[k]
    sc.meta["sample"] = folder.name
    hr = None
    if (folder / "highres.tif").exists():
        with rasterio.open(folder / "highres.tif") as ds:
            img = ds.read([1, 2, 3]).transpose(1, 2, 0)
            res = abs(ds.transform.a)
        meta = info.get("highres", {})
        hr = HighRes(img, res, meta.get("source", "high-res"), meta.get("attribution", ""))
    return sc, hr
