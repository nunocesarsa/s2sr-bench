"""Batch benchmarking from the command line.

Examples
  python -m s2sr.cli sample                    # download the default sample (amsterdam) into samples/
  python -m s2sr.cli sample --preset all       # every preset
  python -m s2sr.cli sample --name myplace --lat 48.85 --lon 2.35
  python -m s2sr.cli models
  python -m s2sr.cli run --stac 52.37 4.90 --start 2025-06-01 --end 2025-08-31 \
      --models bicubic_x4,sen2sr_lite_rgbn_x4,satlas_esrgan_1s2 --out results/amsterdam
  python -m s2sr.cli run --tif scene.tif --band-order B02,B03,B04,B08 --models bicubic_x4
  python -m s2sr.cli run --safe S2B_MSIL2A_....SAFE --center 41.39 2.17 --size 512
"""
from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path

import numpy as np


def _load_scene(a):
    if a.stac:
        from .data import stac
        lat, lon = a.stac
        hits = stac.search(lat, lon, date.fromisoformat(a.start), date.fromisoformat(a.end),
                           a.max_cloud, a.catalog)
        if not hits:
            raise SystemExit("No scenes found; widen the date range or cloud limit.")
        print(f"Using {hits[0].label}")
        return stac.read_subset(hits[0], lat, lon, a.size, a.catalog)
    if a.tif:
        from .data.local import load_geotiff
        return load_geotiff(a.tif, a.band_order, a.scale, a.offset)
    if a.safe:
        from .data.local import load_band_folder
        return load_band_folder(a.safe, tuple(a.center) if a.center else None, a.size, a.scale)
    from .scene import synthetic_scene
    print("No input given: using the synthetic test scene.")
    return synthetic_scene(a.size)


def main(argv=None):
    p = argparse.ArgumentParser(prog="s2sr", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("models", help="list registered models and whether they can run")

    sp = sub.add_parser("sample", help="download a sample scene (+ high-res image) into samples/")
    sp.add_argument("--preset", default="amsterdam", help="preset name or 'all' (see --list)")
    sp.add_argument("--list", action="store_true", help="list presets and downloaded samples")
    sp.add_argument("--name", help="custom sample name (use with --lat/--lon)")
    sp.add_argument("--lat", type=float)
    sp.add_argument("--lon", type=float)
    sp.add_argument("--start", default="2025-04-01")
    sp.add_argument("--end", default="2025-09-30")
    sp.add_argument("--size", type=int, default=512, help="subset size in 10 m pixels")
    sp.add_argument("--max-cloud", type=float, default=5)
    sp.add_argument("--no-highres", action="store_true", help="skip the high-res basemap image")

    r = sub.add_parser("run", help="run models and write metrics + GeoTIFFs")
    src = r.add_mutually_exclusive_group()
    src.add_argument("--stac", nargs=2, type=float, metavar=("LAT", "LON"))
    src.add_argument("--tif")
    src.add_argument("--safe", help="folder with per-band files or a .SAFE")
    r.add_argument("--start", default="2025-06-01")
    r.add_argument("--end", default="2025-08-31")
    r.add_argument("--max-cloud", type=float, default=10)
    r.add_argument("--catalog", default="Earth Search (AWS)")
    r.add_argument("--size", type=int, default=512, help="subset size in 10 m pixels")
    r.add_argument("--center", nargs=2, type=float, metavar=("LAT", "LON"))
    r.add_argument("--band-order", default="B02,B03,B04,B05,B06,B07,B08,B8A,B11,B12")
    r.add_argument("--scale", type=float, default=10000)
    r.add_argument("--offset", type=float, default=0)
    r.add_argument("--models", default="bicubic_x4,bicubic_20to10,hpf_20to10")
    r.add_argument("--mode", choices=["real", "wald", "both"], default="both")
    r.add_argument("--degrade", choices=["mtf", "area"], default="mtf")
    r.add_argument("--device", default="auto")
    r.add_argument("--out", default="results")
    a = p.parse_args(argv)

    if a.cmd == "sample":
        from .sample import PRESETS, download_sample, list_samples
        if a.list:
            for k, v in PRESETS.items():
                print(f"  {k:20s} {v['lat']:.4f}, {v['lon']:.4f}  {v['note']}")
            print("Downloaded:", ", ".join(p.name for p in list_samples()) or "none")
            return
        if a.name:
            if a.lat is None or a.lon is None:
                raise SystemExit("--name needs --lat and --lon")
            jobs = [(a.name, a.lat, a.lon)]
        elif a.preset == "all":
            jobs = [(k, v["lat"], v["lon"]) for k, v in PRESETS.items()]
        else:
            if a.preset not in PRESETS:
                raise SystemExit(f"Unknown preset {a.preset}; choose from {', '.join(PRESETS)} or 'all'")
            v = PRESETS[a.preset]
            jobs = [(a.preset, v["lat"], v["lon"])]
        for name, lat, lon in jobs:
            out = download_sample(name, lat, lon, a.start, a.end, a.size, a.max_cloud, highres=not a.no_highres)
            print(f"Saved {out}")
        return

    from .models import PLUGIN_ERRORS, registry

    if a.cmd == "models":
        for k, cls in registry().items():
            ok, why = cls.available()
            print(f"{'✓' if ok else '✗'} {k:26s} {cls.in_res:>4g}m→{cls.out_res:<4g}m  {cls.name}"
                  + ("" if ok else f"   [{why}]"))
        for f, e in PLUGIN_ERRORS.items():
            print(f"! plugin {f}: {e}")
        return

    from .export import write_geotiff
    from .runner import evaluate

    scene = _load_scene(a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    reg = registry()
    rows = []
    for key in [k.strip() for k in a.models.split(",") if k.strip()]:
        cls = reg.get(key)
        if cls is None:
            print(f"✗ unknown model {key}")
            continue
        ok, why = cls.available()
        if not ok or not scene.has(cls.in_bands):
            print(f"✗ skip {key}: {why or 'scene lacks required bands'}")
            continue
        print(f"→ {key}")
        res = evaluate(cls(device=a.device), scene, a.mode, a.degrade)
        for mode, rr in res.items():
            row = {"model": key, "mode": mode, "runtime_s": round(rr.runtime_s, 3), **rr.metrics.get("summary", {})}
            rows.append(row)
            print("   ", mode, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k not in ("model", "mode")})
            if mode == "real":
                write_geotiff(rr, out / f"{key}.tif")
        (out / f"{key}_metrics.json").write_text(json.dumps(
            {m: rr.metrics for m, rr in res.items()}, indent=2, default=float))
    if rows:
        import csv
        keys = sorted({k for r in rows for k in r}, key=lambda k: (k not in ("model", "mode"), k))
        with open(out / "metrics.csv", "w", newline="") as f:
            w = csv.DictWriter(f, keys)
            w.writeheader()
            w.writerows(rows)
        print(f"Wrote {out / 'metrics.csv'}")


if __name__ == "__main__":
    main()
