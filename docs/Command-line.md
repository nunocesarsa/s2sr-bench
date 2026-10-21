---
tags: [s2sr-bench, cli]
---

# Command line

Back to [Home](Home.md)

Everything the app evaluates can also be run as a batch job. Inside the environment:

```bash
source ~/venvs/s2sr-gpu/bin/activate
python -m s2sr.cli --help
```

On Windows, `run_cli.bat <arguments>` does the same through WSL (file paths must be Linux paths, e.g.
`/mnt/d/data/scene.tif`).

## `models`: what can run

```bash
python -m s2sr.cli models
```

Lists every registered model with ✓/✗, its resolutions, and the reason it cannot run (missing package, no GPU, …).

## `sample`: download sample scenes

```bash
python -m s2sr.cli sample                          # amsterdam
python -m s2sr.cli sample --preset all
python -m s2sr.cli sample --name paris --lat 48.8566 --lon 2.3522 --size 512 --max-cloud 5
python -m s2sr.cli sample --list
```

See [Sample scenes](Sample-scenes.md).

## `run`: benchmark models

```bash
# From a STAC search (least cloudy scene in the date range)
python -m s2sr.cli run --stac 52.37 4.90 --start 2025-06-01 --end 2025-08-31 \
    --models bicubic_x4,sen2sr_lite_rgbn_x4,satlas_esrgan_1s2 --out results/amsterdam

# A local multi-band GeoTIFF (e.g. a sample)
python -m s2sr.cli run --tif samples/amsterdam/s2_l2a.tif --models bicubic_x4,sen2sr_lite_rgbn_x4

# A .SAFE product, 512 px around a point
python -m s2sr.cli run --safe /mnt/d/data/S2B_MSIL2A_....SAFE --center 41.39 2.17 --size 512
```

| Option | Default | Meaning |
|---|---|---|
| `--stac LAT LON` / `--tif PATH` / `--safe DIR` | synthetic scene | Input |
| `--start`, `--end`, `--max-cloud`, `--catalog` | 2025-06-01, 2025-08-31, 10, Earth Search | STAC search |
| `--size` | 512 | Subset size in 10 m pixels |
| `--band-order`, `--scale`, `--offset` | all 10 bands, 10000, 0 | GeoTIFF interpretation |
| `--models` | `bicubic_x4,bicubic_20to10,hpf_20to10` | Comma-separated model keys |
| `--mode` | `both` | `real`, `wald` or `both` (see [Evaluation](Evaluation.md)) |
| `--degrade` | `mtf` | `mtf` or `area` |
| `--device` | `auto` | `auto`, `cuda`, `cpu` |
| `--out` | `results` | Output folder |

Output:

- `metrics.csv`: one row per model and mode with the summary metrics and runtime,
- `<model>_metrics.json`: per-band metrics,
- `<model>.tif`: the real-scene output as float32 reflectance GeoTIFF.
