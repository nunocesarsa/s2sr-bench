---
tags: [s2sr-bench, data]
---

# Sample scenes

Back to [Home](Home.md)

A sample is a self-contained test area stored in `samples/<name>/`. Once downloaded it loads instantly and
offline: choose **Sample scenes** in the sidebar, pick it, **Load sample**.

## Download

Windows: double-click `download_sample.bat`, or run it from a console with arguments:

```
download_sample.bat                  default preset (amsterdam)
download_sample.bat all              every preset
download_sample.bat berlin           one preset
download_sample.bat --name paris --lat 48.8566 --lon 2.3522 --start 2025-05-01 --end 2025-09-30
```

Linux / inside WSL: `python -m s2sr.cli sample` with the same arguments (`--preset berlin`, `--preset all`,
`--name … --lat … --lon …`, `--size`, `--max-cloud`, `--no-highres`). `--list` shows presets and what is
already downloaded.

| Preset | Location | Why it is useful |
|---|---|---|
| `amsterdam` | 52.3765, 4.9000 | Canals, dense historic buildings, water edges |
| `rotterdam_port` | 51.9500, 4.0500 | Containers, cranes, ships: small bright objects |
| `flevoland_farmland` | 52.5300, 5.6200 | Straight field boundaries, homogeneous parcels |
| `berlin` | 52.5163, 13.3777 | Large park next to a dense centre |

## What the downloader does

1. Searches Earth Search for scenes between `--start` and `--end` (default April–September 2025) with
   tile cloud cover below `--max-cloud` (default 5 %).
2. Reads the subset from up to four of the least cloudy scenes and keeps the one with the highest clear
   fraction **over the subset** (from the SCL band), stopping early at ≥ 98 % clear.
3. Writes the files below, then fetches the [High-res reference](High-res-reference.md) for the same footprint.

## Folder contents

| File | Content |
|---|---|
| `s2_l2a.tif` | 10 bands in order B02, B03, B04, B05, B06, B07, B08, B8A, B11, B12 on a 10 m grid; uint16 = reflectance × 10000 with the 2022 offset already removed. 20 m bands are stored by pixel replication, so their native values are preserved exactly and recovered on load. |
| `highres.tif` | RGB, uint8, ~0.6–1.25 m, same footprint and CRS |
| `info.json` | Scene id, acquisition date, cloud figures, coordinates, high-res source and attribution |

`s2_l2a.tif` is a normal GeoTIFF: it opens in QGIS and can also be loaded via **Local GeoTIFF**.

> [!NOTE]
> The `samples/` folder is excluded from Git by default (`.gitignore`): sample files are a few to tens of MB
> each, and the Esri imagery in `highres.tif` should not be redistributed without checking Esri's terms.
> Anyone who clones the repository can recreate the samples with the same command.
