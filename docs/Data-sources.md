---
tags: [s2sr-bench, data]
---

# Data sources

Back to [Home](Home.md)

Every source produces the same internal object, a **Scene**: Sentinel-2 L2A surface reflectance (≈ 0–1) with
each band kept at its **native pixel size**: B02, B03, B04, B08 at 10 m; B05, B06, B07, B8A, B11, B12 at 20 m.
Keeping native sampling matters: a model's 20 m input is the real 20 m measurement, not an interpolated copy,
and metrics compare against what the sensor actually recorded.

Subsets are always cropped to a multiple of 16 pixels at 10 m, so 10 m and 20 m grids nest exactly and the
degradation factors used in [Evaluation](Evaluation.md) divide evenly.

## Sample scenes
Pre-downloaded folders in `samples/`. See [Sample scenes](Sample-scenes.md).

## STAC search
Searches a public STAC catalogue for Sentinel-2 L2A scenes around a point:

| Catalogue | Account | Notes |
|---|---|---|
| Earth Search (AWS, Element 84) | none | default |
| Microsoft Planetary Computer | none (URLs are signed automatically) | |

Workflow: click the map (or type lat/lon) → date range, maximum tile cloud cover, subset size → **Search** →
choose a scene (sorted by cloud cover) → **Load subset**.

Only the requested window is read from the Cloud-Optimised GeoTIFFs over HTTP, so a 512 × 512 px subset
(5.12 km) is a few MB rather than the full 800 MB tile. The window is anchored on the 20 m grid so both
resolutions line up. The **SCL** (scene classification) band is read too and reported as "AOI clear":
tile-level cloud cover says little about your small area.

## Local GeoTIFF
A multi-band file where all bands share one grid (typical of Google Earth Engine, cubo or SNAP exports).

- **Band order**: comma-separated, one name per file band, e.g. `B04,B03,B02,B08`.
- **Scale / Offset**: reflectance = (DN + offset) / scale. Raw L2A DN since 2022 need offset −1000.
- 20 m bands that were resampled to 10 m in the file are block-averaged back to 20 m.

## Local band files / .SAFE
A `.SAFE` product folder or any folder with one file per band (`*_B02_*.tif`, `*_B8A_*.jp2`, …). In a `.SAFE`,
the native-resolution folders (`R10m`, `R20m`) are preferred and `BOA_ADD_OFFSET` is read from
`MTD_MSIL2A.xml`. A window of the chosen size is read around a lat/lon or the tile centre.

## Synthetic demo
A generated scene (fields, roads, smooth gradients with a vegetation-like spectrum). No georeferencing, so the
high-res reference is not available for it. Used by the tests.

## Reflectance and the 2022 offset

> [!IMPORTANT]
> From processing baseline 04.00 (25 January 2022) L2A digital numbers include a +1000 offset:
> reflectance = (DN − 1000) / 10000. The STAC loader removes it unless the catalogue already did
> (Earth Search sets `earthsearch:boa_offset_applied`); the `.SAFE` loader reads it from the metadata.
> The scene header shows the offset that was applied.

Pixels with DN 0 (no data) become reflectance 0; the count is stored in the scene metadata.
