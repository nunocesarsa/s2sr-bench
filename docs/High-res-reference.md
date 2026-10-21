---
tags: [s2sr-bench, visualisation]
---

# High-res reference

Back to [Home](Home.md)

An aerial or very-high-resolution satellite image of the scene footprint, shown next to and underneath the
model results. It answers the question metrics cannot: *is the detail a model added actually there?*

## Sources

Sidebar → **3 · High-res reference** → **Load high-res**:

| Option | Details |
|---|---|
| **Esri World Imagery** | Global basemap, often sub-metre in Europe and North America. Default. |
| **Custom XYZ tiles** | Any `{z}/{x}/{y}` Web-Mercator tile URL, e.g. a national orthophoto service. |
| **Local GeoTIFF** | Any georeferenced RGB raster (a downloaded orthophoto, drone mosaic, …). |

[Sample scenes](Sample-scenes.md) include one already (`highres.tif`).

## How it is aligned

1. The scene footprint is converted to Web Mercator and the tile zoom is chosen so the ground pixel is at least as
   fine as the target (capped at 1000 tiles; tiles are cached in `~/.cache/s2sr_bench/tiles`).
2. Tiles are mosaicked and **reprojected onto the scene's own grid** (same CRS and origin) at a pixel size of
   10 m / 2ⁿ: 0.625 m for subsets up to ~3.8 km, 1.25 m up to ~7.7 km, and so on. Local GeoTIFFs are warped the
   same way.

Because the image shares the scene grid, every zoom/crop in the app cuts exactly the same ground area from it as
from the model outputs.

## Where it appears

- **Compare**: a *High-res reference* panel, cropped in sync with the others.
- **Swipe**: by default underneath the newest model result. Drag to move the divider; **Opacity** fades the model
  over the aerial image, which makes misplaced or invented objects easy to spot.

> [!NOTE]
> Visual reference only. It differs in acquisition date, sensor, spectral bands, viewing angle and processing,
> so it is **not** used in any metric. Expect seasonal differences and building lean in aerial photos.

> [!WARNING]
> Esri World Imagery: © Esri, Maxar, Earthstar Geographics and the GIS User Community. Fine for viewing;
> check Esri's terms of use before redistributing downloaded imagery.
