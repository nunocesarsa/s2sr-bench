---
tags: [s2sr-bench, guide]
---

# Using the app

Back to [Home](Home.md)

Start it with `run_app.bat` (Windows) or `streamlit run app.py`. The **sidebar** is the workflow, top to bottom.
The **main area** shows the results in tabs.

## Sidebar

### 1 · Data
Choose where the Sentinel-2 subset comes from; see [Data sources](Data-sources.md) for the details of each.

- **Sample scenes**: pre-downloaded areas from the `samples` folder ([Sample scenes](Sample-scenes.md)). Fastest start, works offline.
- **STAC search**: pick a location on the map in the main area (one click moves the marker; the yellow
  box is the subset footprint), set dates, cloud limit and subset size, press **Search**, choose a scene, **Load subset**.
- **Local GeoTIFF**, **Local band files / .SAFE**: your own files (Linux paths, e.g. `/mnt/d/data/...`).
- **Synthetic demo**: a generated scene for testing without data.

After loading, the header shows date, size, tile cloud cover and the clear fraction over your subset.

### 2 · Models
Pick models to run (see [Models](Models.md)). Models that cannot run are listed under "unavailable" with
the reason. **Device**: `auto` uses the GPU if PyTorch can really use it. **Satlas options** sets how
reflectance is converted to the 8-bit images Satlas expects.

### 3 · High-res reference
Load an aerial/satellite image of the same footprint (Esri World Imagery, a custom tile URL, or a local GeoTIFF).
See [High-res reference](High-res-reference.md). Samples already include one.

### 4 · Evaluation
- **Mode**: *Both*, *Real scene (no reference)* or *Wald (degraded reference)*. See [Evaluation](Evaluation.md).
- **Degradation model**: `mtf` (sensor-like blur, default) or `area` (block mean).
- **Ignore border**: pixels excluded from metrics at the image edge.
- **▶ Run models** runs every selected model; results stay until you load another scene.

## Tabs

### 🖼️ Compare
A grid of panels showing the same area: the input at its true sampling, the high-res reference (if loaded),
and each model's output. All panels share one **colour stretch** (computed from the input), so brightness
differences are real.

- **Display**: band combination. True colour, false colour NIR, red-edge, SWIR, agriculture, *20 m bands only*,
  or any single band.
- **View**: *Real* (models applied to the actual scene) or *Wald* (applied to the degraded scene, shown next to
  the degraded input and the original as reference).
- **Zoom / Centre Y / Centre X**: zoom in up to 32× on the same spot in every panel.

> [!TIP]
> Band-sharpening models (HPF, SEN2SR 20→10 m) only output the 20 m bands. They appear under *SWIR*,
> *Red-edge* or *20 m bands only*; the missing 10 m band (e.g. B04) is taken from the input and the caption
> says "from input". Under *True colour* they have nothing of their own to show.

### ↔️ Swipe
Two images stacked in the same window. Drag across the image (or use **Swipe**) to move the divider, and use
**Opacity** to fade the top image over the one underneath. By default the newest model result is on top and the
high-res reference underneath.

### 📊 Metrics
One table per evaluation mode, best value per column highlighted, a bar chart of any metric, per-band
details, and CSV download. How to read them: [Evaluation](Evaluation.md).

### 🌈 Spectra
Reflectance per band at one location, for the input and each model. Shows whether a model keeps the spectral
signature: lines should overlap for a 10 m footprint.

### 🔍 Residuals
Difference maps for one model and band:
- *Consistency*: model output degraded back to native resolution, minus the input. Structure here means the model
  changed what the satellite actually measured.
- *Error vs ref* (Wald): output minus the original. Shows where detail was not recovered.

Blue = too dark, red = too bright. Scale is ±99th percentile of the difference.

### 💾 Export
Float32 GeoTIFF of each model's real-scene output (surface reflectance, georeferenced at the output pixel size),
for use in QGIS or elsewhere.

### ℹ️ Metrics guide
A short in-app version of [Evaluation](Evaluation.md).

## A typical session

1. Load the `amsterdam` sample.
2. Run *Bicubic ×4*, *SEN2SR-Lite RGBN ×4*, *Satlas ESRGAN ×4*, *HPF injection 20→10 m*, *Bicubic 20→10 m*.
3. **Compare** at zoom 8 on a canal: typically Satlas looks sharpest and SEN2SR softer but more faithful.
4. **Swipe** Satlas over the high-res image with opacity ~50 %: do the invented buildings line up?
5. **Metrics**: does each model beat its bicubic baseline on the Wald test? Is consistency PSNR high?
6. **Residuals** for Satlas, B04, Consistency: structured patterns mean it altered the observed signal.
