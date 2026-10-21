---
tags: [s2sr-bench, models]
---

# Models

Back to [Home](Home.md)

Every model declares a **band contract**: which bands it reads at which pixel size, and which bands it returns
at which pixel size. The app prepares the input accordingly (bicubic for bands coarser than the model's input
grid) and checks the output shape. Two families of task:

- **×4 super-resolution**: 10 m bands → 2.5 m.
- **×2 band sharpening**: 20 m bands → 10 m, guided by the 10 m bands.

| Model (key) | Type | Input | Output | Needs |
|---|---|---|---|---|
| Bicubic ×4 (`bicubic_x4`) | baseline | B04 B03 B02 B08 @10 m | same @2.5 m | — |
| Nearest ×4 (`nearest_x4`) | baseline | B04 B03 B02 B08 @10 m | same @2.5 m | — |
| Bicubic 20→10 m (`bicubic_20to10`) | baseline | six 20 m bands | same @10 m | — |
| HPF injection 20→10 m (`hpf_20to10`) | band sharpening | all 10 bands @10 m | six 20 m bands @10 m | — |
| SEN2SR-Lite RGBN ×4 (`sen2sr_lite_rgbn_x4`) | CNN | B04 B03 B02 B08 @10 m | same @2.5 m | torch, sen2sr, mlstac |
| SEN2SR-Lite 10 bands → 2.5 m (`sen2sr_lite_full`) | CNN | all 10 bands @10 m | all 10 @2.5 m | torch, sen2sr, mlstac |
| SEN2SR-Lite 20→10 m (`sen2sr_lite_rswir_x2`) | CNN, band sharpening | all 10 bands @10 m | six 20 m bands @10 m | torch, sen2sr, mlstac |
| Satlas ESRGAN ×4 (`satlas_esrgan_1s2`) | GAN | B04 B03 B02 @10 m | same @2.5 m | torch |

`python -m s2sr.cli models` shows which ones can run on your machine and why not.

## Baselines

Always include them. A learned model is only adding real information if it beats its baseline on the Wald
test (see [Evaluation](Evaluation.md)).

- **Bicubic**: smooth interpolation, no new detail. The reference point for every ×4 model.
- **Nearest**: pixel replication; shows the raw 10 m sampling at 2.5 m display size.
- **Bicubic 20→10 m**: the reference point for band sharpening.

## HPF injection 20→10 m

A classical pansharpening method (high-pass filter / detail injection), included as a strong non-learned
baseline for band sharpening. For each 20 m band:

1. Fit a linear regression, at 20 m, of the band on the four 10 m bands (as seen by a 20 m sensor, i.e. blurred
   with the Sentinel-2 MTF). This builds a synthetic 10 m "pan" image that best mimics the band.
2. Take the pan's high frequencies (pan minus its 20 m-blurred version).
3. Add them to the bicubic-upsampled 20 m band.

It works well where the 20 m band's edges coincide with the 10 m bands' edges (field boundaries, roads), less so
for SWIR-specific features.

## SEN2SR-Lite (ESA OpenSR)

Lightweight CNNs from the ESA-funded OpenSR project, trained to super-resolve Sentinel-2 to 2.5 m. They include a
**hard radiometric-consistency constraint**: degraded back to 10 m, the output reproduces the input. Expect
excellent consistency scores and conservative, sometimes soft, detail.

- Weights: `huggingface.co/tacofoundation/sen2sr`, fetched with `mlstac`, cached in `~/.cache/s2sr_bench/sen2sr/`.
- Runs on 128 px patches; the app tiles larger inputs with overlap and feathered blending (any shape works).
- The full SEN2SR model (Mamba architecture) is not included: it needs `mamba-ssm`, which requires a recent GPU
  and compiling CUDA code.

## Satlas ESRGAN (Allen AI)

The single-image ESRGAN generator from Satlas super-resolution, trained to map Sentinel-2 to NAIP aerial imagery
(USA). The network is defined inside the app (no extra package); weights (~67 MB) come from Allen AI's storage.

> [!WARNING]
> - Trained on Sentinel-2 **L1C true-colour (TCI) 8-bit images**, not L2A reflectance. The app converts reflectance
>   with `reflectance / tci_max × 255` (default `tci_max` = 0.2, adjustable under *Satlas options*) and converts the
>   output back the same way. Its output is therefore only approximately reflectance.
> - Trained on the **USA** only. Elsewhere it can produce sharp but invented structure (buildings, cars, boats).
>   Check with [High-res reference](High-res-reference.md) and the Residuals tab.
> - Output is RGB only: no NIR or 20 m bands.

## Tiling

Models that work on fixed patches go through `s2sr/models/tiling.py`: the input is cut into overlapping tiles
(reflect-padded if smaller than a tile), each prediction is weighted with a linear ramp in the overlap, and
the weighted sum is normalised. This avoids seams and handles non-square images.

## Your own models

See [Adding a model](Adding-a-model.md).
