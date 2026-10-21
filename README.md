# S2 SR Bench

**Test super-resolution models on Sentinel-2 imagery: side by side, against aerial photos, and with metrics.**

S2 SR Bench is a local Streamlit app (plus a command-line tool) that runs several super-resolution models on
exactly the same Sentinel-2 L2A subset and lets you judge them three ways:

- **Look**: synchronised zoom across all outputs with one shared colour stretch, a swipe/fade view over a
  high-resolution aerial image, spectral profiles and residual maps.
- **Measure**: a Wald-protocol test against a real reference (PSNR, SSIM, SAM, ERGAS) plus no-reference
  checks on the real scene (radiometric consistency, sharpness gain, spectral change).
- **Export**: super-resolved GeoTIFFs and metric tables.

## Models included

| Model | Task |
|---|---|
| Bicubic / nearest | Baselines, 10 m → 2.5 m |
| Bicubic 20→10 m, **HPF injection** | Baselines for 20 m band sharpening |
| **SEN2SR-Lite** (ESA OpenSR): RGBN ×4, 10 bands → 2.5 m, 20→10 m | CNN with radiometric-consistency constraint |
| **Satlas ESRGAN** (Allen AI) | GAN, RGB ×4 |
| **Your own** | TorchScript / ONNX / PyTorch via a config file or Python plugin |

## Quick start

**Windows (via WSL Ubuntu)**: double-click:

1. `install.bat`: one-time setup (Python 3.12 environment, PyTorch, dependencies, GPU check)
2. `download_sample.bat`: fetch a sample area (Sentinel-2 + aerial image) into `samples/`
3. `run_app.bat`: opens the app at <http://localhost:8501>

**Linux**

```bash
bash install.sh                                  # creates ~/venvs/s2sr-gpu
source ~/venvs/s2sr-gpu/bin/activate
python -m s2sr.cli sample                        # optional sample scene
streamlit run app.py
```

The default PyTorch build (CUDA 12.6) supports GPUs from the GTX 10xx series up to RTX 40xx; other GPUs
and CPU-only setups are covered in [Installation](docs/Installation.md). Without a usable GPU everything runs
on CPU.

## Documentation

Full documentation is in [`docs/`](docs/Home.md). It is also an [Obsidian](https://obsidian.md) vault:
open the folder in Obsidian for linked browsing.

| | |
|---|---|
| [Installation](docs/Installation.md) | Setup, choosing the PyTorch build, `.bat` helpers |
| [Using the app](docs/Using-the-app.md) | Sidebar workflow and every tab |
| [Data sources](docs/Data-sources.md) | STAC search, GeoTIFF, `.SAFE`, reflectance handling |
| [Sample scenes](docs/Sample-scenes.md) | Downloadable test areas |
| [Models](docs/Models.md) | What each model does, and its caveats |
| [Evaluation](docs/Evaluation.md) | Wald protocol, no-reference metrics, how to read them |
| [High-res reference](docs/High-res-reference.md) | Aerial imagery overlay |
| [Command line](docs/Command-line.md) | Batch benchmarks |
| [Adding a model](docs/Adding-a-model.md) | Plug in your own network |
| [Architecture](docs/Architecture.md) | Code layout and design |
| [Troubleshooting](docs/Troubleshooting.md) | GPU, Python and download issues |

## Project layout

```
app.py                Streamlit app
s2sr/                 library: data loading, models, metrics, visualisation, CLI
custom_models/        your model configs / plugins (templates included)
docs/                 documentation (Obsidian vault)
samples/              downloaded sample scenes (not in Git)
tests/                offline smoke tests (python -m pytest -q tests)
*.bat, install.sh     Windows/WSL helpers and installer
```

## Notes

- There is no 2.5 m ground truth for Sentinel-2; read the metrics together and check visually.
  See [Evaluation](docs/Evaluation.md).
- Model weights are downloaded at first use from their original publishers and are covered by their own licences
  (SEN2SR: CC0-1.0; Satlas: Apache-2.0).
- Esri World Imagery, used for the high-res reference, is © Esri, Maxar, Earthstar Geographics and the GIS User
  Community. Check Esri's terms before redistributing downloaded imagery.

## Licence

MIT (see [LICENSE](LICENSE)). Contains modified Copernicus Sentinel data when used with Sentinel-2 imagery.
