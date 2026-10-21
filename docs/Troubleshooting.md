---
tags: [s2sr-bench, setup, troubleshooting]
---

# Troubleshooting

Back to [Home](Home.md) · related: [Installation](Installation.md)

Problems actually hit while setting the tool up on Windows + WSL with a GTX 1070, and their fixes.

## A model "failed": where is the error?

Each failed model shows a red **❌ … failed** panel above the tabs; expand it for the full traceback. To see
errors in a terminal instead:

```bash
python -m s2sr.cli run --models sen2sr_lite_rgbn_x4 --mode real
```

(no input = the synthetic scene, so no Sentinel-2 download is involved).

## `GET was unable to find an engine to execute this computation`

PyTorch sees the GPU but cannot run kernels on it, almost always because the installed build does not support
the card's architecture. Check:

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv
python -c "import torch; print(torch.__version__, torch.cuda.get_arch_list())"
```

If your compute capability (e.g. 6.1 → `sm_61`) is not in the list, install a matching build
([Installation](Installation.md#choosing-the-pytorch-build)). Pascal cards (GTX 10xx) need `cu126`. The app
detects this case and falls back to CPU, showing a warning under **Device**.

## Python 3.14 / no matching PyTorch build

Recent Ubuntu releases (e.g. 26.04) ship Python 3.14. PyTorch builds for older GPUs don't exist for it. Use a
separate Python 3.12 environment, which `install.sh` creates. If apt cannot provide Python 3.12 (e.g. the
deadsnakes PPA is unavailable, `503 Service Unavailable`), the installer falls back to
[uv](https://docs.astral.sh/uv/) to download a standalone Python 3.12.

## `No module named 'matplotlib'` when loading SEN2SR

SEN2SR's model loader, downloaded with the weights, imports matplotlib. It is in `requirements.txt`; in an older
environment run `pip install matplotlib` and restart the app.

## Full SEN2SR (Mamba) will not install

`mamba-ssm` and `causal-conv1d` must be compiled against your exact PyTorch/CUDA/Python combination (errors like
`nvcc was not found` or `name 'bare_metal_version' is not defined`). They target recent GPUs, and the model is
not included in the app for that reason. SEN2SR-Lite provides the same family of models.

## The map doesn't respond / location doesn't change

Use the map in the **main area** (shown when the source is *STAC search*): one click moves the marker and fills
in latitude/longitude. Scrolling zooms. If nothing happens, check that `folium` and `streamlit-folium` are
installed.

## STAC search or high-res download fails

Both need internet access from WSL. Earth Search and Esri occasionally rate-limit; retry after a minute. Tiles
and model weights are cached under `~/.cache/s2sr_bench/`, so repeated runs are faster. Delete that folder to
force fresh downloads.

## `.bat` files

- *"Could not find the Python environment"*: run `install.bat` first, or fix `VENV` at the top of the file.
- Wrong Linux distribution: set `DISTRO` at the top of each `.bat` (names from `wsl -l -v`).
- The browser opens before the app is ready: reload the page after a few seconds.

## Changes don't show up

Streamlit caches loaded models. After installing packages or editing code, stop the app (close the window /
Ctrl+C) and start it again.
