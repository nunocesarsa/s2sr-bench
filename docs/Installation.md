---
tags: [s2sr-bench, setup]
---

# Installation

Back to [Home](Home.md)

The app is Python. It runs on Linux, or on Windows through **WSL** (Windows Subsystem for Linux, an Ubuntu
inside Windows), which is the tested setup. A GPU is optional: everything runs on CPU, only slower.

## Windows (WSL Ubuntu): recommended

Prerequisites: WSL with an Ubuntu distribution (`wsl --install` in an administrator PowerShell), and, for GPU
use, the normal **Windows** NVIDIA driver. Do not install a Linux NVIDIA driver inside Ubuntu; WSL passes
the Windows one through.

1. Put the project folder anywhere on a Windows drive, e.g. `D:\S2SR_Bench`.
2. Double-click **`install.bat`** (one time, 5–15 min, ~3 GB download). It:
   - installs Python 3.12 inside Ubuntu if missing (it may ask for your Ubuntu password; typing is invisible),
   - creates the environment `~/venvs/s2sr-gpu`,
   - installs PyTorch and everything in `requirements.txt`,
   - finishes with a GPU check: `GPU OK: <your card>` or a note that it will use the CPU.
3. Double-click **`run_app.bat`**. The app opens at <http://localhost:8501> in your Windows browser.

Re-running `install.bat` is safe: it reuses what is already installed.

| File | Purpose |
|---|---|
| `install.bat` → `install.sh` | One-time environment setup |
| `run_app.bat` | Start the app |
| `download_sample.bat` | Fetch [Sample scenes](Sample-scenes.md) |
| `run_cli.bat` | Pass arguments to the [Command line](Command-line.md) tool |

If your environment or WSL distribution has a different name, edit `VENV` / `DISTRO` at the top of the `.bat` files
(`wsl -l -v` lists distributions).

## Linux / manual install

```bash
python3.12 -m venv ~/venvs/s2sr-gpu
source ~/venvs/s2sr-gpu/bin/activate
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
streamlit run app.py
```

Or simply `bash install.sh`, then `source ~/venvs/s2sr-gpu/bin/activate && streamlit run app.py`.

## Choosing the PyTorch build

The PyTorch build must support your GPU's *compute capability* (`nvidia-smi --query-gpu=name,compute_cap --format=csv`).

| GPU generation | Example cards | Build to use |
|---|---|---|
| Pascal / Volta (6.x–7.0) | GTX 10xx, Titan V | CUDA 12.6 (`cu126`, the installer default) |
| Turing → Hopper (7.5–9.0) | RTX 20xx/30xx/40xx | `cu126` or newer |
| Blackwell (10.0, 12.0) | RTX 50xx | `cu128` or newer |
| none / CPU only | | `cpu` |

The installer uses `cu126`. To pick another build, set `TORCH_INDEX` before running it, e.g.
`TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash install.sh`.

> [!WARNING]
> CUDA 13 builds of PyTorch dropped support for Pascal and Volta cards (e.g. GTX 1070), and recent Ubuntu
> releases ship Python 3.14, for which only those newest builds exist. That is why the installer creates a
> separate Python 3.12 environment. Details in [Troubleshooting](Troubleshooting.md).

## First run downloads

Model weights download automatically the first time a model runs and are cached in `~/.cache/s2sr_bench/`
(inside Ubuntu): SEN2SR-Lite a few MB per variant, Satlas ESRGAN ~67 MB. High-res tiles are cached in
`~/.cache/s2sr_bench/tiles/`.

## Checking the install

```bash
source ~/venvs/s2sr-gpu/bin/activate
python -m s2sr.cli models        # ✓ / ✗ for every model, with the reason
python -m pytest -q tests        # offline smoke tests
```

The app sidebar also shows the device in use under **Device** ("Running on cuda/cpu"), plus a warning if the
GPU is visible but unusable.
