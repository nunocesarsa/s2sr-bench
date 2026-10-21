#!/usr/bin/env bash
# One-time setup of the Python environment inside WSL Ubuntu.
# Run via install.bat from Windows, or `bash install.sh` in Ubuntu.
set -e
VENV="$HOME/venvs/s2sr-gpu"
# PyTorch build: cu126 supports GTX 10xx (Pascal) up to RTX 40xx. RTX 50xx needs cu128; no GPU: cpu.
#   TORCH_INDEX=https://download.pytorch.org/whl/cu128 bash install.sh
TORCH_INDEX="${TORCH_INDEX:-https://download.pytorch.org/whl/cu126}"
cd "$(dirname "$0")"

echo "== 1/4  Python 3.12 =="
# Ubuntu 26.04 ships Python 3.14, which the GTX 1070-compatible PyTorch does not support.
# Try apt (deadsnakes PPA) first; if that fails, fall back to uv, which downloads a
# standalone Python 3.12 without touching the system.
PY=""
if command -v python3.12 >/dev/null 2>&1 && python3.12 -c "import venv, ensurepip" 2>/dev/null; then
  PY=python3.12
else
  echo "Trying apt (you may be asked for your Ubuntu password)..."
  if sudo apt-get install -y python3.12 python3.12-venv 2>/dev/null \
     && python3.12 -c "import venv, ensurepip" 2>/dev/null; then
    PY=python3.12
  else
    echo "apt could not provide Python 3.12 - using uv instead."
    if ! command -v uv >/dev/null 2>&1 && [ ! -x "$HOME/.local/bin/uv" ]; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
    fi
    export PATH="$HOME/.local/bin:$PATH"
    uv python install 3.12
    PY="$(uv python find 3.12)"
  fi
fi
echo "Using $PY"

echo "== 2/4  Environment $VENV =="
if [ ! -f "$VENV/bin/activate" ]; then
  "$PY" -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip

echo "== 3/4  PyTorch from $TORCH_INDEX =="
pip install torch==2.8.0 torchvision==0.23.0 --index-url "$TORCH_INDEX"
pip install -r requirements.txt
# Make sure nothing in requirements replaced the chosen PyTorch build
pip install torch==2.8.0 torchvision==0.23.0 --index-url "$TORCH_INDEX" >/dev/null

echo "== 4/4  GPU check =="
python - <<'PY'
import torch
print("PyTorch", torch.__version__, "| CUDA", torch.version.cuda)
if not torch.cuda.is_available():
    print("No GPU visible: the app will run on CPU.")
else:
    try:
        x = torch.rand(1, 3, 64, 64, device="cuda")
        torch.nn.functional.conv2d(x, torch.rand(8, 3, 3, 3, device="cuda"))
        print("GPU OK:", torch.cuda.get_device_name(0))
    except Exception as e:
        print("GPU visible but not usable, the app will use CPU:", e)
PY
echo
echo "Done. Start the app with run_app.bat"
