"""Bring-your-own models.

Two ways to add a model, both picked up automatically from `custom_models/`
(or any folder in the S2SR_MODEL_DIRS environment variable, ':'-separated):

1. A TOML file describing a TorchScript, ONNX or plain PyTorch checkpoint:

       name = "My ESRGAN"
       format = "torchscript"          # torchscript | onnx | pytorch
       path = "weights/my_model.pt"    # relative to the TOML file
       in_bands = ["B04", "B03", "B02", "B08"]
       out_bands = ["B04", "B03", "B02", "B08"]
       in_res = 10
       out_res = 2.5
       input_scale = 1.0     # model input  = reflectance * input_scale
       output_scale = 1.0    # reflectance  = model output / output_scale
       tile = 128            # 0 = whole image at once
       overlap = 16
       # format = "pytorch" also needs:  factory = "my_module:build_model"
       #   (a function returning an nn.Module; my_module.py next to the TOML)

2. A Python file that subclasses SRModel and decorates it with @register
   (see custom_models/example_plugin.py).
"""
from __future__ import annotations

import importlib.util
import os
import sys
import tomllib
from pathlib import Path

import numpy as np

from .base import SRModel, register
from .tiling import tiled_predict

DEFAULT_DIR = Path(__file__).resolve().parents[2] / "custom_models"
_errors: dict[str, str] = {}


def model_dirs() -> list[Path]:
    dirs = [DEFAULT_DIR]
    extra = os.environ.get("S2SR_MODEL_DIRS", "")
    dirs += [Path(p) for p in extra.split(os.pathsep) if p]
    return [d for d in dirs if d.is_dir()]


class ConfigModel(SRModel):
    """A model defined by a TOML file."""

    family = "custom"
    config: dict = {}
    config_dir: Path = Path(".")

    @classmethod
    def available(cls):
        fmt = cls.config.get("format", "torchscript")
        need = {"torchscript": "torch", "pytorch": "torch", "onnx": "onnxruntime"}[fmt]
        if importlib.util.find_spec(need) is None:
            return False, f"`{need}` not installed"
        p = cls.config_dir / cls.config["path"]
        if not p.exists():
            return False, f"weights not found: {p}"
        return True, ""

    def load(self):
        cfg, fmt = self.config, self.config.get("format", "torchscript")
        path = self.config_dir / cfg["path"]
        if fmt == "onnx":
            import onnxruntime as ort
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if self.device == "cuda" \
                else ["CPUExecutionProvider"]
            self.sess = ort.InferenceSession(str(path), providers=providers)
            self.inp = self.sess.get_inputs()[0].name
        else:
            import torch
            if fmt == "torchscript":
                self.net = torch.jit.load(str(path), map_location="cpu")
            else:
                mod_name, fn_name = cfg["factory"].split(":")
                spec = importlib.util.spec_from_file_location(mod_name, self.config_dir / f"{mod_name}.py")
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                self.net = getattr(mod, fn_name)()
                state = torch.load(path, map_location="cpu", weights_only=False)
                for k in ("params_ema", "params", "state_dict", "model"):
                    if isinstance(state, dict) and k in state and isinstance(state[k], dict):
                        state = state[k]
                        break
                self.net.load_state_dict(state)
            self.net = self.net.eval().to(self.device)

    def _run(self, patch: np.ndarray) -> np.ndarray:
        x = (patch * float(self.config.get("input_scale", 1.0))).astype(np.float32)[None]
        if hasattr(self, "sess"):
            y = self.sess.run(None, {self.inp: x})[0][0]
        else:
            import torch
            with torch.no_grad():
                y = self.net(torch.from_numpy(x).to(self.device))[0].float().cpu().numpy()
        return y / float(self.config.get("output_scale", 1.0))

    def predict(self, x):
        tile = int(self.config.get("tile", 128))
        if tile <= 0:
            return self._run(x).astype(np.float32)
        return tiled_predict(self._run, x, self.grid_factor, tile=tile,
                             overlap=int(self.config.get("overlap", 16)))


def _register_toml(path: Path):
    cfg = tomllib.loads(path.read_text())
    key = cfg.get("key") or f"custom_{path.stem}"
    attrs = dict(
        key=key, name=cfg.get("name", path.stem), description=cfg.get("description", f"From {path.name}"),
        in_bands=list(cfg["in_bands"]), out_bands=list(cfg["out_bands"]),
        in_res=float(cfg.get("in_res", 10)), out_res=float(cfg.get("out_res", 2.5)),
        config=cfg, config_dir=path.parent, family=cfg.get("family", "custom"),
    )
    register(type(f"Config_{key}", (ConfigModel,), attrs))


def discover() -> dict[str, str]:
    """Import every plugin in the model folders. Returns {file: error} for failures."""
    _errors.clear()
    for d in model_dirs():
        for p in sorted(d.glob("*.toml")):
            try:
                _register_toml(p)
            except Exception as e:  # noqa: BLE001 - report, never crash the app
                _errors[str(p)] = f"{type(e).__name__}: {e}"
        for p in sorted(d.glob("*.py")):
            if p.name.startswith("_"):
                continue
            try:
                name = f"s2sr_plugin_{p.stem}"
                spec = importlib.util.spec_from_file_location(name, p)
                mod = importlib.util.module_from_spec(spec)
                sys.modules[name] = mod
                spec.loader.exec_module(mod)
            except Exception as e:  # noqa: BLE001
                _errors[str(p)] = f"{type(e).__name__}: {e}"
    return dict(_errors)
