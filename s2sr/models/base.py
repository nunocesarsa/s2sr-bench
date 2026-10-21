"""The model interface every super-resolution method implements."""
from __future__ import annotations

import math

import numpy as np

_REGISTRY: dict[str, type["SRModel"]] = {}


def register(cls: type["SRModel"]) -> type["SRModel"]:
    """Class decorator: make a model selectable in the app and CLI."""
    if not getattr(cls, "key", None):
        raise ValueError(f"{cls.__name__} needs a unique `key`")
    _REGISTRY[cls.key] = cls
    return cls


def registry() -> dict[str, type["SRModel"]]:
    return dict(_REGISTRY)


DEVICE_NOTES: list[str] = []
_CUDA_OK: bool | None = None


def _cuda_works() -> bool:
    """Run a tiny convolution on the GPU. torch.cuda.is_available() can be True
    while kernels still fail (e.g. a PyTorch build that no longer supports the
    card's architecture, or a broken cuDNN). Falls back to native CUDA kernels
    with cuDNN disabled before giving up on the GPU."""
    global _CUDA_OK
    if _CUDA_OK is not None:
        return _CUDA_OK
    import torch

    def probe():
        x = torch.rand(1, 3, 16, 16, device="cuda")
        w = torch.rand(8, 3, 3, 3, device="cuda")
        torch.nn.functional.conv2d(x, w, padding=1).sum().item()

    try:
        probe()
        _CUDA_OK = True
    except Exception as e:  # noqa: BLE001
        try:
            torch.backends.cudnn.enabled = False
            probe()
            _CUDA_OK = True
            DEVICE_NOTES.append(f"cuDNN failed ({e}); using CUDA with cuDNN disabled.")
        except Exception as e2:  # noqa: BLE001
            torch.backends.cudnn.enabled = True
            _CUDA_OK = False
            name = torch.cuda.get_device_name(0)
            DEVICE_NOTES.append(
                f"GPU '{name}' is visible but PyTorch cannot run on it ({e2}). Using CPU. "
                f"This PyTorch build supports: {', '.join(torch.cuda.get_arch_list()) or 'unknown'}.")
    return _CUDA_OK


def auto_device(preferred: str = "auto") -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    if preferred == "cuda" or (preferred == "auto" and torch.cuda.is_available()):
        if torch.cuda.is_available() and _cuda_works():
            return "cuda"
        if preferred == "cuda" and not torch.cuda.is_available():
            DEVICE_NOTES.append("CUDA requested but not available; using CPU.")
        if preferred == "cuda":
            return "cpu"
    if preferred in ("auto", "mps") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu" if preferred in ("auto", "cuda", "mps") else preferred


class SRModel:
    """Base class.

    Subclasses set the class attributes below and implement `predict`.

    in_bands / in_res : bands the model consumes, all resampled to `in_res`
                        metres (bicubic for bands coarser than `in_res`).
    out_bands / out_res: bands it returns, on a grid of `out_res` metres.

    `predict` receives float32 surface reflectance (~0..1) shaped
    (len(in_bands), H, W) and must return (len(out_bands), H*k, W*k) with
    k = in_res / out_res, also as reflectance.
    """

    key: str = ""
    name: str = ""
    family: str = "other"
    description: str = ""
    in_bands: list[str] = []
    in_res: float = 10.0
    out_bands: list[str] = []
    out_res: float = 2.5
    # Extra install hint shown when `available()` fails.
    install_hint: str = ""

    def __init__(self, device: str = "auto", **options):
        self.device = auto_device(device)
        self.options = options
        self._loaded = False

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def available(cls) -> tuple[bool, str]:
        """Return (ok, reason). Override to check optional dependencies."""
        return True, ""

    def load(self) -> None:
        """Download weights / build the network. Called once before predict."""
        self._loaded = True

    def ensure_loaded(self):
        if not self._loaded:
            self.load()
            self._loaded = True

    def predict(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    # ---------------------------------------------------------------- helpers
    @property
    def grid_factor(self) -> int:
        f = self.in_res / self.out_res
        if not math.isclose(f, round(f)):
            raise ValueError(f"{self.key}: in_res/out_res must be an integer")
        return int(round(f))

    def wald_factor(self, native_res: dict[str, float]) -> int:
        """Degradation factor for the reference (Wald) test: the smallest
        native-to-output ratio among the output bands, i.e. the scale the model
        is meant to add to its sharpest output band."""
        ratios = [native_res.get(b, self.in_res) / self.out_res for b in self.out_bands]
        return max(1, int(round(min(ratios))))

    def __repr__(self) -> str:
        return f"<{self.key}: {self.in_res}m {self.in_bands} -> {self.out_res}m {self.out_bands}>"
