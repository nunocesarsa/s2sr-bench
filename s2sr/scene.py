"""In-memory representation of a Sentinel-2 subset."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .bands import ALL_BANDS, MTF_NYQUIST, NATIVE_RES, sort_bands
from .resample import degrade, to_res


@dataclass
class Scene:
    """Surface reflectance bands (float32, ~0..1) kept at their own pixel size.

    `bands[b]` is a (H_b, W_b) array whose pixel size is `res[b]` metres.
    `transform` is the affine transform of the 10 m grid (rasterio Affine) and
    may be None for synthetic data.
    """

    bands: dict[str, np.ndarray]
    res: dict[str, float]
    crs: Any = None
    transform: Any = None
    meta: dict = field(default_factory=dict)

    # ------------------------------------------------------------------ info
    @property
    def band_names(self) -> list[str]:
        return sort_bands(self.bands)

    @property
    def base_res(self) -> float:
        return min(self.res.values())

    def shape_at(self, res: float) -> tuple[int, int]:
        b = self.band_names[0]
        h, w = self.bands[b].shape
        k = self.res[b] / res
        return int(round(h * k)), int(round(w * k))

    def has(self, bands) -> bool:
        return all(b in self.bands for b in bands)

    # ------------------------------------------------------------ accessors
    def get(self, bands: list[str], res: float, method: str = "bicubic") -> np.ndarray:
        """Stack `bands` resampled onto a grid with pixel size `res`."""
        missing = [b for b in bands if b not in self.bands]
        if missing:
            raise KeyError(f"Scene lacks bands {missing}")
        return np.stack([to_res(self.bands[b], self.res[b], res, method) for b in bands]).astype(np.float32)

    # ------------------------------------------------------------ transforms
    def crop_to_multiple(self, multiple: int = 16) -> "Scene":
        """Crop so the 10 m grid is a multiple of `multiple` (keeps 20 m aligned)."""
        h, w = self.shape_at(self.base_res)
        h2, w2 = h // multiple * multiple, w // multiple * multiple
        out = {}
        for b, a in self.bands.items():
            k = self.res[b] / self.base_res
            out[b] = a[: int(h2 / k), : int(w2 / k)]
        return Scene(out, dict(self.res), self.crs, self.transform, dict(self.meta))

    def crop_window(self, row: int, col: int, size: int) -> "Scene":
        """Crop a square window given in base-resolution pixels."""
        out = {}
        for b, a in self.bands.items():
            k = int(round(self.res[b] / self.base_res))
            out[b] = a[row // k:(row + size) // k, col // k:(col + size) // k]
        tr = self.transform
        if tr is not None:
            tr = tr * tr.translation(col, row)
        return Scene(out, dict(self.res), self.crs, tr, dict(self.meta))

    def degraded(self, factor: int, mode: str = "mtf") -> "Scene":
        """Wald protocol: every band imaged `factor` times coarser.

        The nominal `res` labels are kept so models see data that "looks like"
        ordinary Sentinel-2, while the true pixel size is res * factor.
        """
        out = {b: degrade(a, factor, mode, MTF_NYQUIST.get(b, 0.3)) for b, a in self.bands.items()}
        tr = self.transform
        if tr is not None:
            tr = tr * tr.scale(factor)
        meta = dict(self.meta, wald_factor=factor, wald_mode=mode)
        return Scene(out, dict(self.res), self.crs, tr, meta)

    @classmethod
    def from_stack(cls, stack: np.ndarray, band_names: list[str], res: float = 10,
                   derive_native: bool = True, **kw) -> "Scene":
        """Build a scene from a (C, H, W) stack where every band sits on one grid.

        With `derive_native`, 20 m bands that were resampled to the finer grid are
        block-averaged back to their native 20 m so models and metrics see the
        real sampling.
        """
        bands, res_map = {}, {}
        for i, b in enumerate(band_names):
            a = stack[i].astype(np.float32)
            nat = NATIVE_RES.get(b, res)
            if derive_native and nat > res:
                a = to_res(a, res, nat)
                res_map[b] = nat
            else:
                res_map[b] = res
            bands[b] = a
        return cls(bands, res_map, **kw)


def synthetic_scene(size: int = 256, seed: int = 0) -> Scene:
    """A smooth-plus-edges synthetic scene for tests and the demo button."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    base = 0.08 + 0.05 * np.sin(6 * np.pi * xx) * np.cos(4 * np.pi * yy)
    fields = ((xx * 8).astype(int) + (yy * 8).astype(int)) % 3 * 0.04
    roads = ((np.abs((xx * size) % 64 - 32) < 1.5) | (np.abs((yy * size) % 80 - 40) < 1.5)) * 0.1
    noise = rng.normal(0, 0.004, (size, size))
    signatures = {  # rough vegetation-like spectral shape
        "B02": 0.6, "B03": 0.8, "B04": 0.7, "B05": 1.2, "B06": 2.0,
        "B07": 2.4, "B08": 2.6, "B8A": 2.6, "B11": 1.8, "B12": 1.1,
    }
    stack = []
    for b in ALL_BANDS:
        s = signatures[b]
        stack.append(np.clip(base * s + fields * (s - 1.0) * 0.5 + roads + noise, 0, 1))
    sc = Scene.from_stack(np.stack(stack), ALL_BANDS, 10, meta={"id": "synthetic", "source": "synthetic"})
    return sc.crop_to_multiple(16)
