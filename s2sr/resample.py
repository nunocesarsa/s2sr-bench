"""Grid-aligned resampling and sensor-like degradation.

All functions treat pixels as areas (pixel-is-area), so a 20 m pixel covers
exactly 2x2 10 m pixels and nothing shifts by half a pixel when changing grids.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage

_ORDER = {"nearest": 0, "bilinear": 1, "bicubic": 3}


def upsample(arr: np.ndarray, factor: int, method: str = "bicubic") -> np.ndarray:
    """Upsample a (H, W) or (C, H, W) array by an integer factor."""
    if factor == 1:
        return arr.astype(np.float32, copy=False)
    order = _ORDER[method]
    if arr.ndim == 3:
        return np.stack([upsample(a, factor, method) for a in arr])
    if order == 0:
        return np.repeat(np.repeat(arr, factor, 0), factor, 1).astype(np.float32)
    out = ndimage.zoom(arr.astype(np.float32), factor, order=order,
                       mode="grid-mirror", grid_mode=True, prefilter=True)
    return out.astype(np.float32)


def block_mean(arr: np.ndarray, factor: int) -> np.ndarray:
    """Area-average downsampling by an integer factor (crops any remainder)."""
    if factor == 1:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 3:
        return np.stack([block_mean(a, factor) for a in arr])
    h, w = arr.shape[0] // factor * factor, arr.shape[1] // factor * factor
    a = arr[:h, :w].astype(np.float32)
    return a.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))


def mtf_sigma(factor: int, mtf_nyquist: float) -> float:
    """Gaussian sigma (in fine pixels) whose response equals `mtf_nyquist`
    at the Nyquist frequency of the coarse grid.

    G(f) = exp(-2 pi^2 sigma^2 f^2), coarse Nyquist f = 1 / (2 * factor).
    """
    mtf_nyquist = min(max(mtf_nyquist, 1e-3), 0.999)
    return factor * math.sqrt(-2.0 * math.log(mtf_nyquist)) / math.pi


def degrade(arr: np.ndarray, factor: int, mode: str = "mtf", mtf_nyquist: float = 0.3) -> np.ndarray:
    """Simulate imaging the scene at `factor` times coarser resolution.

    mode="mtf": Gaussian blur matched to the sensor MTF, then decimation at
                block centres (Wald protocol, as used in pansharpening literature).
    mode="area": plain block averaging.
    """
    if factor == 1:
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 3:
        return np.stack([degrade(a, factor, mode, mtf_nyquist) for a in arr])
    if mode == "area":
        return block_mean(arr, factor)
    sigma = mtf_sigma(factor, mtf_nyquist)
    blurred = ndimage.gaussian_filter(arr.astype(np.float32), sigma, mode="mirror")
    # Sample at the centre of each coarse pixel. For even factors the centre
    # falls between fine pixels, so average the central 2x2 neighbourhood.
    h, w = arr.shape[0] // factor * factor, arr.shape[1] // factor * factor
    blurred = blurred[:h, :w]
    if factor % 2 == 1:
        c = factor // 2
        return blurred[c::factor, c::factor]
    c = factor // 2
    return 0.25 * (blurred[c - 1::factor, c - 1::factor] + blurred[c::factor, c - 1::factor]
                   + blurred[c - 1::factor, c::factor] + blurred[c::factor, c::factor])


def to_res(arr: np.ndarray, from_res: float, to_res: float, method: str = "bicubic") -> np.ndarray:
    """Move an array between pixel sizes that are integer multiples of each other."""
    if math.isclose(from_res, to_res):
        return arr.astype(np.float32, copy=False)
    if from_res > to_res:
        f = from_res / to_res
        if not math.isclose(f, round(f)):
            raise ValueError(f"Non-integer resampling factor {f}")
        return upsample(arr, int(round(f)), method)
    f = to_res / from_res
    if not math.isclose(f, round(f)):
        raise ValueError(f"Non-integer resampling factor {f}")
    return block_mean(arr, int(round(f)))
