"""Quality metrics.

Two complementary evaluations, because real Sentinel-2 has no 2.5 m truth:

1. Reference (Wald protocol): degrade the scene by the model's factor, super-
   resolve the degraded scene, compare with the original. Measures how well
   a model restores detail it has *seen the statistics of*; the scale-
   invariance assumption is the protocol's known weakness.

2. No-reference on the real scene:
   - consistency: the SR output, degraded back to each band's native pixel
     size, should reproduce the input (radiometric fidelity / no invented
     signal at the observed scale),
   - spectral distortion vs. bicubic (SAM), and
   - high-frequency gain: how much detail was added beyond bicubic.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage
from skimage.metrics import structural_similarity

from .bands import MTF_NYQUIST
from .resample import block_mean, degrade, upsample


def _crop(a: np.ndarray, border: int) -> np.ndarray:
    return a[..., border:a.shape[-2] - border, border:a.shape[-1] - border] if border else a


def _match(a, b):
    h, w = min(a.shape[-2], b.shape[-2]), min(a.shape[-1], b.shape[-1])
    return a[..., :h, :w], b[..., :h, :w]


def psnr(ref: np.ndarray, est: np.ndarray, data_range: float | None = None) -> float:
    mse = float(np.mean((ref.astype(np.float64) - est) ** 2))
    if mse == 0:
        return float("inf")
    dr = data_range if data_range else float(np.percentile(ref, 99.5) - np.percentile(ref, 0.5)) or 1.0
    return 10 * math.log10(dr ** 2 / mse)


def ssim(ref: np.ndarray, est: np.ndarray, data_range: float | None = None) -> float:
    dr = data_range if data_range else float(np.percentile(ref, 99.5) - np.percentile(ref, 0.5)) or 1.0
    win = min(7, (min(ref.shape) // 2) * 2 - 1)
    if win < 3:
        return float("nan")
    return float(structural_similarity(ref.astype(np.float64), est.astype(np.float64),
                                       data_range=dr, win_size=win))


def sam_deg(ref: np.ndarray, est: np.ndarray) -> float:
    """Mean spectral angle (degrees) between (C, H, W) stacks."""
    if ref.shape[0] < 2:
        return float("nan")
    num = (ref * est).sum(0)
    den = np.linalg.norm(ref, axis=0) * np.linalg.norm(est, axis=0)
    valid = den > 1e-8
    ang = np.arccos(np.clip(num[valid] / den[valid], -1, 1))
    return float(np.degrees(ang.mean())) if ang.size else float("nan")


def ergas(ref: np.ndarray, est: np.ndarray, ratio: float) -> float:
    """ERGAS; `ratio` = high-res pixel size / low-res pixel size (e.g. 1/4)."""
    terms = []
    for r, e in zip(ref, est):
        mu = float(r.mean())
        if mu > 1e-6:
            terms.append((np.sqrt(np.mean((r - e) ** 2)) / mu) ** 2)
    return float(100 * ratio * np.sqrt(np.mean(terms))) if terms else float("nan")


def hf_energy(a: np.ndarray) -> float:
    """Mean squared Laplacian — a simple measure of high-frequency content."""
    return float(np.mean(ndimage.laplace(a.astype(np.float32)) ** 2))


# --------------------------------------------------------------------------
def reference_metrics(sr: dict[str, np.ndarray], ref: dict[str, np.ndarray],
                      grid_ratio: float, border: int = 4) -> dict:
    """Compare SR bands against reference bands on the same grid.

    Returns {"per_band": {band: {...}}, "summary": {...}}.
    """
    per_band, sr_stack, ref_stack = {}, [], []
    for b, est in sr.items():
        r = ref.get(b)
        if r is None:
            per_band[b] = {"note": "no reference at this resolution"}
            continue
        r, e = _match(r, est)
        r, e = _crop(r, border), _crop(e, border)
        dr = float(np.percentile(r, 99.5) - np.percentile(r, 0.5)) or 1.0
        per_band[b] = {
            "PSNR": psnr(r, e, dr), "SSIM": ssim(r, e, dr),
            "RMSE": float(np.sqrt(np.mean((r - e) ** 2))), "bias": float(np.mean(e - r)),
        }
        sr_stack.append(e)
        ref_stack.append(r)
    summary = {}
    if ref_stack:
        R, E = np.stack(ref_stack), np.stack(sr_stack)
        vals = [v for v in per_band.values() if "PSNR" in v]
        summary = {
            "PSNR": float(np.mean([v["PSNR"] for v in vals])),
            "SSIM": float(np.nanmean([v["SSIM"] for v in vals])),
            "RMSE": float(np.mean([v["RMSE"] for v in vals])),
            "SAM°": sam_deg(R, E),
            "ERGAS": ergas(R, E, grid_ratio),
            "bands scored": len(vals),
        }
    return {"per_band": per_band, "summary": summary}


def consistency_metrics(sr: dict[str, np.ndarray], out_res: float, inputs: dict[str, np.ndarray],
                        input_res: dict[str, float], degrade_mode: str = "area", border: int = 2) -> dict:
    """Degrade each SR band to its native pixel size and compare with the input band,
    plus spectral angle and high-frequency gain relative to bicubic."""
    per_band, sr_stack, bic_stack = {}, [], []
    for b, est in sr.items():
        if b not in inputs:
            continue
        f = input_res[b] / out_res
        if f < 1 or not math.isclose(f, round(f)):
            continue
        f = int(round(f))
        back = degrade(est, f, degrade_mode, MTF_NYQUIST.get(b, 0.3)) if f > 1 else est
        inp, back = _match(inputs[b], back)
        inp, back = _crop(inp, border), _crop(back, border)
        dr = float(np.percentile(inp, 99.5) - np.percentile(inp, 0.5)) or 1.0
        bic = upsample(inputs[b], f, "bicubic") if f > 1 else inputs[b]
        bic, e = _match(bic, est)
        per_band[b] = {
            "consistency PSNR": psnr(inp, back, dr),
            "consistency bias": float(np.mean(back - inp)),
            "corr": float(np.corrcoef(inp.ravel(), back.ravel())[0, 1]) if inp.std() > 0 else float("nan"),
            "HF gain": (hf_energy(e) / hf_energy(bic)) if hf_energy(bic) > 0 else float("nan"),
        }
        sr_stack.append(e)
        bic_stack.append(bic)
    summary = {}
    if per_band:
        vals = list(per_band.values())
        summary = {
            "consistency PSNR": float(np.mean([v["consistency PSNR"] for v in vals])),
            "|bias|": float(np.mean([abs(v["consistency bias"]) for v in vals])),
            "corr": float(np.nanmean([v["corr"] for v in vals])),
            "HF gain": float(np.nanmean([v["HF gain"] for v in vals])),
            "SAM° vs bicubic": sam_deg(np.stack(bic_stack), np.stack(sr_stack)) if len(sr_stack) > 1 else float("nan"),
        }
    return {"per_band": per_band, "summary": summary}


def reference_bands(original: dict[str, np.ndarray], original_res: dict[str, float],
                    bands: list[str], target_res: float) -> dict[str, np.ndarray]:
    """Original bands brought to `target_res` — only where that is not finer than native."""
    out = {}
    for b in bands:
        if b not in original:
            continue
        r = original_res[b]
        if target_res < r - 1e-6:
            continue  # would need upsampling: no valid reference
        f = int(round(target_res / r))
        out[b] = block_mean(original[b], f) if f > 1 else original[b]
    return out
