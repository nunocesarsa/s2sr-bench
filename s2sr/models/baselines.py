"""Non-learned reference methods. Every comparison should include these:
a learned model that cannot beat bicubic on the Wald test is not adding detail."""
from __future__ import annotations

import numpy as np

from ..bands import BANDS_10M, BANDS_20M, MTF_NYQUIST
from ..resample import block_mean, degrade, upsample
from .base import SRModel, register

RGBN = ["B04", "B03", "B02", "B08"]


class _Interp(SRModel):
    family = "baseline"
    method = "bicubic"

    def predict(self, x):
        return upsample(x, self.grid_factor, self.method)


@register
class BicubicX4(_Interp):
    key = "bicubic_x4"
    name = "Bicubic ×4 (RGBN)"
    description = "Bicubic interpolation of B04/B03/B02/B08 from 10 m to 2.5 m. The floor every model must beat."
    in_bands, in_res, out_bands, out_res = RGBN, 10.0, RGBN, 2.5


@register
class NearestX4(_Interp):
    key = "nearest_x4"
    name = "Nearest ×4 (RGBN)"
    method = "nearest"
    description = "Pixel replication: shows the raw 10 m sampling at 2.5 m display size."
    in_bands, in_res, out_bands, out_res = RGBN, 10.0, RGBN, 2.5


@register
class Bicubic20to10(SRModel):
    key = "bicubic_20to10"
    name = "Bicubic 20→10 m"
    family = "baseline"
    description = "Bicubic interpolation of the six 20 m bands onto the 10 m grid."
    in_bands, in_res, out_bands, out_res = BANDS_20M, 20.0, BANDS_20M, 10.0

    def predict(self, x):
        return upsample(x, 2, "bicubic")


@register
class HPFSharpen20to10(SRModel):
    """High-pass-filter injection (a classic pansharpening scheme).

    For each 20 m band: bicubic-upsample to 10 m, then add the high-frequency
    detail of a synthetic 10 m 'pan' image, scaled by a regression gain
    estimated at 20 m. The pan is the least-squares combination of the four
    10 m bands that best predicts the 20 m band.
    """

    key = "hpf_20to10"
    name = "HPF injection 20→10 m"
    family = "band sharpening"
    description = ("Classical detail injection: per-band regression on the 10 m bands builds a "
                   "synthetic pan whose high frequencies are added to the bicubic 20 m band.")
    in_bands = BANDS_10M + BANDS_20M
    in_res = 10.0
    out_bands = BANDS_20M
    out_res = 10.0

    def predict(self, x):
        hr = x[:4]                     # 10 m bands on the 10 m grid
        lr_up = x[4:]                  # 20 m bands, already bicubic on 10 m grid
        # 10 m bands as a 20 m sensor would see them (MTF blur + decimation),
        # so pan_low carries the same blur as the real 20 m bands.
        hr20 = np.stack([degrade(b, 2, "mtf", MTF_NYQUIST[n]) for b, n in zip(hr, BANDS_10M)])
        hr_low = upsample(hr20, 2, "bicubic")
        a = hr20.reshape(4, -1).T
        a1 = np.c_[a, np.ones(len(a))]
        out = []
        for i in range(lr_up.shape[0]):
            target = block_mean(lr_up[i], 2).ravel()
            coef, *_ = np.linalg.lstsq(a1, target, rcond=None)
            pan = np.tensordot(coef[:4], hr, 1) + coef[4]
            pan_low = np.tensordot(coef[:4], hr_low, 1) + coef[4]
            out.append(lr_up[i] + (pan - pan_low))
        return np.stack(out).astype(np.float32)
