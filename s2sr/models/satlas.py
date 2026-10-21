"""Satlas super-resolution ESRGAN (allenai/satlas-super-resolution).

Single-image RGB generator trained on S2-NAIP (USA). The network definition
below mirrors `SSR_RRDBNet` from that repository so no extra package is
needed; only `torch` is required. Weights (~67 MB) download on first use.

Notes on domain:
- Trained on L1C TCI (8-bit true-colour). We emulate TCI from L2A reflectance
  with reflectance / tci_max * 255; `tci_max` (default 0.2) is adjustable.
- Trained on NAIP in the USA, so expect hallucinated structures elsewhere.
"""
from __future__ import annotations

import importlib.util
import urllib.request
from pathlib import Path

import numpy as np

from .base import SRModel, register
from .tiling import tiled_predict

CACHE = Path.home() / ".cache" / "s2sr_bench" / "satlas"
WEIGHTS = {
    "esrgan_1S2": "https://storage.googleapis.com/satlas-satellite-super-resolution/esrgan_1S2.pth",
}


def build_rrdbnet(num_in_ch=3, num_out_ch=3, scale=4, num_feat=64, num_block=23, num_grow_ch=32):
    import torch
    from torch import nn
    from torch.nn import functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, nf, gc):
            super().__init__()
            self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
            self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
            self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, 1, 1)
            self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, 1, 1)
            self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, nf, gc):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(nf, gc)
            self.rdb2 = ResidualDenseBlock(nf, gc)
            self.rdb3 = ResidualDenseBlock(nf, gc)

        def forward(self, x):
            return self.rdb3(self.rdb2(self.rdb1(x))) * 0.2 + x

    class SSR_RRDBNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(*[RRDB(num_feat, num_grow_ch) for _ in range(num_block)])
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(0.2, inplace=True)

        def forward(self, x):
            feat = self.conv_first(x)
            feat = feat + self.conv_body(self.body(feat))
            feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
            feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
            return self.conv_last(self.lrelu(self.conv_hr(feat)))

    assert scale == 4, "only the x4 generator is implemented"
    return SSR_RRDBNet()


def _download(url: str, dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


@register
class SatlasESRGAN(SRModel):
    key = "satlas_esrgan_1s2"
    name = "Satlas ESRGAN ×4 (RGB, 1 image)"
    family = "GAN"
    description = ("Allen AI Satlas ESRGAN trained on Sentinel-2 → NAIP (USA). Sharp, perceptual output; "
                   "prone to hallucinating buildings/vehicles outside its training domain.")
    in_bands, in_res, out_bands, out_res = ["B04", "B03", "B02"], 10.0, ["B04", "B03", "B02"], 2.5
    install_hint = "pip install torch"

    @classmethod
    def available(cls):
        if importlib.util.find_spec("torch") is None:
            return False, "`torch` not installed — pip install torch"
        return True, ""

    def load(self):
        import torch

        path = _download(WEIGHTS["esrgan_1S2"], CACHE / "esrgan_1S2.pth")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
        net = build_rrdbnet()
        net.load_state_dict(state, strict=True)
        self.net = net.eval().to(self.device)

    def predict(self, x):
        import torch

        tci_max = float(self.options.get("tci_max", 0.2))
        tci = np.clip(x / tci_max, 0, 1).astype(np.float32)   # ≈ ESA TCI / 255

        def run(patch):
            with torch.no_grad():
                t = torch.from_numpy(np.ascontiguousarray(patch))[None].to(self.device)
                return self.net(t)[0].clamp(0, 1).float().cpu().numpy()

        y = tiled_predict(run, tci, 4, tile=int(self.options.get("tile", 128)),
                          overlap=int(self.options.get("overlap", 16)))
        # Back to approximate reflectance. Values that were saturated in TCI stay at tci_max.
        return (y * tci_max).astype(np.float32)
