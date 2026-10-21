"""ESA OpenSR SEN2SR models (https://github.com/ESAOpenSR/SEN2SR).

Weights are fetched with `mlstac` from huggingface.co/tacofoundation/sen2sr on
first use and cached under ~/.cache/s2sr_bench/sen2sr/.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from ..bands import ALL_BANDS, BANDS_20M
from .base import SRModel, register
from .tiling import tiled_predict

CACHE = Path.home() / ".cache" / "s2sr_bench" / "sen2sr"
HF = "https://huggingface.co/tacofoundation/sen2sr/resolve/main"
RGBN = ["B04", "B03", "B02", "B08"]


class _SEN2SRBase(SRModel):
    family = "SEN2SR"
    mlm_path: str = ""          # path under the HF repo
    patch = 128                 # training patch size
    overlap = 32
    install_hint = "pip install sen2sr mlstac"

    @classmethod
    def available(cls):
        for mod in ("torch", "mlstac", "sen2sr"):
            if importlib.util.find_spec(mod) is None:
                return False, f"`{mod}` not installed — {cls.install_hint}"
        return True, ""

    def load(self):
        import mlstac
        import torch

        out_dir = CACHE / self.key
        if not (out_dir / "mlm.json").exists():
            out_dir.mkdir(parents=True, exist_ok=True)
            mlstac.download(file=f"{HF}/{self.mlm_path}", output_dir=str(out_dir))
        dev = torch.device(self.device if self.device != "mps" else "cpu")
        self._torch_device = dev
        self.model = mlstac.load(str(out_dir)).compiled_model(device=dev)
        try:
            self.model = self.model.to(dev)
        except AttributeError:
            pass

    def predict(self, x):
        import torch

        # SEN2SR is trained on 128 px patches. sen2sr.predict_large assumes square
        # inputs, so use our own overlapping tiler (handles any shape, feathered seams).
        idx = None

        def run(patch):
            nonlocal idx
            X = torch.from_numpy(np.ascontiguousarray(patch)).to(self._torch_device)
            with torch.no_grad():
                y = self.model(X[None]).squeeze(0).float().cpu().numpy()
            # Some variants return every input band; keep only the declared outputs.
            if y.shape[0] != len(self.out_bands) and y.shape[0] == len(self.in_bands):
                idx = idx or [self.in_bands.index(b) for b in self.out_bands]
                y = y[idx]
            return y

        x = np.nan_to_num(x).astype(np.float32)
        return tiled_predict(run, x, self.grid_factor, tile=self.patch, overlap=self.overlap // 2)


@register
class SEN2SRLiteRGBN(_SEN2SRBase):
    key = "sen2sr_lite_rgbn_x4"
    name = "SEN2SR-Lite RGBN ×4"
    description = "Lightweight CNN, 10 m RGB+NIR → 2.5 m, with a hard radiometric-consistency constraint."
    mlm_path = "SEN2SRLite/NonReference_RGBN_x4/mlm.json"
    in_bands, in_res, out_bands, out_res = RGBN, 10.0, RGBN, 2.5


@register
class SEN2SRLiteFull(_SEN2SRBase):
    key = "sen2sr_lite_full"
    name = "SEN2SR-Lite 10 bands → 2.5 m"
    description = "Full pipeline: RGBN ×4 plus 20 m bands sharpened to 10 m and then to 2.5 m."
    mlm_path = "SEN2SRLite/main/mlm.json"
    in_bands, in_res, out_bands, out_res = ALL_BANDS, 10.0, ALL_BANDS, 2.5


@register
class SEN2SRLiteRSWIR(_SEN2SRBase):
    key = "sen2sr_lite_rswir_x2"
    name = "SEN2SR-Lite 20→10 m sharpening"
    family = "band sharpening"
    description = "Reference-guided sharpening of B05, B06, B07, B8A, B11, B12 to 10 m using all 10 bands."
    mlm_path = "SEN2SRLite/Reference_RSWIR_x2/mlm.json"
    in_bands, in_res, out_bands, out_res = ALL_BANDS, 10.0, BANDS_20M, 10.0
