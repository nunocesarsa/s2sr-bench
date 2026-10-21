"""Template plugin. Files starting with "_" are NOT loaded, so this one is hidden.
To add your own model, copy it to e.g. my_model.py (no leading underscore) and edit.

Any other .py file in this folder is imported at startup.

Subclass SRModel, set the band/resolution contract, implement `predict`,
and decorate with @register. """
import numpy as np

from s2sr.models.base import SRModel, register
from s2sr.resample import upsample


@register
class UnsharpBicubicX4(SRModel):
    key = "example_unsharp_x4"
    name = "Example: bicubic + unsharp mask ×4"
    family = "example plugin"
    description = "Template plugin — bicubic ×4 followed by an unsharp mask. Copy this file to add your own model."
    in_bands = ["B04", "B03", "B02", "B08"]
    in_res = 10.0
    out_bands = ["B04", "B03", "B02", "B08"]
    out_res = 2.5

    def load(self):
        # Load weights here (self.device is "cuda", "mps" or "cpu").
        # e.g. self.net = torch.jit.load("my_model.pt").to(self.device).eval()
        self.amount = float(self.options.get("amount", 0.6))

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x: (4, H, W) reflectance at 10 m  ->  return (4, 4H, 4W) reflectance at 2.5 m
        from scipy.ndimage import gaussian_filter
        up = upsample(x, self.grid_factor, "bicubic")
        blur = np.stack([gaussian_filter(b, 2.0) for b in up])
        return (up + self.amount * (up - blur)).astype(np.float32)
