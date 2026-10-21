---
tags: [s2sr-bench, models, development]
---

# Adding a model

Back to [Home](Home.md) · related: [Models](Models.md), [Architecture](Architecture.md)

Models are discovered at startup from `custom_models/` and from any folder listed in the environment variable
`S2SR_MODEL_DIRS` (colon-separated). Restart the app after adding one; a broken plugin is reported in the sidebar
under "⚠️ Plugin errors" instead of crashing the app.

## The contract

Whatever the model, the app gives it surface reflectance (≈ 0–1):

```
input : float32 (len(in_bands), H, W)      all bands on the in_res grid
output: float32 (len(out_bands), H*k, W*k)  reflectance, k = in_res / out_res (integer)
```

Bands coarser than `in_res` are bicubic-resampled onto the input grid (that is how SEN2SR receives 20 m bands at
10 m). Band names: `B02 B03 B04 B05 B06 B07 B08 B8A B11 B12`.

## Option A: a config file (no code)

For a TorchScript, ONNX or PyTorch `state_dict` model. Copy `custom_models/my_model.toml.example` to e.g.
`custom_models/my_model.toml`:

```toml
name = "My ESRGAN"
format = "torchscript"            # torchscript | onnx | pytorch
path = "weights/my_model.pt"      # relative to the .toml
in_bands  = ["B04", "B03", "B02", "B08"]
out_bands = ["B04", "B03", "B02", "B08"]
in_res = 10
out_res = 2.5
input_scale = 1.0     # model input  = reflectance * input_scale (10000 for DN-trained models)
output_scale = 1.0    # reflectance  = model output / output_scale
tile = 128            # patch size, 0 = whole image
overlap = 16
# format = "pytorch" also needs: factory = "my_arch:build_model"  (my_arch.py next to the .toml)
```

The model appears as `custom_my_model`. Tiling, device placement and batching are handled for you.

## Option B: a Python plugin

For anything needing custom pre/post-processing. Copy `custom_models/_example_plugin.py` to a name **without**
the leading underscore (files starting with `_` are ignored):

```python
import numpy as np
from s2sr.models.base import SRModel, register
from s2sr.models.tiling import tiled_predict

@register
class MyModel(SRModel):
    key = "my_model"                      # unique id (CLI --models)
    name = "My model ×4"                  # shown in the app
    family = "GAN"
    description = "What it is and what it was trained on"
    in_bands = ["B04", "B03", "B02"]
    in_res = 10.0
    out_bands = ["B04", "B03", "B02"]
    out_res = 2.5

    @classmethod
    def available(cls):                   # optional: check dependencies
        return True, ""

    def load(self):                       # once; self.device is "cuda" or "cpu"
        import torch
        self.net = torch.jit.load("custom_models/weights/my.pt").to(self.device).eval()

    def predict(self, x):
        import torch
        def run(patch):
            with torch.no_grad():
                t = torch.from_numpy(np.ascontiguousarray(patch))[None].to(self.device)
                return self.net(t)[0].cpu().numpy()
        return tiled_predict(run, x, self.grid_factor, tile=128, overlap=16)
```

`self.options` holds any options passed from the UI or CLI. After restarting, the model is listed in the sidebar
and in `python -m s2sr.cli models`, and is scored like every other model.

## Checklist

- [ ] Output shape exactly `(len(out_bands), H*k, W*k)`; the runner checks it.
- [ ] Output in reflectance, not DN or 0–255.
- [ ] Run with its baseline and check the Wald test ([Evaluation](Evaluation.md)).
- [ ] Note in `description` what data it was trained on: users need that to judge hallucinations.
