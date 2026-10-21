"""Run models on a scene and score them."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .metrics import consistency_metrics, reference_bands, reference_metrics
from .models.base import SRModel
from .scene import Scene


@dataclass
class SRResult:
    model_key: str
    model_name: str
    bands: dict[str, np.ndarray]     # output bands on the out_res grid
    out_res: float                    # nominal pixel size of the output (m)
    true_res: float                   # actual pixel size (differs in Wald mode)
    runtime_s: float
    input_scene: Scene                # what the model actually saw
    metrics: dict = field(default_factory=dict)
    error: str | None = None


def run_model(model: SRModel, scene: Scene) -> SRResult:
    model.ensure_loaded()
    x = scene.get(model.in_bands, model.in_res)
    t0 = time.perf_counter()
    y = model.predict(x)
    dt = time.perf_counter() - t0
    k = model.grid_factor
    exp = (len(model.out_bands), x.shape[1] * k, x.shape[2] * k)
    if y.shape != exp:
        raise ValueError(f"{model.key} returned {y.shape}, expected {exp}")
    factor = scene.meta.get("wald_factor", 1)
    return SRResult(model.key, model.name, {b: y[i] for i, b in enumerate(model.out_bands)},
                    model.out_res, model.out_res * factor, dt, scene)


def evaluate(model: SRModel, scene: Scene, mode: str = "both", degrade_mode: str = "mtf",
             border: int = 4) -> dict[str, SRResult]:
    """Run `model` and compute metrics.

    mode: "real" (no-reference on the actual scene), "wald" (reference test on a
    degraded copy), or "both". Returns {"real": SRResult, "wald": SRResult}.
    """
    out = {}
    if mode in ("real", "both"):
        r = run_model(model, scene)
        r.metrics = consistency_metrics(r.bands, r.out_res, scene.bands, scene.res, degrade_mode)
        out["real"] = r
    if mode in ("wald", "both"):
        f = model.wald_factor(scene.res)
        low = scene.degraded(f, degrade_mode)
        r = run_model(model, low)
        refs = reference_bands(scene.bands, scene.res, model.out_bands, r.true_res)
        r.metrics = reference_metrics(r.bands, refs, 1.0 / f, border)
        r.metrics["wald_factor"] = f
        r.metrics["reference_res"] = r.true_res
        out["wald"] = r
    return out
