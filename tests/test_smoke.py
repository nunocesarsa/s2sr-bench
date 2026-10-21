"""Offline smoke tests: run with `pytest -q` (no network or GPU needed)."""
import numpy as np
import pytest

from s2sr.models import registry
from s2sr.models.tiling import tiled_predict
from s2sr.resample import block_mean, degrade, upsample
from s2sr.runner import evaluate
from s2sr.scene import synthetic_scene


@pytest.fixture(scope="module")
def scene():
    return synthetic_scene(256)


def test_scene_shapes(scene):
    assert scene.bands["B02"].shape == (256, 256)
    assert scene.bands["B11"].shape == (128, 128)
    assert scene.get(["B04", "B11"], 10).shape == (2, 256, 256)


def test_resampling_roundtrip():
    a = np.random.default_rng(0).random((32, 32)).astype(np.float32)
    assert np.allclose(block_mean(upsample(a, 4, "nearest"), 4), a)
    assert degrade(a, 4).shape == (8, 8)


def test_tiling_matches_direct():
    x = np.random.default_rng(1).random((3, 100, 140)).astype(np.float32)
    fn = lambda p: upsample(p, 2, "nearest")  # noqa: E731
    y = tiled_predict(fn, x, 2, tile=48, overlap=8)
    assert y.shape == (3, 200, 280)
    assert np.allclose(y, upsample(x, 2, "nearest"), atol=1e-5)


@pytest.mark.parametrize("key", ["bicubic_x4", "nearest_x4", "bicubic_20to10", "hpf_20to10"])
def test_model_runs_and_scores(scene, key):
    model = registry()[key]()
    res = evaluate(model, scene, "both")
    real, wald = res["real"], res["wald"]
    assert real.metrics["summary"]["consistency PSNR"] > 15
    assert wald.metrics["summary"]["PSNR"] > 10
    for b, a in real.bands.items():
        assert a.shape == scene.shape_at(model.out_res)


def test_hpf_beats_bicubic_on_wald(scene):
    reg = registry()
    bic = evaluate(reg["bicubic_20to10"](), scene, "wald")["wald"].metrics["summary"]["PSNR"]
    hpf = evaluate(reg["hpf_20to10"](), scene, "wald")["wald"].metrics["summary"]["PSNR"]
    assert hpf > bic


def test_geotiff_export(scene):
    pytest.importorskip("rasterio")
    from s2sr.export import geotiff_bytes
    r = evaluate(registry()["bicubic_x4"](), scene, "real")["real"]
    assert geotiff_bytes(r)[:2] in (b"II", b"MM")
