"""Write super-resolved outputs as GeoTIFF."""
from __future__ import annotations

import io

import numpy as np

from .runner import SRResult


def result_transform(result: SRResult):
    sc = result.input_scene
    if sc.transform is None:
        return None
    # input_scene.transform is the (possibly degraded) 10 m-nominal grid.
    k = result.out_res / sc.base_res
    return sc.transform * sc.transform.scale(k)


def geotiff_bytes(result: SRResult) -> bytes:
    import rasterio
    from rasterio.io import MemoryFile

    names = list(result.bands)
    arr = np.stack([result.bands[b] for b in names]).astype(np.float32)
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": arr.shape[0],
        "height": arr.shape[1], "width": arr.shape[2], "compress": "deflate",
        "tiled": True, "blockxsize": 256, "blockysize": 256,
    }
    tr = result_transform(result)
    if tr is not None:
        profile.update(transform=tr, crs=result.input_scene.crs)
    with MemoryFile() as mem:
        with mem.open(**profile) as dst:
            dst.write(arr)
            for i, b in enumerate(names, 1):
                dst.set_band_description(i, b)
            dst.update_tags(model=result.model_key, pixel_size_m=str(result.true_res),
                            units="surface reflectance")
        return mem.read()


def write_geotiff(result: SRResult, path) -> None:
    with open(path, "wb") as f:
        f.write(geotiff_bytes(result))


def png_bytes(img: np.ndarray) -> bytes:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return buf.getvalue()
