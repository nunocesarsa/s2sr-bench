"""Search and read Sentinel-2 L2A subsets from public STAC catalogues.

Only the requested window is read (Cloud-Optimised GeoTIFFs over HTTP), so a
5 km x 5 km subset is a few MB rather than a full 800 MB tile.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date

import numpy as np

from ..bands import (ALL_BANDS, EARTH_SEARCH_ASSETS, PLANETARY_COMPUTER_ASSETS,
                     NATIVE_RES, SCL_CLOUDY)
from ..scene import Scene

CATALOGS = {
    "Earth Search (AWS)": {
        "url": "https://earth-search.aws.element84.com/v1",
        "collection": "sentinel-2-l2a",
        "assets": EARTH_SEARCH_ASSETS,
        "sign": False,
    },
    "Planetary Computer": {
        "url": "https://planetarycomputer.microsoft.com/api/stac/v1",
        "collection": "sentinel-2-l2a",
        "assets": PLANETARY_COMPUTER_ASSETS,
        "sign": True,
    },
}

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF,.tiff,.jp2",
    "GDAL_HTTP_MULTIRANGE": "YES",
    "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    "VSI_CACHE": "TRUE",
}


@dataclass
class SearchHit:
    id: str
    datetime: str
    cloud_cover: float
    item: object  # pystac.Item

    @property
    def label(self) -> str:
        return f"{self.datetime[:10]}  ·  {self.cloud_cover:5.1f}% cloud  ·  {self.id}"


def search(lat: float, lon: float, start: date, end: date, max_cloud: float = 30,
           catalog: str = "Earth Search (AWS)", limit: int = 50) -> list[SearchHit]:
    from pystac_client import Client

    cfg = CATALOGS[catalog]
    modifier = None
    if cfg["sign"]:
        import planetary_computer
        modifier = planetary_computer.sign_inplace
    client = Client.open(cfg["url"], modifier=modifier)
    s = client.search(
        collections=[cfg["collection"]],
        intersects={"type": "Point", "coordinates": [lon, lat]},
        datetime=f"{start.isoformat()}/{end.isoformat()}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
        max_items=limit,
    )
    hits = []
    for it in s.items():
        hits.append(SearchHit(it.id, str(it.properties.get("datetime", "")),
                              float(it.properties.get("eo:cloud_cover", float("nan"))), it))
    hits.sort(key=lambda h: (h.cloud_cover, h.datetime))
    return hits


def _reflectance_offset(item, catalog: str) -> float:
    """DN offset to add before dividing by 10000.

    Processing baseline >= 04.00 (25 Jan 2022 onwards) stores DN + 1000.
    Earth Search v1 may already have removed it (earthsearch:boa_offset_applied).
    """
    p = item.properties
    if p.get("earthsearch:boa_offset_applied") is True:
        return 0.0
    baseline = p.get("s2:processing_baseline") or p.get("processing:version")
    try:
        if baseline is not None and float(str(baseline)) >= 4.0:
            return -1000.0
    except ValueError:
        pass
    if baseline is None:
        dt = str(p.get("datetime", ""))[:10]
        if dt >= "2022-01-25":
            return -1000.0
    return 0.0


def read_subset(hit: SearchHit, lat: float, lon: float, size_px: int = 512,
                catalog: str = "Earth Search (AWS)", bands=None, progress=None) -> Scene:
    """Read a `size_px` x `size_px` (10 m pixels) window centred on lat/lon."""
    import rasterio
    from rasterio.warp import transform as warp_transform
    from rasterio.windows import Window

    bands = bands or ALL_BANDS
    cfg = CATALOGS[catalog]
    item = hit.item
    if cfg["sign"]:
        import planetary_computer
        item = planetary_computer.sign(item)
    assets = cfg["assets"]
    size_px = int(math.ceil(size_px / 16) * 16)
    offset = _reflectance_offset(item, catalog)

    out, res = {}, {}
    crs = transform10 = None
    scl_clear = None
    with rasterio.Env(**GDAL_ENV):
        # Anchor the window on the 20 m grid so 10 m and 20 m windows nest exactly.
        ref_href = item.assets[assets["B05"]].href
        with rasterio.open(ref_href) as ds20:
            crs = ds20.crs
            xs, ys = warp_transform("EPSG:4326", crs, [lon], [lat])
            col_c, row_c = ~ds20.transform * (xs[0], ys[0])
            half20 = size_px // 4
            col0 = int(round(col_c)) - half20
            row0 = int(round(row_c)) - half20
            col0 = max(0, min(col0, ds20.width - 2 * half20))
            row0 = max(0, min(row0, ds20.height - 2 * half20))
            t20 = ds20.transform
        from affine import Affine
        x0, y0 = t20 * (col0, row0)
        transform10 = Affine(10.0, 0, x0, 0, -10.0, y0)

        todo = list(bands) + (["SCL"] if "SCL" in assets and assets["SCL"] in item.assets else [])
        for i, b in enumerate(todo):
            if progress:
                progress(i / len(todo), f"Reading {b}")
            href = item.assets[assets[b]].href
            nat = NATIVE_RES.get(b, 20)
            k = 20 // nat
            n = size_px * 10 // nat
            win = Window(col0 * k, row0 * k, n, n)
            with rasterio.open(href) as ds:
                arr = ds.read(1, window=win, boundless=True, fill_value=0)
            if b == "SCL":
                scl_clear = ~np.isin(arr, list(SCL_CLOUDY))
                continue
            refl = (arr.astype(np.float32) + offset) / 10000.0
            refl[arr == 0] = np.nan
            out[b] = refl
            res[b] = float(nat)

    nodata = sum(int(np.isnan(a).sum()) for a in out.values())
    for b in out:
        out[b] = np.nan_to_num(np.clip(out[b], 0, 1.5), nan=0.0)
    meta = {
        "id": hit.id, "datetime": hit.datetime, "tile_cloud_cover": hit.cloud_cover,
        "source": catalog, "lat": lat, "lon": lon, "offset_applied": offset,
        "nodata_pixels": nodata,
    }
    if scl_clear is not None:
        meta["aoi_clear_fraction"] = float(scl_clear.mean())
    if progress:
        progress(1.0, "Done")
    return Scene(out, res, crs, transform10, meta).crop_to_multiple(16)
