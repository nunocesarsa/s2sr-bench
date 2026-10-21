"""Sentinel-2 band metadata used across the app."""
from __future__ import annotations

# Native ground sampling distance (m) of the bands this app works with.
NATIVE_RES: dict[str, int] = {
    "B02": 10, "B03": 10, "B04": 10, "B08": 10,
    "B05": 20, "B06": 20, "B07": 20, "B8A": 20, "B11": 20, "B12": 20,
}

BANDS_10M = ["B02", "B03", "B04", "B08"]
BANDS_20M = ["B05", "B06", "B07", "B8A", "B11", "B12"]
# Canonical order (same as SEN2SR / cubo examples)
ALL_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

CENTRAL_WAVELENGTH_NM: dict[str, int] = {
    "B02": 490, "B03": 560, "B04": 665, "B05": 705, "B06": 740,
    "B07": 783, "B08": 842, "B8A": 865, "B11": 1610, "B12": 2190,
}

# Approximate pre-flight MTF at Nyquist for S2A (ESA S2 MPC data quality reports).
# Used to build a sensor-like Gaussian blur for the Wald degradation protocol.
MTF_NYQUIST: dict[str, float] = {
    "B02": 0.304, "B03": 0.276, "B04": 0.233, "B08": 0.222,
    "B05": 0.343, "B06": 0.336, "B07": 0.338, "B8A": 0.267,
    "B11": 0.334, "B12": 0.317,
}

# Asset keys used by different STAC catalogues.
EARTH_SEARCH_ASSETS: dict[str, str] = {
    "B02": "blue", "B03": "green", "B04": "red", "B08": "nir",
    "B05": "rededge1", "B06": "rededge2", "B07": "rededge3", "B8A": "nir08",
    "B11": "swir16", "B12": "swir22", "SCL": "scl",
}
PLANETARY_COMPUTER_ASSETS: dict[str, str] = {b: b for b in ALL_BANDS} | {"SCL": "SCL"}

COMPOSITES: dict[str, list[str]] = {
    "True colour (B04, B03, B02)": ["B04", "B03", "B02"],
    "False colour NIR (B08, B04, B03)": ["B08", "B04", "B03"],
    "Red-edge (B07, B05, B04)": ["B07", "B05", "B04"],
    "SWIR (B12, B8A, B04)": ["B12", "B8A", "B04"],
    "Agriculture (B11, B08, B02)": ["B11", "B08", "B02"],
    "20 m bands only (B11, B8A, B05)": ["B11", "B8A", "B05"],
}

# SCL classes treated as "not clear" when reporting cloud cover over the AOI.
SCL_CLOUDY = {0, 1, 3, 8, 9, 10}  # no-data, saturated, shadow, cloud med/high, cirrus


def sort_bands(bands) -> list[str]:
    return sorted(bands, key=lambda b: ALL_BANDS.index(b) if b in ALL_BANDS else 99)
