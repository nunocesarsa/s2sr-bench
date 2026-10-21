"""Sentinel-2 Super-Resolution Bench — Streamlit app.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import traceback
from datetime import date, timedelta

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from s2sr import viz
from s2sr.bands import ALL_BANDS, CENTRAL_WAVELENGTH_NM, COMPOSITES
from s2sr.models import PLUGIN_ERRORS, auto_device, registry
from s2sr.models.base import DEVICE_NOTES
from s2sr.runner import evaluate
from s2sr.scene import synthetic_scene

st.set_page_config(page_title="S2 Super-Resolution Bench", page_icon="🛰️", layout="wide")
ss = st.session_state
ss.setdefault("scene", None)
ss.setdefault("results", {})      # {model_key: {"real": SRResult, "wald": SRResult}}
ss.setdefault("errors", {})
ss.setdefault("hits", [])
ss.setdefault("highres", None)     # s2sr.highres.HighRes aligned to the current scene
ss.setdefault("lat_in", 52.3765)
ss.setdefault("lon_in", 4.9000)
ss.setdefault("map_click_seen", None)
# A map click from the previous run: apply it before the lat/lon widgets are created.
if "pending_ll" in ss:
    ss.lat_in, ss.lon_in = ss.pop("pending_ll")


def set_scene(sc, hr=None):
    ss.scene, ss.highres = sc, hr
    ss.results, ss.errors = {}, {}

HIGHER_BETTER = {"PSNR", "SSIM", "consistency PSNR", "corr"}
LOWER_BETTER = {"RMSE", "SAM°", "ERGAS", "|bias|", "SAM° vs bicubic", "runtime s"}


# ============================================================== cached models
@st.cache_resource(show_spinner=False)
def get_model(key: str, device: str, opts: tuple):
    m = registry()[key](device=device, **dict(opts))
    m.ensure_loaded()
    return m


# ===================================================================== sidebar
with st.sidebar:
    st.title("🛰️ S2 SR Bench")

    # ---------------------------------------------------------------- data
    st.header("1 · Data")
    source = st.radio("Source", ["Sample scenes", "STAC search", "Local GeoTIFF", "Local band files / .SAFE",
                                 "Synthetic demo"], label_visibility="collapsed")

    if source == "Sample scenes":
        from s2sr.sample import list_samples, load_sample

        samples = list_samples()
        if not samples:
            st.info("No samples yet. Double-click `download_sample.bat` (or run "
                    "`python -m s2sr.cli sample`) to fetch one into the `samples` folder.")
        else:
            pick_s = st.selectbox("Sample", samples, format_func=lambda p: p.name)
            if st.button("Load sample", type="primary", width="stretch"):
                try:
                    set_scene(*load_sample(pick_s))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Load failed: {e}")

    elif source == "STAC search":
        from s2sr.data import stac

        catalog = st.selectbox("Catalogue", list(stac.CATALOGS))
        lat = st.number_input("Latitude", format="%.5f", key="lat_in")
        lon = st.number_input("Longitude", format="%.5f", key="lon_in")
        st.caption("Or click the map in the main panel.")
        d0, d1 = st.date_input("Date range", (date.today() - timedelta(days=120), date.today()))
        max_cloud = st.slider("Max tile cloud cover %", 0, 100, 10)
        size = st.select_slider("Subset size (10 m px)", [256, 384, 512, 768, 1024, 1536], 512,
                                help="512 px = 5.12 km. Larger subsets are slower for GAN/diffusion models.")
        if st.button("Search", width="stretch"):
            with st.spinner("Searching…"):
                try:
                    ss.hits = stac.search(lat, lon, d0, d1, max_cloud, catalog)
                    if not ss.hits:
                        st.warning("No scenes. Widen the dates or cloud limit.")
                except Exception as e:  # noqa: BLE001
                    st.error(f"Search failed: {e}")
        if ss.hits:
            pick = st.selectbox("Scene", range(len(ss.hits)), format_func=lambda i: ss.hits[i].label)
            if st.button("Load subset", type="primary", width="stretch"):
                bar = st.progress(0.0, "Reading bands…")
                try:
                    set_scene(stac.read_subset(ss.hits[pick], lat, lon, size, catalog,
                                               progress=lambda p, t: bar.progress(p, t)))
                except Exception as e:  # noqa: BLE001
                    st.error(f"Read failed: {e}")
                bar.empty()

    elif source == "Local GeoTIFF":
        from s2sr.data.local import DEFAULT_ORDER, load_geotiff

        path = st.text_input("Path to GeoTIFF", placeholder="/data/s2_subset.tif")
        up = st.file_uploader("…or upload", type=["tif", "tiff"])
        order = st.text_input("Band order in file", DEFAULT_ORDER,
                              help="Comma-separated, one per file band. Use '-' for bands to ignore.")
        c1, c2 = st.columns(2)
        scale = c1.number_input("Scale", value=10000.0, help="reflectance = (DN + offset) / scale")
        offset = c2.number_input("Offset", value=0.0, help="-1000 for raw L2A DN since baseline 04.00")
        if st.button("Load", type="primary", width="stretch"):
            try:
                src = path or up
                if src is None:
                    st.warning("Give a path or upload a file.")
                else:
                    if up is not None and not path:
                        import tempfile
                        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False)
                        tmp.write(up.getbuffer())
                        tmp.close()
                        src = tmp.name
                    set_scene(load_geotiff(src, order, scale, offset))
            except Exception as e:  # noqa: BLE001
                st.error(f"Load failed: {e}")

    elif source == "Local band files / .SAFE":
        from s2sr.data.local import load_band_folder

        folder = st.text_input("Folder", placeholder="/data/S2B_MSIL2A_….SAFE")
        use_c = st.checkbox("Centre on lat/lon (else tile centre)")
        c1, c2 = st.columns(2)
        clat = c1.number_input("Lat", value=0.0, format="%.5f", disabled=not use_c)
        clon = c2.number_input("Lon", value=0.0, format="%.5f", disabled=not use_c)
        size = st.select_slider("Subset size (10 m px)", [256, 384, 512, 768, 1024, 1536], 512)
        if st.button("Load", type="primary", width="stretch"):
            try:
                set_scene(load_band_folder(folder, (clat, clon) if use_c else None, size))
            except Exception as e:  # noqa: BLE001
                st.error(f"Load failed: {e}")
    else:
        size = st.select_slider("Size", [128, 256, 512], 256)
        if st.button("Load synthetic scene", type="primary", width="stretch"):
            set_scene(synthetic_scene(size))

    # -------------------------------------------------------------- models
    st.header("2 · Models")
    reg = registry()
    avail = {k: c.available() for k, c in reg.items()}
    usable = [k for k, (ok, _) in avail.items() if ok]
    default = [k for k in ["bicubic_x4", "hpf_20to10", "sen2sr_lite_rgbn_x4", "satlas_esrgan_1s2"] if k in usable]
    chosen = st.multiselect("Run", usable, default, format_func=lambda k: f"{reg[k].name}  ·  {reg[k].family}")
    blocked = {k: why for k, (ok, why) in avail.items() if not ok}
    if blocked:
        with st.expander(f"{len(blocked)} unavailable"):
            for k, why in blocked.items():
                st.caption(f"**{reg[k].name}** — {why}")
    if PLUGIN_ERRORS:
        with st.expander("⚠️ Plugin errors"):
            for f, e in PLUGIN_ERRORS.items():
                st.code(f"{f}\n{e}")
    device = st.selectbox("Device", ["auto", "cuda", "cpu"], help=f"auto → {auto_device()}")
    st.caption(f"Running on **{auto_device(device)}**")
    for note in dict.fromkeys(DEVICE_NOTES):
        st.warning(note)
    model_opts: dict[str, dict] = {}
    if "satlas_esrgan_1s2" in chosen:
        with st.expander("Satlas options"):
            model_opts["satlas_esrgan_1s2"] = {"tci_max": st.slider(
                "TCI saturation reflectance", 0.1, 0.4, 0.2, 0.01,
                help="Satlas was trained on 8-bit TCI; reflectance is scaled by 255/this value.")}

    # ---------------------------------------------------------- high-res
    st.header("3 · High-res reference")
    from s2sr import highres as hrmod

    hr_src = st.selectbox("Imagery", ["Esri World Imagery", "Custom XYZ tiles", "Local GeoTIFF"],
                          help="Shown next to and under the results (Compare and Swipe tabs). Visual reference "
                               "only: a different date and sensor, so it is not used in the metrics.")
    hr_url = hr_path = None
    if hr_src == "Custom XYZ tiles":
        hr_url = st.text_input("Tile URL", placeholder="https://…/{z}/{x}/{y}.jpg",
                               help="Any XYZ/TMS-style URL with {z}, {x}, {y} in Web Mercator, e.g. a national "
                                    "orthophoto service.")
    elif hr_src == "Local GeoTIFF":
        hr_path = st.text_input("Path", placeholder="/mnt/d/data/orthophoto.tif",
                                help="Any georeferenced RGB raster; it is warped onto the scene grid.")
    can_hr = ss.scene is not None and ss.scene.transform is not None
    if st.button("Load high-res", width="stretch", disabled=not can_hr):
        bar = st.progress(0.0, "Fetching…")
        try:
            if hr_src == "Local GeoTIFF":
                ss.highres = hrmod.from_geotiff(ss.scene, hr_path)
            elif hr_src == "Custom XYZ tiles":
                ss.highres = hrmod.from_xyz(ss.scene, hr_url, "Custom tiles", "", 20,
                                            progress=lambda p, t: bar.progress(p, t))
            else:
                ss.highres = hrmod.from_xyz(ss.scene, progress=lambda p, t: bar.progress(p, t))
        except Exception as e:  # noqa: BLE001
            st.error(f"High-res failed: {e}")
        bar.empty()
    if ss.scene is not None and ss.scene.transform is None:
        st.caption("Needs a georeferenced scene (not the synthetic demo).")
    if ss.highres is not None:
        st.caption(f"✓ {ss.highres.label}, {ss.highres.res:g} m"
                   + (f" · © {ss.highres.attribution}" if ss.highres.attribution else ""))

    # ---------------------------------------------------------- evaluation
    st.header("4 · Evaluation")
    mode_label = st.radio("Mode", ["Both", "Real scene (no reference)", "Wald (degraded reference)"])
    mode = {"Both": "both", "Real scene (no reference)": "real", "Wald (degraded reference)": "wald"}[mode_label]
    degrade_mode = st.radio("Degradation model", ["mtf", "area"], horizontal=True,
                            help="mtf: Gaussian blur matched to the S2 MTF, then decimation. area: block mean.")
    border = st.number_input("Ignore border (px)", 0, 32, 4)
    run = st.button("▶ Run models", type="primary", width="stretch",
                    disabled=ss.scene is None or not chosen)

# ======================================================================= main
scene = ss.scene

if source == "STAC search":
    with st.expander("📍 Location — click the map to set the subset centre", expanded=scene is None):
        try:
            import folium
            from streamlit_folium import st_folium

            la, lo = float(ss.lat_in), float(ss.lon_in)
            fm = folium.Map(location=[la, lo], zoom_start=12, tiles=None, control_scale=True)
            folium.TileLayer(hrmod.PRESETS["Esri World Imagery"]["url"], attr="Esri World Imagery",
                             name="Satellite", max_zoom=19).add_to(fm)
            folium.TileLayer("OpenStreetMap", name="Map").add_to(fm)
            half = size * 10 / 2
            dlat = half / 111_320
            dlon = half / (111_320 * max(np.cos(np.radians(la)), 0.01))
            folium.Rectangle([[la - dlat, lo - dlon], [la + dlat, lo + dlon]], color="#ffcc00", weight=2,
                             fill=False, tooltip=f"Subset {size}×{size} px ({size * 10 / 1000:.1f} km)").add_to(fm)
            folium.Marker([la, lo]).add_to(fm)
            folium.LayerControl(collapsed=True).add_to(fm)
            res_map = st_folium(fm, height=460, use_container_width=True, returned_objects=["last_clicked"],
                                key="picker")
            c = (res_map or {}).get("last_clicked")
            if c and (round(c["lat"], 6), round(c["lng"], 6)) != ss.map_click_seen:
                ss.map_click_seen = (round(c["lat"], 6), round(c["lng"], 6))
                ss.pending_ll = (c["lat"], c["lng"])
                ss.hits = []          # search results belong to the old location
                st.rerun()
            st.caption(f"Centre {la:.5f}, {lo:.5f}. Then press **Search** in the sidebar. "
                       "Layers (top right) switch between satellite and map.")
        except ImportError:
            st.warning("Install `streamlit-folium` and `folium` for the map picker.")
if scene is None:
    st.title("Sentinel-2 Super-Resolution Bench")
    st.markdown(
        "Load a Sentinel-2 subset in the sidebar (a downloaded sample, STAC search, a local GeoTIFF/.SAFE, or the synthetic demo), "
        "pick models, and run. Every model is scored two ways:\n\n"
        "- **Wald (degraded reference)** — the scene is degraded by the model's factor, super-resolved, and compared "
        "with the original. Gives PSNR/SSIM/SAM/ERGAS against real data.\n"
        "- **Real scene (no reference)** — the SR output is degraded back to native resolution and compared with the "
        "input (consistency), plus spectral angle and high-frequency gain relative to bicubic.\n\n"
        "Always keep a bicubic baseline in the run: a model that can't beat it on the Wald test is not recovering detail."
    )
    st.stop()

m = scene.meta
h10, w10 = scene.shape_at(10)
cols = st.columns(5)
cols[0].metric("Scene", (m.get("datetime") or m.get("id", ""))[:10])
cols[1].metric("Size", f"{h10}×{w10} px", f"{h10 * 10 / 1000:.1f} km", delta_color="off")
cols[2].metric("Tile cloud", f"{m['tile_cloud_cover']:.1f}%" if "tile_cloud_cover" in m else "—")
cols[3].metric("AOI clear", f"{100 * m['aoi_clear_fraction']:.0f}%" if "aoi_clear_fraction" in m else "—")
cols[4].metric("Bands", len(scene.bands))
st.caption(f"{m.get('id', '')} · {m.get('source', '')} · CRS {scene.crs} · "
           f"offset {m.get('offset_applied', 0):+g} DN" + (f" · ⚠️ {m['warning']}" if "warning" in m else ""))

# --------------------------------------------------------------------- run
if run:
    ss.errors = {k: v for k, v in ss.errors.items() if k not in chosen}
    prog = st.progress(0.0)
    for i, key in enumerate(chosen):
        cls = reg[key]
        prog.progress(i / len(chosen), f"{cls.name}…")
        if not scene.has(cls.in_bands):
            ss.errors[key] = f"Scene lacks bands {[b for b in cls.in_bands if b not in scene.bands]}"
            continue
        try:
            with st.spinner(f"Loading {cls.name} (downloads weights on first use)…"):
                model = get_model(key, device, tuple(sorted(model_opts.get(key, {}).items())))
            with st.spinner(f"Running {cls.name}…"):
                ss.results[key] = evaluate(model, scene, mode, degrade_mode, int(border))
        except Exception as e:  # noqa: BLE001
            ss.errors[key] = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"
    prog.empty()

for key, err in ss.errors.items():
    with st.expander(f"❌ {reg[key].name if key in reg else key} failed", expanded=False):
        st.code(err)

results = ss.results
if not results:
    st.info("Pick models in the sidebar and press **Run models**.")

tabs = st.tabs(["🖼️ Compare", "↔️ Swipe", "📊 Metrics", "🌈 Spectra", "🔍 Residuals", "💾 Export", "ℹ️ Metrics guide"])


# ------------------------------------------------------------- view state
modes_present = [md for md in ("real", "wald") if any(md in r for r in results.values())] or ["real"]
GH, GW = h10 * 4, w10 * 4   # common display grid: the subset at 2.5 m


def band_triplet(comp: str) -> list[str]:
    names = COMPOSITES.get(comp) or [comp.split()[-1]]
    return names * 3 if len(names) == 1 else names


def view_controls(prefix: str, show_view: bool = True):
    c = st.columns([2.2, 1.3, 1.2, 1, 1])
    comp_names = list(COMPOSITES) + [f"Single band {b}" for b in ALL_BANDS]
    comp = c[0].selectbox("Display", comp_names, key=f"{prefix}_comp")
    view = c[1].radio("View", modes_present, horizontal=True, key=f"{prefix}_view",
                      format_func=lambda v: {"real": "Real", "wald": "Wald"}[v]) if show_view else "real"
    zoom = c[2].select_slider("Zoom", [1, 2, 4, 8, 16, 32], 4, key=f"{prefix}_zoom", help="1 = whole subset")
    cy = c[3].slider("Centre Y", 0.0, 1.0, 0.5, 0.01, key=f"{prefix}_cy")
    cx = c[4].slider("Centre X", 0.0, 1.0, 0.5, 0.01, key=f"{prefix}_cx")
    return comp, view, viz.view_window(GH, GW, cy, cx, 1 / zoom)


HR_LABEL = "High-res reference"


def make_panels(comp: str, view: str) -> tuple[list[tuple[str, np.ndarray | None, str]], list]:
    """[(label, full image or None, note)] for the inputs and every model in `view`."""
    names3 = band_triplet(comp)
    if not all(b in scene.bands for b in names3):
        return [], None
    lims = viz.stretch_limits(scene.bands, names3)   # one stretch for all panels
    out = []
    if view == "real":
        out.append(("Input (native sampling)", viz.panel_image(scene.bands, scene.res, names3, lims), ""))
    else:
        seen = set()
        for r in results.values():
            if "wald" in r and r["wald"].metrics.get("wald_factor") not in seen:
                f = r["wald"].metrics["wald_factor"]
                seen.add(f)
                low = r["wald"].input_scene
                out.append((f"Degraded input ×{f}", viz.panel_image(low.bands, low.res, names3, lims),
                            f"true pixel {10 * f:g} m"))
        out.append(("Reference = original", viz.panel_image(scene.bands, scene.res, names3, lims), ""))
    if ss.highres is not None:
        out.append((HR_LABEL, ss.highres.image, f"{ss.highres.label}, {ss.highres.res:g} m · not used in metrics"))
    for key, r in results.items():
        if view in r:
            rr = r[view]
            # Band-sharpening models (e.g. HPF 20→10 m) only output the 20 m bands. Bands they don't
            # output are taken from their input when that input is already at least as sharp as
            # the output (e.g. native 10 m B04), so their results can still be shown in colour.
            src = rr.input_scene
            filled = [b for b in dict.fromkeys(names3) if b not in rr.bands
                      and b in src.bands and src.res[b] <= rr.out_res + 1e-6]
            bands, res = dict(rr.bands), {b: rr.out_res for b in rr.bands}
            for b in filled:
                bands[b], res[b] = src.bands[b], src.res[b]
            own = [b for b in dict.fromkeys(names3) if b in rr.bands]
            img = viz.panel_image(bands, res, names3, lims) if own else None
            missing = [b for b in dict.fromkeys(names3) if b not in rr.bands]
            if img is None:
                note = f"no {', '.join(missing)} output — pick a display with "  \
                       f"{', '.join(rr.bands)}"
            else:
                note = f"{rr.runtime_s:.2f} s" + (f" · {', '.join(filled)} from input" if filled else "")
            out.append((rr.model_name, img, note))
    return out, lims


# ============================================================== Compare tab
with tabs[0]:
    comp, view, win = view_controls("cmp")
    ncols = st.select_slider("Panels per row", [1, 2, 3, 4], 3)
    pnl, _ = make_panels(comp, view)
    if not pnl:
        st.warning("The scene lacks the bands for this composite.")
    grid = st.columns(ncols)
    for i, (label, img, note) in enumerate(pnl):
        with grid[i % ncols]:
            if img is None:
                st.markdown(f"**{label}**")
                st.info(note)
                continue
            crop = viz.crop_to_window(img, GH, win)
            st.image(viz.resize_nearest(crop, 512), caption=f"{label}" + (f" · {note}" if note else ""),
                     width="stretch")
    if view == "wald":
        st.caption("Wald view: models super-resolved the degraded input; the reference is the real scene at the "
                   "resolution the output claims.")

# ================================================================ Swipe tab
with tabs[1]:
    comp_s, view_s, win_s = view_controls("swp")
    pnl_s, _ = make_panels(comp_s, view_s)
    pnl_s = [p for p in pnl_s if p[1] is not None]
    if len(pnl_s) >= 2:
        labels = [p[0] for p in pnl_s]
        c1, c2 = st.columns(2)
        # Default: newest model result on the left, high-res (or the input) underneath on the right
        right_default = labels.index(HR_LABEL) if HR_LABEL in labels else 0
        left_default = len(labels) - 1 if labels[-1] != HR_LABEL else 0
        if left_default == right_default:
            left_default = (right_default + 1) % len(labels)
        left = c1.selectbox("Left / top", labels, left_default)
        right = c2.selectbox("Right / underneath", labels, right_default)
        st.caption("Drag on the image or use the sliders: **Swipe** moves the divider, **Opacity** fades the "
                   "top image over the one underneath.")
        li = viz.resize_nearest(viz.crop_to_window(dict((p[0], p[1]) for p in pnl_s)[left], GH, win_s), 800)
        ri = viz.resize_nearest(viz.crop_to_window(dict((p[0], p[1]) for p in pnl_s)[right], GH, win_s), 800)
        html = viz.swipe_html(li, ri, left, right, 760)
        if hasattr(st, "iframe"):          # Streamlit >= 1.5x
            st.iframe(html, height=900)
        else:
            components.html(html, height=900)
    else:
        st.info("Run at least one model (or load a high-res reference) to compare.")

# ============================================================== Metrics tab
with tabs[2]:
    if not results:
        st.info("No results yet.")


    def summary_table(md: str) -> pd.DataFrame:
        rows = []
        for key, r in results.items():
            if md in r:
                rr = r[md]
                row = {"model": rr.model_name, "family": reg[key].family if key in reg else "",
                       **rr.metrics.get("summary", {}), "runtime s": rr.runtime_s}
                if md == "wald":
                    row["factor"] = rr.metrics.get("wald_factor")
                    row["ref pixel m"] = rr.metrics.get("reference_res")
                rows.append(row)
        return pd.DataFrame(rows).set_index("model") if rows else pd.DataFrame()


    def styled(df: pd.DataFrame):
        sty = df.style.format(precision=3, na_rep="–")
        hi = [c for c in df.columns if c in HIGHER_BETTER]
        lo = [c for c in df.columns if c in LOWER_BETTER]
        if len(df) > 1:
            sty = sty.highlight_max(subset=hi, color="#2e7d3240").highlight_min(subset=lo, color="#2e7d3240")
        return sty


    for md, title, expl in [
        ("wald", "Wald protocol — against a real reference",
         "Each model gets the scene degraded by its own factor (×4 for 10→2.5 m, ×2 for 20→10 m) and is scored "
         "against the original. Compare models **with the same factor** and always against the bicubic baseline."),
        ("real", "Real scene — no reference",
         "**consistency PSNR**: SR degraded back to native resolution vs. the input (high = radiometrically "
         "faithful). **HF gain**: high-frequency energy relative to bicubic (>1 = sharper; very high values can be "
         "noise or blockiness). **SAM° vs bicubic**: spectral change relative to plain interpolation."),
    ]:
        df = summary_table(md)
        if df.empty:
            continue
        st.subheader(title)
        st.caption(expl)
        st.dataframe(styled(df), width="stretch")
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        mcol = st.selectbox("Chart metric", num_cols, key=f"chart_{md}")
        st.bar_chart(df[mcol], horizontal=True)
        with st.expander("Per-band metrics"):
            rows = []
            for key, r in results.items():
                if md in r:
                    for b, v in r[md].metrics.get("per_band", {}).items():
                        rows.append({"model": r[md].model_name, "band": b, **v})
            if rows:
                st.dataframe(pd.DataFrame(rows).set_index(["model", "band"]).style.format(precision=4, na_rep="–"),
                             width="stretch")
        st.download_button(f"Download {md} metrics (CSV)", df.to_csv().encode(), f"metrics_{md}.csv", "text/csv",
                           key=f"dl_{md}")

# ============================================================== Spectra tab
with tabs[3]:
    if not results:
        st.info("No results yet.")
    else:
        st.caption("Reflectance at one location: the input band values vs. each model's output averaged over the "
                   "same 10 m footprint (spectral preservation) and at a single 2.5 m sub-pixel (local detail).")
        c1, c2, c3 = st.columns(3)
        py = c1.slider("Row (fraction)", 0.0, 1.0, 0.5, 0.005, key="sp_y")
        px = c2.slider("Col (fraction)", 0.0, 1.0, 0.5, 0.005, key="sp_x")
        foot = c3.radio("Footprint", ["10 m pixel", "40 m (4×4 px)"], horizontal=True)
        size_m = 10 if foot.startswith("10") else 40
        yc, xc = py * h10 * 10, px * w10 * 10    # metres from top-left

        def sample(bands, res, b):
            a, r = bands[b], (res[b] if isinstance(res, dict) else res)
            n = max(1, int(round(size_m / r)))
            r0 = int(np.clip(yc / r - n / 2, 0, a.shape[0] - n))
            c0 = int(np.clip(xc / r - n / 2, 0, a.shape[1] - n))
            return float(a[r0:r0 + n, c0:c0 + n].mean())

        df = pd.DataFrame(index=[f"{b} ({CENTRAL_WAVELENGTH_NM[b]} nm)" for b in ALL_BANDS])
        df["Input"] = [sample(scene.bands, scene.res, b) if b in scene.bands else np.nan for b in ALL_BANDS]
        for key, r in results.items():
            if "real" in r:
                rr = r["real"]
                df[rr.model_name] = [sample(rr.bands, rr.out_res, b) if b in rr.bands else np.nan for b in ALL_BANDS]
        st.line_chart(df)
        st.dataframe(df.T.style.format(precision=4, na_rep="–"), width="stretch")

# ============================================================ Residuals tab
with tabs[4]:
    if not results:
        st.info("No results yet.")
    else:
        from s2sr.bands import MTF_NYQUIST
        from s2sr.metrics import reference_bands
        from s2sr.resample import degrade

        c1, c2, c3 = st.columns([2, 1, 1])
        key = c1.selectbox("Model", list(results), format_func=lambda k: results[k][next(iter(results[k]))].model_name)
        md = c2.radio("View", list(results[key]), horizontal=True, key="res_view",
                      format_func=lambda v: {"real": "Consistency", "wald": "Error vs ref"}[v])
        rr = results[key][md]
        band = c3.selectbox("Band", list(rr.bands))
        if md == "real":
            f = int(round(scene.res[band] / rr.out_res))
            back = degrade(rr.bands[band], f, degrade_mode, MTF_NYQUIST.get(band, 0.3)) if f > 1 else rr.bands[band]
            ref = scene.bands[band]
            title = f"SR degraded to {scene.res[band]:g} m − input"
        else:
            refs = reference_bands(scene.bands, scene.res, [band], rr.true_res)
            back = rr.bands[band]
            ref = refs.get(band)
            title = f"SR − reference ({rr.true_res:g} m)"
        if ref is None:
            st.info("No reference at this resolution for this band (it would need upsampling).")
        else:
            h = min(back.shape[0], ref.shape[0])
            w = min(back.shape[1], ref.shape[1])
            diff = back[:h, :w] - ref[:h, :w]
            vmax = float(np.percentile(np.abs(diff), 99)) or 1e-3
            zoom_r = st.select_slider("Zoom", [1, 2, 4, 8], 1, key="res_zoom")
            win_r = viz.view_window(h, w, 0.5, 0.5, 1 / zoom_r)
            cc1, cc2 = st.columns(2)
            lo, hi = np.percentile(ref, [2, 98])
            cc1.image(viz.resize_nearest(viz.crop_to_window(viz.grey(ref[:h, :w], lo, hi), h, win_r), 512),
                      caption=f"{'Input' if md == 'real' else 'Reference'} {band}", width="stretch")
            cc2.image(viz.resize_nearest(viz.crop_to_window(viz.diverging(diff, vmax), h, win_r), 512),
                      caption=f"{title} · ±{vmax:.4f} reflectance (blue = too dark, red = too bright)",
                      width="stretch")
            st.caption(f"mean {diff.mean():+.5f} · RMSE {np.sqrt((diff ** 2).mean()):.5f}")

# =============================================================== Export tab
with tabs[5]:
    if not results:
        st.info("No results yet.")
    else:
        from s2sr.export import geotiff_bytes, png_bytes

        st.caption("Super-resolved outputs as float32 GeoTIFF (surface reflectance), georeferenced at the output "
                   "pixel size. Real-scene outputs only.")
        for key, r in results.items():
            if "real" not in r:
                continue
            rr = r["real"]
            c1, c2 = st.columns([3, 1])
            c1.markdown(f"**{rr.model_name}** — {', '.join(rr.bands)} @ {rr.out_res:g} m")
            if c2.button("Prepare GeoTIFF", key=f"prep_{key}"):
                ss[f"tif_{key}"] = geotiff_bytes(rr)
            if f"tif_{key}" in ss:
                c2.download_button("Download", ss[f"tif_{key}"], f"{scene.meta.get('id', 'scene')}_{key}.tif",
                                   "image/tiff", key=f"dl_tif_{key}")

# ======================================================== Metrics guide tab
with tabs[6]:
    st.markdown("""
### How to read the scores

**There is no 2.5 m ground truth for Sentinel-2**, so this app combines two imperfect views:

| Test | What it measures | Blind spot |
|---|---|---|
| **Wald protocol** (degrade ×f → super-resolve → compare with original) | Real detail recovery, with a true reference: PSNR, SSIM, SAM, ERGAS | Assumes a model behaves the same at 40→10 m as at 10→2.5 m. Models trained on real HR pairs (e.g. NAIP/SPOT) may score lower here than they deserve. |
| **Consistency** (SR degraded back to native vs. input) | Radiometric faithfulness: the SR image must still *be* the observed image at 10/20 m | A blurry output passes easily. Read together with HF gain. |
| **HF gain** (Laplacian energy vs. bicubic) | How much sharper than interpolation | Noise, ringing and hallucinated texture also increase it. |
| **SAM° vs bicubic** | Spectral change introduced by the model | Some change is expected where real detail is added. |

**Practical reading:** a good model beats bicubic on Wald PSNR/SSIM, keeps consistency PSNR high (≥ bicubic),
has HF gain clearly above 1, and a small spectral angle. GAN models often look sharpest but trade
consistency and Wald PSNR for perceptual detail — inspect the residuals tab for invented structures.

**Degradation model:** `mtf` blurs with a Gaussian matched to the Sentinel-2 MTF at Nyquist before
decimation, which is what the sensor does; `area` is a simple block mean.

**Reflectance:** L2A DN are converted with (DN + offset)/10000, where the offset is −1000 for
processing baseline ≥ 04.00 unless the catalogue already applied it.
""")
