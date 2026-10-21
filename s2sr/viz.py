"""Rendering helpers: composites with a shared stretch, crops, diff maps, swipe HTML."""
from __future__ import annotations

import base64
import io

import numpy as np

from .resample import upsample


def stretch_limits(bands: dict[str, np.ndarray], names: list[str], p=(2, 98)) -> list[tuple[float, float]]:
    lims = []
    for b in names:
        a = bands[b]
        lo, hi = np.percentile(a, p)
        lims.append((float(lo), float(hi if hi > lo else lo + 1e-3)))
    return lims


def composite(bands: dict[str, np.ndarray], names: list[str], limits, gamma: float = 1.0) -> np.ndarray | None:
    """(H, W, 3) uint8 from three bands on the same grid; None if a band is missing."""
    if not all(b in bands for b in names):
        return None
    chans = []
    for b, (lo, hi) in zip(names, limits):
        c = np.clip((bands[b] - lo) / (hi - lo), 0, 1) ** (1 / gamma)
        chans.append(c)
    h = min(c.shape[0] for c in chans)
    w = min(c.shape[1] for c in chans)
    return (np.dstack([c[:h, :w] for c in chans]) * 255).astype(np.uint8)


def grey(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    return np.dstack([g, g, g])


def diverging(a: np.ndarray, vmax: float) -> np.ndarray:
    """Blue-white-red map for signed differences."""
    t = np.clip(a / (vmax or 1e-6), -1, 1)
    r = np.where(t > 0, 1.0, 1 + t)
    g = 1 - np.abs(t)
    b = np.where(t < 0, 1.0, 1 - t)
    return (np.dstack([r, g, b]) * 255).astype(np.uint8)


def bands_on_grid(bands: dict[str, np.ndarray], res: dict[str, float] | float, target_res: float,
                  names: list[str]) -> dict[str, np.ndarray]:
    """Nearest-neighbour bands onto a finer display grid (shows true sampling)."""
    out = {}
    for b in names:
        if b not in bands:
            continue
        r = res[b] if isinstance(res, dict) else res
        f = int(round(r / target_res))
        out[b] = upsample(bands[b], f, "nearest") if f > 1 else bands[b]
    return out


def panel_image(bands: dict[str, np.ndarray], res: dict[str, float] | float, names3: list[str],
                limits) -> np.ndarray | None:
    """Composite of three bands at the finest pixel size among them (nearest for coarser ones)."""
    if not all(b in bands for b in names3):
        return None
    rr = [res[b] if isinstance(res, dict) else res for b in names3]
    return composite(bands_on_grid(bands, res, min(rr), list(dict.fromkeys(names3))), names3, limits)


def view_window(gh: int, gw: int, cy: float, cx: float, frac: float) -> tuple[int, int, int, int]:
    """Integer window (r0, r1, c0, c1) on a common display grid of gh x gw."""
    ch, cw = max(4, int(round(gh * frac))), max(4, int(round(gw * frac)))
    r0 = int(np.clip(round(cy * gh - ch / 2), 0, gh - ch))
    c0 = int(np.clip(round(cx * gw - cw / 2), 0, gw - cw))
    return r0, r0 + ch, c0, c0 + cw


def crop_to_window(img: np.ndarray, gh: int, win) -> np.ndarray:
    """Cut the same geographic window out of an image of any resolution.

    Coarser images are pixel-replicated so their true sampling shows; finer
    images (e.g. aerial photos) are cut at their own resolution."""
    r0, r1, c0, c1 = win
    ratio = img.shape[0] / gh
    if ratio > 1.0001:
        m = int(round(ratio))
        return img[r0 * m:r1 * m, c0 * m:c1 * m]
    k = max(1, int(round(gh / img.shape[0])))
    a = img[r0 // k: -(-r1 // k), c0 // k: -(-c1 // k)]
    if k > 1:
        a = np.repeat(np.repeat(a, k, 0), k, 1)
    dr, dc = r0 - (r0 // k) * k, c0 - (c0 // k) * k
    return a[dr: dr + (r1 - r0), dc: dc + (c1 - c0)]


def crop_frac(img: np.ndarray, cy: float, cx: float, frac: float) -> np.ndarray:
    """Crop a window covering `frac` of the image, centred at fractional (cy, cx)."""
    h, w = img.shape[:2]
    ch, cw = max(1, int(h * frac)), max(1, int(w * frac))
    r0 = int(np.clip(cy * h - ch / 2, 0, h - ch))
    c0 = int(np.clip(cx * w - cw / 2, 0, w - cw))
    return img[r0:r0 + ch, c0:c0 + cw]


def resize_nearest(img: np.ndarray, size: int) -> np.ndarray:
    """Resize (H, W, 3) to `size` px on the long side for display: nearest when
    enlarging (keeps pixels visible), area-averaging when shrinking (no aliasing)."""
    h, w = img.shape[:2]
    s = size / max(h, w)
    if s < 1:
        from PIL import Image
        return np.asarray(Image.fromarray(img).resize((max(1, int(w * s)), max(1, int(h * s))), Image.BOX))
    ys = np.minimum((np.arange(int(h * s)) / s).astype(int), h - 1)
    xs = np.minimum((np.arange(int(w * s)) / s).astype(int), w - 1)
    return img[ys][:, xs]


def to_png_b64(img: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def swipe_html(left: np.ndarray, right: np.ndarray, left_label: str, right_label: str, height: int = 600) -> str:
    """Two images stacked: drag the divider to swipe, and fade the top one."""
    a, b = to_png_b64(left), to_png_b64(right)
    return f"""
<div style="max-width:{height}px;margin:auto;font-family:sans-serif;font-size:13px;color:#888">
  <div id="wrap" style="position:relative;width:100%;aspect-ratio:{right.shape[1]}/{right.shape[0]};overflow:hidden;cursor:ew-resize">
    <img src="data:image/png;base64,{b}" style="position:absolute;inset:0;width:100%;height:100%;image-rendering:pixelated">
    <div id="clip" style="position:absolute;inset:0;width:50%;overflow:hidden;border-right:2px solid #fff">
      <img id="limg" src="data:image/png;base64,{a}" style="position:absolute;top:0;left:0;height:100%;image-rendering:pixelated">
    </div>
    <span style="position:absolute;left:8px;top:8px;background:#0009;color:#fff;padding:2px 6px">{left_label}</span>
    <span style="position:absolute;right:8px;top:8px;background:#0009;color:#fff;padding:2px 6px">{right_label}</span>
  </div>
  <label>Swipe <input type="range" min="0" max="100" value="50" style="width:100%" id="sl"></label>
  <label>Opacity of left image <input type="range" min="0" max="100" value="100" style="width:100%" id="op"></label>
</div>
<script>
const wrap=document.getElementById('wrap'),clip=document.getElementById('clip'),
      limg=document.getElementById('limg'),sl=document.getElementById('sl'),op=document.getElementById('op');
function fit(){{limg.style.width=wrap.clientWidth+'px';}}
sl.oninput=()=>clip.style.width=sl.value+'%';
op.oninput=()=>clip.style.opacity=op.value/100;
function drag(e){{const r=wrap.getBoundingClientRect();sl.value=100*(e.clientX-r.left)/r.width;sl.oninput();}}
wrap.onmousemove=e=>{{if(e.buttons)drag(e);}};wrap.onmousedown=drag;
window.onresize=fit;new ResizeObserver(fit).observe(wrap);fit();
</script>"""
