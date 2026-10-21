"""Overlapping-tile inference with feathered blending, for models that
cannot take a whole scene at once."""
from __future__ import annotations

from typing import Callable

import numpy as np


def _weight(n: int, overlap: int) -> np.ndarray:
    w = np.ones(n, np.float32)
    if overlap > 0:
        ramp = (np.arange(overlap, dtype=np.float32) + 0.5) / overlap
        w[:overlap] = ramp
        w[-overlap:] = ramp[::-1]
    return w


def tiled_predict(fn: Callable[[np.ndarray], np.ndarray], x: np.ndarray, scale: int,
                  tile: int = 128, overlap: int = 16, out_channels: int | None = None) -> np.ndarray:
    """Run `fn` on (C, tile, tile) patches of `x` (C, H, W) and stitch the results.

    `fn` must return (C_out, tile*scale, tile*scale). Inputs smaller than a tile
    are reflect-padded.
    """
    c, h, w = x.shape
    ph, pw = max(0, tile - h), max(0, tile - w)
    if ph or pw:
        x = np.pad(x, ((0, 0), (0, ph), (0, pw)), mode="reflect")
    _, H, W = x.shape
    step = tile - 2 * overlap
    if step <= 0:
        raise ValueError("tile must be larger than 2*overlap")

    def starts(n):
        s = list(range(0, max(n - tile, 0) + 1, step))
        if s[-1] + tile < n:
            s.append(n - tile)
        return s

    out = acc = None
    for r in starts(H):
        for q in starts(W):
            y = fn(x[:, r:r + tile, q:q + tile])
            if out is None:
                co = out_channels or y.shape[0]
                out = np.zeros((co, H * scale, W * scale), np.float32)
                acc = np.zeros((H * scale, W * scale), np.float32)
            ov = overlap * scale
            wy = _weight(tile * scale, ov)
            wx = _weight(tile * scale, ov)
            if ov:  # do not feather the outer image border
                if r == 0:
                    wy[:ov] = 1
                if r + tile >= H:
                    wy[-ov:] = 1
                if q == 0:
                    wx[:ov] = 1
                if q + tile >= W:
                    wx[-ov:] = 1
            wgt = np.outer(wy, wx)
            rs, qs = r * scale, q * scale
            out[:, rs:rs + tile * scale, qs:qs + tile * scale] += y * wgt
            acc[rs:rs + tile * scale, qs:qs + tile * scale] += wgt
    out /= np.maximum(acc, 1e-6)
    return out[:, : h * scale, : w * scale]
