"""Per-image check that an upscale actually contains the source picture.

An upscaler can fail silently (a lost GPU device writes blank or garbled images
with a success exit code), so every output is shrunk back to the source size
and compared. Thresholds were calibrated on known-good 4x outputs: alpha
differs by <= 4/255 on average, colour by <= 21/255; blank or garbled output
differs by ~100+.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
from PIL import Image

MAX_ALPHA_DIFF = 12.0
MAX_COLOR_DIFF = 40.0


def check_upscale(src: Path | Image.Image, out: Path | Image.Image, scale: int) -> str | None:
    """None if ``out`` is a faithful ``scale``x version of ``src``, else the reason it isn't."""
    s_img = (Image.open(src) if isinstance(src, Path) else src).convert("RGBA")
    try:
        o_img = (Image.open(out) if isinstance(out, Path) else out).convert("RGBA")
    except (OSError, SyntaxError, ValueError, struct.error) as exc:  # e.g. half-written when a run was stopped
        return f"unreadable output: {exc}"
    if o_img.size != (s_img.width * scale, s_img.height * scale):
        return f"size {o_img.size}, expected {(s_img.width * scale, s_img.height * scale)}"
    s = np.asarray(s_img, dtype=np.float32)
    o_full = np.asarray(o_img)
    if o_full.max() == 0 and s.max() > 0:
        return "blank output"
    o = np.asarray(o_img.resize(s_img.size, Image.BOX), dtype=np.float32)
    alpha_diff = float(np.abs(o[..., 3] - s[..., 3]).mean())
    if alpha_diff > MAX_ALPHA_DIFF:
        return f"alpha differs by {alpha_diff:.1f}/255"
    mask = s[..., 3] > 128
    if mask.any():
        color_diff = float(np.abs(o[..., :3] - s[..., :3])[mask].mean())
        if color_diff > MAX_COLOR_DIFF:
            return f"color differs by {color_diff:.1f}/255"
    return None
