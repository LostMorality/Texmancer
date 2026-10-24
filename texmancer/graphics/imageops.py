"""Image helpers shared by the CUDA and Vulkan upscalers (no torch needed)."""

from __future__ import annotations

import numpy as np
from PIL import Image

# Conventional resizes used as last resorts, when the model's output isn't faithful.
RESIZE_FILTERS = {"resample": Image.LANCZOS, "bilinear": Image.BILINEAR, "nearest": Image.NEAREST}


def keep_binary_alpha(src: np.ndarray, out: Image.Image) -> Image.Image:
    """1-bit alpha (DXT1 cut-outs) stays 1-bit: soft upscaled edges would be
    thresholded by the encoder anyway, and on small icons they shift the shape."""
    a = src[..., 3]
    if a.min() == 255 or not np.isin(a, (0, 255)).all():
        return out
    arr = np.array(out)
    arr[..., 3] = np.where(arr[..., 3] >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


def resize(rgba: np.ndarray, scale: int, mode: str) -> Image.Image:
    return Image.fromarray(rgba, "RGBA").resize((rgba.shape[1] * scale, rgba.shape[0] * scale), RESIZE_FILTERS[mode])


def with_lanczos_alpha(src: np.ndarray, out: Image.Image) -> Image.Image:
    """``out``'s colour with the source alpha resized conventionally."""
    rgb = np.asarray(out.convert("RGB"))
    h, w = rgb.shape[:2]
    alpha = src[..., 3]
    a = np.full((h, w), 255, np.uint8) if alpha.min() == 255 else \
        np.asarray(Image.fromarray(alpha, "L").resize((w, h), Image.LANCZOS))
    return Image.fromarray(np.dstack([rgb, a]), "RGBA")
