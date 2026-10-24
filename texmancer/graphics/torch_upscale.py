"""CUDA Real-ESRGAN via PyTorch + spandrel (more reliable than the ncnn/Vulkan
build on large inputs). Handles alpha by running it through the model as a gray
image, the same way Real-ESRGAN's own inference script does, and tiles large
inputs so VRAM stays bounded.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import torch
from PIL import Image

DEFAULT_MODEL = Path.home() / "models" / "RealESRGAN_x4plus_anime_6B.pth"


def _keep_binary_alpha(src: np.ndarray, out: Image.Image) -> Image.Image:
    """1-bit alpha (DXT1 cut-outs) stays 1-bit: soft upscaled edges would be
    thresholded by the encoder anyway, and on small icons they shift the shape."""
    a = src[..., 3]
    if a.min() == 255 or not np.isin(a, (0, 255)).all():
        return out
    arr = np.array(out)
    arr[..., 3] = np.where(arr[..., 3] >= 128, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, "RGBA")


class TorchUpscaler:
    def __init__(self, model_path: Path = DEFAULT_MODEL, tile: int = 384, pad: int = 16):
        import spandrel

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available; install torch from the cu130 index")
        desc = spandrel.ModelLoader().load_from_file(str(model_path))
        self.scale = desc.scale
        self.half = desc.supports_half
        self.model = desc.model.eval().cuda()
        if self.half:
            self.model = self.model.half()
        self.tile, self.pad = tile, pad

    @torch.inference_mode()
    def _run(self, rgb: np.ndarray) -> np.ndarray:
        """HxWx3 uint8 -> (H*s)x(W*s)x3 uint8."""
        x = torch.from_numpy(rgb).cuda().permute(2, 0, 1).unsqueeze(0)
        x = x.half() if self.half else x.float()
        x = x / 255.0
        _, _, h, w = x.shape
        s, t, p = self.scale, self.tile, self.pad
        if h * w <= t * t:
            y = self.model(x)
        else:
            y = torch.empty((1, 3, h * s, w * s), dtype=x.dtype, device=x.device)
            for y0 in range(0, h, t):
                for x0 in range(0, w, t):
                    y1, x1 = min(y0 + t, h), min(x0 + t, w)
                    # Overlap each tile by `pad` pixels so seams don't show.
                    py0, px0 = max(y0 - p, 0), max(x0 - p, 0)
                    py1, px1 = min(y1 + p, h), min(x1 + p, w)
                    out = self.model(x[:, :, py0:py1, px0:px1])
                    oy, ox = (y0 - py0) * s, (x0 - px0) * s
                    y[:, :, y0 * s:y1 * s, x0 * s:x1 * s] = out[:, :, oy:oy + (y1 - y0) * s, ox:ox + (x1 - x0) * s]
        y = y.clamp_(0, 1).mul_(255).round_().byte()
        return y.squeeze(0).permute(1, 2, 0).cpu().numpy()

    @torch.inference_mode()
    def _run_batch(self, arrs: list[np.ndarray]) -> list[np.ndarray]:
        """Same-shape HxWx3 inputs, each small enough to skip tiling, in one pass."""
        x = torch.from_numpy(np.stack(arrs)).cuda().permute(0, 3, 1, 2)
        x = (x.half() if self.half else x.float()) / 255.0
        y = self.model(x).clamp_(0, 1).mul_(255).round_().byte()
        return list(y.permute(0, 2, 3, 1).cpu().numpy())

    def _many(self, arrs: list[np.ndarray]) -> list[np.ndarray]:
        """Upscale a list of HxWx3 arrays, batching same-shape small ones."""
        out: list[np.ndarray | None] = [None] * len(arrs)
        groups: dict[tuple[int, int], list[int]] = {}
        for i, a in enumerate(arrs):
            h, w = a.shape[:2]
            if h * w <= self.tile * self.tile:
                groups.setdefault((h, w), []).append(i)
            else:
                out[i] = self._run(a)
        # Pixel budget per batch: two full tiles' worth keeps VRAM where single tiles were.
        budget = 2 * self.tile * self.tile
        for (h, w), idx in groups.items():
            n = max(1, budget // (h * w))
            for k in range(0, len(idx), n):
                part = idx[k:k + n]
                for i, y in zip(part, self._run_batch([arrs[i] for i in part])):
                    out[i] = y
        return out  # type: ignore[return-value]

    def _upscale_arrays(self, rgbas: list[np.ndarray], alpha_mode: str) -> list[Image.Image]:
        return [_keep_binary_alpha(src, out) for src, out in zip(rgbas, self._upscale_raw(rgbas, alpha_mode))]

    def _upscale_raw(self, rgbas: list[np.ndarray], alpha_mode: str) -> list[Image.Image]:
        if alpha_mode in ("resample", "bilinear", "nearest"):  # last resorts: conventional resizes, no model
            filt = {"resample": Image.LANCZOS, "bilinear": Image.BILINEAR, "nearest": Image.NEAREST}[alpha_mode]
            return [Image.fromarray(a, "RGBA").resize((a.shape[1] * self.scale, a.shape[0] * self.scale), filt)
                    for a in rgbas]
        jobs: list[np.ndarray] = []
        for rgba in rgbas:
            jobs.append(np.ascontiguousarray(rgba[..., :3]))
            if alpha_mode == "model" and rgba[..., 3].min() != 255:
                jobs.append(np.ascontiguousarray(np.repeat(rgba[..., 3:4], 3, axis=2)))
        res = iter(self._many(jobs))
        imgs = []
        for rgba in rgbas:
            rgb = next(res)
            alpha = rgba[..., 3]
            h, w = rgb.shape[:2]
            if alpha.min() == 255:
                a = np.full((h, w), 255, np.uint8)
            elif alpha_mode == "lanczos":
                a = np.asarray(Image.fromarray(alpha, "L").resize((w, h), Image.LANCZOS))
            else:
                a = next(res).mean(axis=2).round().astype(np.uint8)
            imgs.append(Image.fromarray(np.dstack([rgb, a]), "RGBA"))
        return imgs

    def upscale(self, img: Image.Image, alpha_mode: str = "model") -> Image.Image:
        """``alpha_mode="model"`` sharpens alpha with the network; ``"lanczos"`` resizes
        the source alpha conventionally. The model turns faint, semi-transparent
        strokes (spell-list doodles) into solid ink, which the validator rejects."""
        return self._upscale_arrays([np.asarray(img.convert("RGBA"))], alpha_mode)[0]

    def upscale_files(self, pairs: Iterable[tuple[Path, Path]], alpha_mode: str = "model",
                      progress: Callable[[int, Path], None] | None = None,
                      chunk: int = 64, workers: int = 8) -> None:
        """Decode the next chunk and encode finished PNGs on threads while the GPU
        works, so it never waits on file I/O. Pass pairs sorted by size so
        same-shape images land in one chunk and batch together."""
        from concurrent.futures import ThreadPoolExecutor

        pairs = list(pairs)

        def load(src: Path) -> np.ndarray:
            with Image.open(src) as im:
                return np.asarray(im.convert("RGBA"))

        def save(img: Image.Image, dst: Path) -> None:
            img.save(dst, compress_level=1)

        with ThreadPoolExecutor(workers) as io:
            chunks = [pairs[i:i + chunk] for i in range(0, len(pairs), chunk)]
            pending = [io.submit(load, s) for s, _ in chunks[0]] if chunks else []
            saves: list = []
            done = 0
            for ci, part in enumerate(chunks):
                rgbas = [f.result() for f in pending]
                pending = [io.submit(load, s) for s, _ in chunks[ci + 1]] if ci + 1 < len(chunks) else []
                for f in saves:  # surface write errors and keep at most one chunk queued
                    f.result()
                saves = [io.submit(save, img, dst) for img, (_, dst) in zip(self._upscale_arrays(rgbas, alpha_mode), part)]
                if progress:
                    for k, (src, _) in enumerate(part):
                        progress(done + k, src)
                done += len(part)
            for f in saves:
                f.result()
