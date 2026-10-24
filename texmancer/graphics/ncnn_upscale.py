"""Real-ESRGAN on any Vulkan GPU (AMD, Intel, older NVIDIA) through the official
realesrgan-ncnn-vulkan binary. Same interface as TorchUpscaler.upscale_files.

The binary loads its model once per run, so files go through in chunks: each
chunk is hard-linked into a scratch folder under neutral names and run as one
folder. A chunk that crashes (a lost Vulkan device) just leaves its outputs
missing; the caller's validation sends those to the next fallback.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from PIL import Image

from .imageops import keep_binary_alpha, resize, with_lanczos_alpha

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class NcnnUpscaler:
    def __init__(self, exe: str, model: str = "realesrgan-x4plus-anime", scale: int = 4, gpu: str | None = None,
                 ncnn_tile: int = 0, ncnn_jobs: str = "1:2:2", rest: float = 0.0, workers: int = 4,
                 chunk: int | None = None, log: Callable[[str], None] = print, **_ignored):
        self.exe, self.model, self.scale, self.gpu = exe, model, scale, gpu
        self.tile, self.jobs, self.rest, self.workers = ncnn_tile, ncnn_jobs, rest, workers
        # Small chunks when resting, so the pauses come often enough to matter.
        self.chunk = chunk or (32 if rest else 256)
        self.log = log

    def _run_dir(self, src: Path, dst: Path, largest: int) -> None:
        """Run one folder. The binary can stall after its last image (seen once in
        a 116-image run); with no new output for a while it's killed, and its
        missing outputs go to the caller's next fallback."""
        cmd = [self.exe, "-i", str(src), "-o", str(dst), "-n", self.model, "-s", str(self.scale), "-f", "png",
               "-t", str(self.tile), "-j", self.jobs] + (["-g", self.gpu] if self.gpu is not None else [])
        # Generous for a slow integrated GPU: 20k source pixels a second.
        stall = 60 + largest / 20_000
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, creationflags=_NO_WINDOW,
                                cwd=str(Path(self.exe).parent))
        err: list[bytes] = []
        reader = threading.Thread(target=lambda: err.append(proc.stderr.read()), daemon=True)
        reader.start()
        seen, last = 0, time.monotonic()
        while True:
            try:
                proc.wait(timeout=2)
                break
            except subprocess.TimeoutExpired:
                pass
            n = sum(1 for _ in dst.iterdir())
            if n != seen:
                seen, last = n, time.monotonic()
            elif time.monotonic() - last > stall:
                proc.kill()
                proc.wait()
                self.log(f"  realesrgan-ncnn-vulkan stalled for {stall:.0f} s; stopped it ({n} outputs kept)")
                break
        reader.join(5)
        if proc.returncode not in (0, None) and seen == 0:
            msg = b"".join(err).decode(errors="replace").strip().splitlines()
            self.log(f"  realesrgan-ncnn-vulkan exited {proc.returncode}: {msg[-1] if msg else ''}")

    def upscale_files(self, pairs: Iterable[tuple[Path, Path]], alpha_mode: str = "model",
                      progress: Callable[[int, Path], None] | None = None, **_ignored) -> None:
        pairs = list(pairs)
        if not pairs:
            return

        def load(src: Path) -> np.ndarray:
            with Image.open(src) as im:
                return np.asarray(im.convert("RGBA"))

        if alpha_mode in ("resample", "bilinear", "nearest"):  # no model involved
            def plain(pair):
                resize(load(pair[0]), self.scale, alpha_mode).save(pair[1], compress_level=1)
            with ThreadPoolExecutor(self.workers) as pool:
                for k, _ in enumerate(pool.map(plain, pairs)):
                    if progress:
                        progress(k, pairs[k][0])
            return

        scratch = pairs[0][1].parent / ".ncnn"
        done = 0
        for c in range(0, len(pairs), self.chunk):
            part = pairs[c:c + self.chunk]
            inp, out = scratch / "in", scratch / "out"
            shutil.rmtree(scratch, ignore_errors=True)
            inp.mkdir(parents=True)
            out.mkdir()
            largest = 0
            for k, (src, _) in enumerate(part):
                try:
                    os.link(src, inp / f"{k}.png")
                except OSError:
                    shutil.copyfile(src, inp / f"{k}.png")
                with Image.open(src) as im:
                    largest = max(largest, im.width * im.height)
            t = time.perf_counter()
            self._run_dir(inp, out, largest)
            if self.rest:
                time.sleep((time.perf_counter() - t) * self.rest)

            def finish(k: int) -> None:
                src, dst = part[k]
                res = out / f"{k}.png"
                if not res.exists():
                    dst.unlink(missing_ok=True)
                    return
                try:
                    with Image.open(res) as im:
                        img = im.convert("RGBA")
                except (OSError, SyntaxError, ValueError):  # half-written when a stalled run was stopped
                    dst.unlink(missing_ok=True)
                    return
                s = load(src)
                img = with_lanczos_alpha(s, img) if alpha_mode == "lanczos" else keep_binary_alpha(s, img)
                img.save(dst, compress_level=1)

            with ThreadPoolExecutor(self.workers) as pool:
                list(pool.map(finish, range(len(part))))
            if progress:
                for k, (src, _) in enumerate(part):
                    progress(done + k, src)
            done += len(part)
        shutil.rmtree(scratch, ignore_errors=True)
