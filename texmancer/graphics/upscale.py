"""Offline texture remaster pipeline: WAD -> DDS -> PNG -> AI upscale -> DDS -> WAD.

External tools (both free, portable, no install):
  texconv.exe                   https://github.com/microsoft/DirectXTex/releases
  realesrgan-ncnn-vulkan.exe    https://github.com/xinntao/Real-ESRGAN/releases  (Vulkan, any GPU)

Located via --tools DIR, the W101_TOOLS env var, or PATH.
Nothing here touches the install: output WADs go to a separate folder.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import shutil
import struct
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from ..nif import FMT_DXT1, FMT_DXT3, FMT_DXT5, NifError, NifFile, TextureNif
from ..swap import record_source
from .validate import check_upscale
from .. import GAME_DATA
from ..wad import MAX_WAD_BYTES, Wad, WadError, repack, write_wad

# NiPixelData format -> texconv target. Uncompressed sources go to DXT: at 4x a
# DXT5 texture is the same size as the uncompressed original was.
_NIF_DXGI = {FMT_DXT1: "BC1_UNORM", FMT_DXT3: "BC2_UNORM", FMT_DXT5: "BC3_UNORM"}

# DDS FourCC / pixel format -> texconv -f value. Keep source compression so VRAM
# use only grows by the scale factor squared, not by a format change too.
_FOURCC_TO_DXGI = {b"DXT1": "BC1_UNORM", b"DXT2": "BC2_UNORM", b"DXT3": "BC2_UNORM",
                   b"DXT4": "BC3_UNORM", b"DXT5": "BC3_UNORM", b"ATI2": "BC5_UNORM", b"BC5U": "BC5_UNORM"}


# Loose image files the client decodes with D3DX, re-saved in their own format.
_IMAGE_EXT = {".png": "PNG", ".tga": "TGA", ".jpg": "JPEG", ".jpeg": "JPEG", ".bmp": "BMP", ".gif": "GIF"}


@dataclass(slots=True)
class TexInfo:
    name: str          # path inside the WAD
    wad: str           # WAD file name
    width: int
    height: int
    mips: int
    fourcc: str
    dxgi: str          # format to re-encode with (PIL format name for "img")
    kind: str = "dds"  # "dds" file; "nif" = standalone Textures/*.nif; "nifblock" = texture embedded
                       # in a NIF, name "<entry>#<block index>" (cube faces "<entry>#<block>@<face>");
                       # "img" = loose PNG/TGA/JPG/BMP/GIF; "nifraw" = bump/luma block, resampled not AI'd


def _out_size(f: Path) -> tuple[int, int]:
    """Width/height of an encoded output: DDS header, raw block, or any image."""
    with open(f, "rb") as fh:
        head = fh.read(128)
    if head[:4] == b"DDS ":
        return parse_dds_header(head)[:2]
    if f.suffix == ".blk":
        from ..nif import PixelData
        version, persistent = struct.unpack_from("<I?", head, 0)
        px = PixelData.parse(f.read_bytes()[5:], version, persistent)
        return px.width, px.height
    from PIL import Image
    with Image.open(f) as im:
        return im.size


def parse_dds_header(data: bytes) -> tuple[int, int, int, str, str]:
    if len(data) < 128 or data[:4] != b"DDS ":
        raise ValueError("not a DDS file")
    height, width = struct.unpack_from("<II", data, 12)
    mips = struct.unpack_from("<I", data, 28)[0] or 1
    pf_flags = struct.unpack_from("<I", data, 80)[0]
    fourcc = data[84:88]
    if pf_flags & 0x4:  # DDPF_FOURCC
        dxgi = _FOURCC_TO_DXGI.get(fourcc)
        if dxgi is None:
            raise ValueError(f"unsupported FourCC {fourcc!r}")
        return width, height, mips, fourcc.decode("ascii", "replace"), dxgi
    has_alpha = bool(pf_flags & 0x1)
    # D3D9 uncompressed textures are BGRA in memory.
    return width, height, mips, "RGB" + ("A" if has_alpha else ""), "B8G8R8A8_UNORM" if has_alpha else "B8G8R8X8_UNORM"


def find_tool(exe: str, tools_dir: str | None) -> str:
    for d in filter(None, (tools_dir, os.environ.get("W101_TOOLS"))):
        for cand in Path(d).rglob(exe):
            return str(cand)
    found = shutil.which(exe)
    if not found:
        raise FileNotFoundError(f"{exe} not found; pass --tools DIR or set W101_TOOLS")
    return found


def _run(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"{Path(cmd[0]).name} failed ({proc.returncode}): {proc.stderr.strip() or proc.stdout.strip()}")


def _unique_paths(proj: "Project", items: list) -> list:
    """Drop items whose file name matches an earlier one ignoring case: on Windows
    they are one file, and two texconv processes writing it at once collide."""
    seen, out = set(), []
    for t in items:
        k = os.path.normcase(proj.flat(t, ""))
        if k not in seen:
            seen.add(k)
            out.append(t)
    return out


def _try(fn, *args):
    """Run fn; return the exception instead of raising (for per-file batch work)."""
    try:
        fn(*args)
    except Exception as exc:  # noqa: BLE001 - reported per file by the caller
        return exc
    return None


def _pmap(fn, items) -> list:
    """Thread-parallel map for per-file checks (PIL, zlib and numpy release the GIL)."""
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(os.cpu_count() or 4) as pool:
        return list(pool.map(fn, items, chunksize=64))


def _run_many(cmds: list[list[str]], workers: int | None = None) -> None:
    """Run independent tool invocations side by side; the first failure is raised
    after the rest finish."""
    from concurrent.futures import ThreadPoolExecutor

    workers = workers or max(2, (os.cpu_count() or 4) // 2)
    with ThreadPoolExecutor(workers) as pool:
        for f in [pool.submit(_run, c) for c in cmds]:
            f.result()


class Project:
    """Working folder: src/ (original DDS), png/, up/ (upscaled PNG), dds/ (final), manifest.json."""

    def __init__(self, root: str | os.PathLike[str]):
        self.root = Path(root)
        self.src, self.png, self.up, self.dds = (self.root / d for d in ("src", "png", "up", "dds"))
        self.manifest_path = self.root / "manifest.json"

    def config(self) -> dict:
        """How this project was extracted (WADs, pattern, sizes), for rebuilds."""
        try:
            return json.loads((self.root / "project.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def load(self) -> list[TexInfo]:
        if not self.manifest_path.exists():
            return []
        return [TexInfo(**t) for t in json.loads(self.manifest_path.read_text())]

    def save(self, items: Iterable[TexInfo]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps([asdict(t) for t in items], indent=1))

    def flat(self, t: TexInfo, ext: str) -> str:
        # One flat namespace per stage keeps CLI tool batch calls simple.
        entry, _, block = t.name.partition("#")
        stem = entry.removesuffix(Path(entry).suffix).replace("/", "__") + (f"__b{block}" if block else "")
        if t.kind == "img":  # a WAD can hold Foo.dds and Foo.png side by side
            stem += "_" + Path(entry).suffix.lstrip(".").lower()
        return f"{Path(t.wad).stem}__{stem}{ext}"

    # -- stages -------------------------------------------------------------

    def _write_src(self, t: TexInfo, ext: str, data: bytes) -> None:
        """Store a texture's source; if it differs from what's there (the game
        updated it), drop every derived stage so the next run redoes it."""
        f = self.src / self.flat(t, ext)
        if f.exists() and f.read_bytes() != data:
            for d, e in ((self.png, ".png"), (self.up, ".png"), (self.dds, ".dds"), (self.dds, ".blk")):
                (d / self.flat(t, e)).unlink(missing_ok=True)
        f.write_bytes(data)

    def _extract_embedded(self, wad_name: str, entry: str, data: bytes, min_size: int) -> list[TexInfo]:
        """Every texture block in a NIF: colour art and cube-map faces for the
        model, bump/luma data for exact resampling. One TexInfo per block/face."""
        try:
            nif = NifFile.parse(data)
            blocks = nif.pixel_blocks()
        except (NifError, struct.error, IndexError):
            return []
        out = []
        for i in blocks:
            try:
                px = nif.pixels(i)
            except (NifError, struct.error, IndexError):
                continue
            if max(px.width, px.height) < min_size:
                continue
            if px.is_raw_resizable:
                t = TexInfo(f"{entry}#{i}", wad_name, px.width, px.height, len(px.mips), "RAW", "RAW", "nifraw")
                self._write_src(t, ".blk", struct.pack("<I?", px.version, px.persistent) + px.serialize())
                out.append(t)
                continue
            faces = [(f"{entry}#{i}@{k}", px.face(k)) for k in range(px.num_faces)] if px.num_faces > 1 else [(f"{entry}#{i}", px)]
            for name, fpx in faces:
                try:
                    if fpx.num_faces != 1 or not (fpx.is_color or fpx.pixel_format in (FMT_DXT1, FMT_DXT3, FMT_DXT5)):
                        continue
                    dds = fpx.to_dds()
                except (NifError, struct.error, IndexError):
                    continue
                w, h, mips, fourcc, _ = parse_dds_header(dds)
                dxgi = _NIF_DXGI.get(fpx.pixel_format, "BC3_UNORM" if fpx.has_alpha else "BC1_UNORM")
                t = TexInfo(name, wad_name, w, h, mips, fourcc, dxgi, "nifblock")
                self._write_src(t, ".dds", dds)
                out.append(t)
        return out

    def _extract_image(self, wad_name: str, entry: str, data: bytes, min_size: int) -> TexInfo | None:
        import io
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            im.load()
            fmt = _IMAGE_EXT[Path(entry).suffix.lower()]
            mode = "RGBA" if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info) else "RGB"
            if max(im.size) < min_size:
                return None
            t = TexInfo(entry, wad_name, im.width, im.height, 1, mode, fmt, "img")
            self._write_src(t, ".src", hashlib.sha1(data).digest())
            png = self.png / self.flat(t, ".png")
            if not png.exists():
                im.convert("RGBA").save(png, compress_level=1)
        return t

    def _extract_one(self, wad: Wad, wp: Path, e, items: dict, min_size: int, embedded: bool) -> None:
        if not wad.is_downloaded(e):  # streamed asset not fetched yet
            return
        data = wad.read(e)
        low = e.name.lower()
        if Path(low).suffix in _IMAGE_EXT:
            t = self._extract_image(wp.name, e.name, data, min_size)
            if t:
                items[(t.wad, t.name)] = t
            return
        if embedded and low.endswith(".nif") and not low.startswith("textures/"):
            for t in self._extract_embedded(wp.name, e.name, data, min_size):
                items[(t.wad, t.name)] = t
            return
        kind = "dds"
        if low.endswith(".nif"):
            try:
                px = TextureNif.parse(data).pixels
            except (NifError, struct.error):
                return  # a model, not a texture
            if not px.is_color:  # bump maps and cube maps go through the per-block path
                for t in self._extract_embedded(wp.name, e.name, data, min_size):
                    items[(t.wad, t.name)] = t
                return
            try:
                data = px.to_dds()
            except NifError:
                return
            kind = "nif"
        try:
            w, h, mips, fourcc, dxgi = parse_dds_header(data)
        except ValueError:
            return
        if max(w, h) < min_size:  # thin strips (16x256 trims) still count
            return
        if kind == "nif":
            dxgi = _NIF_DXGI.get(px.pixel_format, "BC3_UNORM" if px.has_alpha else "BC1_UNORM")
        t = TexInfo(e.name, wp.name, w, h, mips, fourcc, dxgi, kind)
        self._write_src(t, ".dds", data)
        items[(t.wad, t.name)] = t

    def extract(self, wad_paths: Iterable[Path], pattern: str = "*.dds", min_size: int = 64,
                embedded: bool = False) -> int:
        wad_paths = list(wad_paths)
        cfg = self.config()
        known = cfg.get("wads", [])
        cfg.update(pattern=pattern, min_size=min_size, embedded=embedded,
                   wads=["*"] if "*" in known else sorted(set(known) | {Path(w).name for w in wad_paths}))
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "project.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        items = {(t.wad, t.name): t for t in self.load()}
        self.src.mkdir(parents=True, exist_ok=True)
        self.png.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        for wp in wad_paths:
            with Wad(wp) as wad:
                for e in wad.glob(pattern):
                    try:
                        self._extract_one(wad, wp, e, items, min_size, embedded)
                    except Exception as exc:  # one bad file must not stop a 3,600-WAD run
                        errors.append(f"{wp.name}:{e.name}: {exc!r}")
        # A catch-all project (everything the others miss) keeps excluding what they cover.
        for other in cfg.get("exclude_projects", []):
            covered = {(t.wad, t.name) for t in Project(other).load()}
            for key in [k for k in items if k in covered]:
                del items[key]
        self.save(items.values())
        (self.root / "extract_errors.txt").write_text("\n".join(errors), encoding="utf-8")
        if errors:
            print(f"extract: {len(errors)} files skipped with errors, see {self.root / 'extract_errors.txt'}")
        return len(items)

    def upscale(self, scale: int = 4, model: str = "realesrgan-x4plus-anime", max_size: int = 8192,
                tools: str | None = None, gpu: str | None = None, backend: str = "torch",
                torch_model: Path | None = None, log=print,
                encode_scale: dict[str, int] | None = None) -> int:
        """Decode -> upscale -> validate every image -> encode. Raises if any image fails.

        ``backend="torch"`` (CUDA PyTorch, default) or ``"ncnn"`` (realesrgan-ncnn-vulkan,
        which can lose the Vulkan device on large inputs and write
        blanks). Either way nothing is encoded unless every output passes
        ``check_upscale``; outputs that already pass are kept (resume).

        ``encode_scale`` maps a WAD name to the factor its textures are encoded
        at (the model still runs at ``scale``; texconv downsamples, which beats
        upscaling straight to the lower factor). It keeps big archives under
        KIWAD's 4 GiB limit, e.g. {"Equipment-WorldData.wad": 2}.
        """
        self.encode_scale = encode_scale or {}
        if scale not in (2, 3, 4):
            raise ValueError("Real-ESRGAN supports native x2/x3/x4; chain runs for more")
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"no project at {self.root} (missing {self.manifest_path.name}); run extract first")
        texconv = find_tool("texconv.exe", tools)
        items = self.load()
        for d in (self.png, self.up, self.dds):
            d.mkdir(parents=True, exist_ok=True)
        # Bump/luma data isn't a picture: it's resampled exactly, not run through the model.
        raw = [t for t in items if t.kind == "nifraw"]
        items = [t for t in items if t.kind != "nifraw"]
        self.max_size = max_size

        # Same texture names exist in several WADs (e.g. Equipment-WorldData and
        # Equipment-HighDetail), so everything is keyed by "wad:name".
        def key(t: TexInfo) -> str:
            return f"{t.wad}:{t.name}"

        kept: dict[str, str] = {}
        # Already at or past the size cap: nothing to gain, don't spend GPU time on it.
        for t in items:
            if self._factor(t, scale, max_size) <= 1:
                kept[key(t)] = f"{t.width}x{t.height} is already at the {max_size}px cap"
        items = [t for t in items if key(t) not in kept]
        missing_png = [str(self.src / self.flat(t, ".dds")) for t in items if not (self.png / self.flat(t, ".png")).exists()]
        # Windows caps a command line at 32767 chars; batch the decode. A batch
        # that fails retries file by file so one odd texture can't stop the run.
        undecodable: dict[str, str] = {}
        for i in range(0, len(missing_png), 100):
            batch = missing_png[i:i + 100]
            try:
                _run([texconv, "-nologo", "-y", "-ft", "png", "-o", str(self.png), *batch])
            except RuntimeError:
                for f in batch:
                    try:
                        _run([texconv, "-nologo", "-y", "-ft", "png", "-o", str(self.png), f])
                    except RuntimeError as exc:
                        undecodable[f] = str(exc)
        if undecodable:
            for t in items:
                reason = undecodable.get(str(self.src / self.flat(t, ".dds")))
                if reason:
                    kept[key(t)] = f"decode failed: {reason}"
            items = [t for t in items if key(t) not in kept]
            log(f"{len(kept)} textures couldn't be decoded and keep their original")

        def failures(ts) -> dict[str, str]:
            def one(t):
                up = self.up / self.flat(t, ".png")
                if up.exists():
                    return key(t), check_upscale(self.png / self.flat(t, ".png"), up, scale)
                # The upscaled PNGs may have been deleted to save space: a finished,
                # valid DDS at the right size means this texture is done.
                f = self.dds / self.flat(t, ".dds")
                if (f.exists() and _out_size(f)[0] == t.width * self._factor(t, scale, max_size)
                        and self._check_dds(t, f) is None):
                    return key(t), None
                return key(t), "missing"
            return {k: r for k, r in _pmap(one, ts) if r}

        initial = failures(items)
        todo = [t for t in items if key(t) in initial]
        if todo and len(todo) < len(items):
            # A twin of a pending image may already be done (an interrupted run
            # stops before twins are copied); reuse that instead of re-running it.
            def digest(t: TexInfo) -> bytes:
                return hashlib.sha1((self.png / self.flat(t, ".png")).read_bytes()).digest()
            done = {d: t for t, d in zip(items, _pmap(digest, items))
                    if key(t) not in initial and (self.up / self.flat(t, ".png")).exists()}
            reused = 0
            for t, d in zip(todo, _pmap(digest, todo)):
                src = done.get(d)
                if src is not None:
                    dst = self.up / self.flat(t, ".png")
                    if os.path.normcase(dst) != os.path.normcase(self.up / self.flat(src, ".png")):
                        shutil.copyfile(self.up / self.flat(src, ".png"), dst)
                    reused += 1
            if reused:
                initial = failures(todo)
                todo = [t for t in todo if key(t) in initial]
                log(f"{reused} reused from finished twins")
        log(f"{len(items) - len(todo)} upscales already valid, {len(todo)} to do")
        # Last resorts are plain Lanczos, then bilinear (no ringing on hard alpha
        # edges): every texture ends up at the target size, AI where faithful.
        alpha_modes = {1: "model", 2: "lanczos", 3: "resample", 4: "bilinear", 5: "nearest"}
        for attempt in alpha_modes:
            if not todo:
                break
            for t in todo:
                (self.up / self.flat(t, ".png")).unlink(missing_ok=True)
            if backend == "torch":
                from .torch_upscale import DEFAULT_MODEL, TorchUpscaler
                up = TorchUpscaler(torch_model or DEFAULT_MODEL)
                if up.scale != scale:
                    raise ValueError(f"model is x{up.scale}, asked for x{scale}")
                # The same texture is copied into many WADs and models; upscale
                # each distinct image once and copy the result to its twins.
                groups: dict[bytes, list[TexInfo]] = {}
                for t in todo:
                    digest = hashlib.sha1((self.png / self.flat(t, ".png")).read_bytes()).digest()
                    groups.setdefault(digest, []).append(t)
                # Size order puts same-shape images together so they batch on the GPU.
                reps = sorted((g[0] for g in groups.values()), key=lambda t: (t.height, t.width))
                if len(reps) < len(todo):
                    log(f"  {len(todo)} images, {len(reps)} distinct")
                step = max(len(reps) // 20, 1)
                # Second pass keeps the AI color but resizes the source alpha
                # conventionally (the network solidifies faint strokes).
                up.upscale_files(((self.png / self.flat(t, ".png"), self.up / self.flat(t, ".png")) for t in reps),
                                 alpha_mode=alpha_modes[attempt],
                                 progress=lambda i, _p: (i + 1) % step == 0 and log(f"  {i + 1}/{len(reps)}"))
                for g in groups.values():
                    src = self.up / self.flat(g[0], ".png")
                    for t in g[1:]:
                        dst = self.up / self.flat(t, ".png")
                        # Names that differ only in case are one file on Windows.
                        if src.exists() and os.path.normcase(dst) != os.path.normcase(src):
                            shutil.copyfile(src, dst)
            else:
                esrgan = find_tool("realesrgan-ncnn-vulkan.exe", tools)
                for t in todo:  # one file per process: a lost device can't poison the rest
                    cmd = [esrgan, "-i", str(self.png / self.flat(t, ".png")), "-o", str(self.up / self.flat(t, ".png")),
                           "-n", model, "-s", str(scale), "-f", "png"] + (["-g", gpu] if gpu is not None else [])
                    _run(cmd)
            bad = failures(todo)
            todo = [t for t in todo if key(t) in bad]
            if todo:
                log(f"attempt {attempt}: {len(todo)} outputs failed validation, e.g. {next(iter(bad.items()))}")
        # Anything still failing keeps its original texture: the mod never ships an
        # image that doesn't faithfully match the source, and one stubborn icon
        # doesn't block the other 17k.
        kept.update(failures(todo) if todo else {})
        for t in todo:
            (self.dds / self.flat(t, ".dds")).unlink(missing_ok=True)
        items = [t for t in items if key(t) not in kept]

        # Gate on what actually ships: decode every DDS and re-check. Ones that
        # already pass (from an interrupted run) aren't re-encoded. BC2's 4-bit
        # alpha can tip a borderline image over; those get the conventional-alpha
        # fallback and are re-encoded.
        def check(t: TexInfo) -> str | None:
            f = self.dds / self.flat(t, ".dds")
            r = self._check_dds(t, f)
            if r is None and _out_size(f)[0] != t.width * self._factor(t, scale, max_size):
                return "encoded at a different scale"
            return r
        stale = [t for t, r in zip(items, _pmap(check, items)) if r]
        log(f"{len(items) - len(stale)} encoded textures already valid, {len(stale)} to encode")
        self._encode(texconv, stale, scale, max_size)
        bad = {key(t): r for t, r in zip(stale, _pmap(check, stale)) if r}
        for mode in ("lanczos",):
            if not bad or backend != "torch":
                break
            log(f"{len(bad)} encoded textures failed; re-doing them with {mode}-resized source alpha")
            from .torch_upscale import DEFAULT_MODEL, TorchUpscaler
            redo = [t for t in items if key(t) in bad]
            TorchUpscaler(torch_model or DEFAULT_MODEL).upscale_files(
                ((self.png / self.flat(t, ".png"), self.up / self.flat(t, ".png")) for t in redo), alpha_mode=mode)
            self._encode(texconv, redo, scale, max_size)
            bad = {key(t): r for t, r in zip(redo, _pmap(lambda t: self._check_dds(t, self.dds / self.flat(t, ".dds")), redo)) if r}
        for t in items:  # never leave a bad DDS where pack could pick it up
            if key(t) in bad:
                (self.dds / self.flat(t, ".dds")).unlink(missing_ok=True)
        kept.update(bad)
        raw_bad = self._resample_raw(raw, scale, max_size)
        if raw:
            log(f"{len(raw) - len(raw_bad)} bump/luma maps resampled")
        kept.update(raw_bad)
        report = self.root / "kept_original.json"
        report.write_text(json.dumps(kept, indent=1))
        if kept:
            log(f"{len(kept)} textures keep their original (couldn't be upscaled faithfully): see {report}")
        return len(items) - len(bad) + len(raw) - len(raw_bad)

    def _factor(self, t: TexInfo, scale: int, max_size: int) -> int:
        # Cap the long edge by halving, so power-of-two sources stay power-of-two.
        factor = min(scale, getattr(self, "encode_scale", {}).get(t.wad, scale))
        while max(t.width, t.height) * factor > max_size and factor > 1:
            factor //= 2
        return factor

    def _resample_raw(self, items: list[TexInfo], scale: int, max_size: int, out: Path | None = None,
                      factor: int | None = None) -> dict[str, str]:
        """Write each bump/luma block resampled to <flat>.blk; returns failures."""
        from ..nif import PixelData
        out = out or self.dds
        bad: dict[str, str] = {}

        def one(t: TexInfo) -> None:
            f = out / self.flat(t, ".blk")
            src = (self.src / self.flat(t, ".blk")).read_bytes()
            version, persistent = struct.unpack_from("<I?", src, 0)
            px = PixelData.parse(src[5:], version, persistent)
            fac = factor or self._factor(t, scale, max_size)
            if fac <= 1:
                bad[f"{t.wad}:{t.name}"] = "capped to 1x"
                return
            if f.exists() and _out_size(f) == (t.width * fac, t.height * fac):
                return
            f.write_bytes(src[:5] + px.resampled(fac).serialize())
        for t, exc in zip(items, _pmap(lambda t: _try(one, t), items)):
            if exc:
                bad[f"{t.wad}:{t.name}"] = f"resample failed: {exc}"
        return bad

    def _encode(self, texconv: str, items: list[TexInfo], scale: int, max_size: int,
                out: Path | None = None, factor: int | None = None) -> None:
        out = out or self.dds
        # Group by (format, target size, mips) so texconv encodes 100 files per process.
        groups: dict[tuple[str, int, int, str], list[str]] = {}
        images: list[tuple[TexInfo, int]] = []
        for t in _unique_paths(self, items):
            fac = factor or self._factor(t, scale, max_size)
            if t.kind == "img":
                images.append((t, fac))
                continue
            # Cube faces keep their single level; everything else gets a full chain.
            mips = "1" if "@" in t.name and t.mips == 1 else "0"
            groups.setdefault((t.dxgi, t.width * fac, t.height * fac, mips), []).append(str(self.up / self.flat(t, ".png")))
        _run_many([[texconv, "-nologo", "-y", "-f", fmt, "-m", mips, "-w", str(w), "-h", str(h),
                     "-if", "LINEAR", "-sepalpha", "-dx9", "-o", str(out), *files[i:i + 100]]
                    for (fmt, w, h, mips), files in groups.items() for i in range(0, len(files), 100)])
        _pmap(lambda tf: self._save_image(tf[0], tf[1], out), images)

    def _save_image(self, t: TexInfo, factor: int, out: Path) -> None:
        """Loose image re-saved in its own format at ``factor``x."""
        from PIL import Image
        with Image.open(self.up / self.flat(t, ".png")) as im:
            im = im.convert("RGBA")
            size = (t.width * factor, t.height * factor)
            if im.size != size:
                im = im.resize(size, Image.LANCZOS)
            if t.fourcc == "RGB" or t.dxgi in ("JPEG", "BMP"):
                im = im.convert("RGB")
            kw = {"PNG": {"compress_level": 6}, "JPEG": {"quality": 95, "subsampling": 0},
                  "TGA": {"rle": True}}.get(t.dxgi, {})
            if t.dxgi == "GIF":
                im = im.convert("RGB").convert("P", palette=Image.ADAPTIVE)
            im.save(out / self.flat(t, ".dds"), format=t.dxgi, **kw)

    def _check_dds(self, t: TexInfo, f: Path) -> str | None:
        from PIL import Image

        if t.kind == "nifraw":
            try:
                w, h = _out_size(f)
            except Exception as exc:
                return f"unreadable block: {exc}"
            return None if w % t.width == 0 and w // t.width == h // t.height and w > t.width else f"unexpected size {w}x{h}"
        try:
            with Image.open(f) as im:
                im.load()
                out = im.convert("RGBA")
        except Exception as exc:
            return f"undecodable DDS: {exc}"
        if out.width % t.width or out.width // t.width != out.height // t.height:
            return f"unexpected size {out.size} for {t.width}x{t.height} source"
        return check_upscale(self.png / self.flat(t, ".png"), out, out.width // t.width)

    def _build_nifs(self, src_wad: Path, wad_name: str, dds_files: dict[str, Path]) -> dict[str, Path]:
        """Rebuild each texture NIF from its original with the upscaled pixels,
        written to nif/ (thousands of 4x textures don't fit in memory at once).
        Each result is re-parsed to prove it holds exactly the encoded DDS."""
        out_dir = self.root / "nif"
        out_dir.mkdir(exist_ok=True)
        built: dict[str, Path] = {}
        with Wad(src_wad) as wad:
            for name, dds_path in dds_files.items():
                dds = dds_path.read_bytes()
                new = TextureNif.parse(wad.read(name)).with_dds(dds).serialize()
                back = TextureNif.parse(new).pixels
                if back.data != dds[128:128 + len(back.data)] or (back.width, back.height) != parse_dds_header(dds)[:2]:
                    raise WadError(f"rebuilt NIF for {name} doesn't match its DDS")
                target = out_dir / (Path(dds_path.name).stem + ".nif")
                target.write_bytes(new)
                built[name] = target
        return built

    def _build_embedded(self, src_wad: Path, blocks: dict[str, Path]) -> dict[str, Path]:
        """Rebuild each model NIF once with all its upscaled embedded textures."""
        from ..nif import PixelData
        by_entry: dict[str, dict[int, dict[int, Path]]] = {}
        for name, path in blocks.items():
            entry, _, rest = name.partition("#")
            i, _, face = rest.partition("@")
            by_entry.setdefault(entry, {}).setdefault(int(i), {})[int(face) if face else -1] = path
        out_dir = self.root / "nif"
        out_dir.mkdir(exist_ok=True)
        built: dict[str, Path] = {}
        with Wad(src_wad) as wad:
            for entry, per_block in by_entry.items():
                nif = NifFile.parse(wad.read(entry))
                expect: dict[int, bytes] = {}
                for i, parts in per_block.items():
                    px = nif.pixels(i)
                    if -1 in parts and parts[-1].suffix == ".blk":
                        blk = parts[-1].read_bytes()
                        new_px = PixelData.parse(blk[5:], px.version, px.persistent)
                    elif -1 in parts:
                        new_px = px.with_dds(parts[-1].read_bytes())
                    else:
                        if sorted(parts) != list(range(px.num_faces)):
                            continue  # a cube map is all six faces or untouched
                        new_px = px.with_faces([px.face(k).with_dds(parts[k].read_bytes()) for k in range(px.num_faces)])
                    nif.set_pixels(i, new_px)
                    expect[i] = new_px.data
                new = nif.serialize()
                back = NifFile.parse(new)
                for i, data in expect.items():
                    if back.pixels(i).data != data:
                        raise WadError(f"rebuilt {entry} block {i} doesn't match its encoded texture")
                target = out_dir / (entry.replace("/", "__") + ".model")
                target.write_bytes(new)
                built[entry] = target
        return built

    def _encode_at(self, texconv: str, items: list[TexInfo], factor: int) -> dict[str, Path]:
        """Encode items' upscaled PNGs at a lower ``factor`` into dds_x<factor>/,
        validated; returns wad:name -> path for the ones that pass."""
        out = self.root / f"dds_x{factor}"
        out.mkdir(exist_ok=True)
        raw = [t for t in items if t.kind == "nifraw"]
        self._encode(texconv, [t for t in items if t.kind != "nifraw"], factor, 1 << 30, out=out, factor=factor)
        self._resample_raw(raw, factor, 1 << 30, out=out, factor=factor)
        ok = {}
        for t in items:
            f = out / self.flat(t, ".blk" if t.kind == "nifraw" else ".dds")
            if f.exists() and not self._check_dds(t, f):
                ok[f"{t.wad}:{t.name}"] = f
        return ok

    def pack(self, game_data: Path, out_dir: Path, pattern: str = "*", keep_crc: bool = False,
             budget: int = MAX_WAD_BYTES - 64 * 2**20, tools: str | None = None, log=print) -> list[Path]:
        """Write modified copies of each touched WAD into out_dir (see pack_projects)."""
        return pack_projects([self], [game_data], out_dir, pattern=pattern, budget=budget, tools=tools, log=log)


@dataclass
class _Unit:
    proj: Project
    t: TexInfo
    path: Path          # DDS currently chosen
    factor: int
    new_c: int = 0      # compressed size at the chosen factor
    orig_c: int = 0     # compressed size of the original texture


def pack_projects(projects: list[Project], sources: list[Path], out_dir: Path, pattern: str = "*",
                  budget: int = MAX_WAD_BYTES - 64 * 2**20, tools: str | None = None, log=print,
                  record_against: Path | None = None, wads: set[str] | None = None,
                  max_factor: int = 4, max_source: int | None = None,
                  overlays: bool = True, locale: str = "en-US", extended: bool = False) -> list[Path]:
    """Pack every project's validated textures into one copy of each WAD.

    Several projects can feed the same archive (interface DDS from one, world
    .nif textures from another); they're merged per WAD. ``sources`` are
    searched in order for each WAD's original (e.g. the full download first,
    then the install). Each archive is kept under ``budget`` (the client reads
    offsets as signed 32-bit, so 2 GiB minus a margin) by stepping the
    textures with the largest compressed size down a level, 4x -> 2x ->
    original, so small art keeps full quality. Size is estimated as
    original WAD size + sum(new - original texture size); if the real archive
    still overshoots, the budget tightens and it refits.

    With ``overlays`` (default), a package that doesn't fit spills whole entries
    into a second archive, ``Locale_<locale>-<package>.wad``: the client checks
    that overlay before the package itself (registered through Overrides.xml in
    the Root overlay), so each package gets two 2 GiB archives before anything
    has to step down to 2x.
    """
    import zlib
    from concurrent.futures import ThreadPoolExecutor

    from ..wad import EXTENDED_WAD_BYTES
    # Extended archives need the patched client (swap.patch_client_bytes): 4 GiB each.
    max_bytes = EXTENDED_WAD_BYTES if extended else MAX_WAD_BYTES
    if extended and budget == MAX_WAD_BYTES - 64 * 2**20:
        budget = EXTENDED_WAD_BYTES - 128 * 2**20
    pat = pattern.lower()
    units: dict[str, list[_Unit]] = {}
    bad: dict[str, str] = {}
    for proj in projects:
        cands = [(t, proj.dds / proj.flat(t, ".blk" if t.kind == "nifraw" else ".dds")) for t in proj.load()
                 if fnmatch.fnmatchcase(t.name.lower(), pat)]
        cands = [(t, f) for t, f in cands if f.exists()]

        # Final gate on the exact bytes going into the archive.
        def gate(tf, proj=proj):
            t, f = tf
            reason = proj._check_dds(t, f)
            if reason:
                return reason, 0
            return None, _out_size(f)[0] // t.width
        for (t, f), (reason, fac) in zip(cands, _pmap(gate, cands)):
            if reason:
                bad[f"{t.wad}:{t.name}"] = reason
            elif fac > 1 and (max_source is None or max(t.width, t.height) <= max_source):
                # (capped back to 1x is no upscale; past a preset's source limit the original stays)
                units.setdefault(t.wad, []).append(_Unit(proj, t, f, fac))
        log(f"{proj.root.name}: {len(cands)} encoded textures checked")
    if bad:
        report = projects[0].root / "pack_failures.json"
        report.write_text(json.dumps(bad, indent=1))
        raise WadError(f"{len(bad)} encoded textures failed validation; nothing packed. See {report}")

    def csize(path: Path) -> int:
        return len(zlib.compress(path.read_bytes(), 6))

    out_dir.mkdir(parents=True, exist_ok=True)
    texconv: str | None = None
    written = []
    index: dict[str, list] = {}
    for wad_name, us in sorted(units.items()):
        if wads is not None and wad_name not in wads:
            continue
        src = next((d / wad_name for d in sources if (d / wad_name).exists()), None)
        if src is None:
            raise WadError(f"no source for {wad_name} in {sources}")
        dest = out_dir / wad_name
        if dest.resolve() == src.resolve():
            raise WadError("refusing to overwrite the source WAD; pick another out_dir")
        with Wad(src) as wad:
            stored = {e.name: e.stored_size for e in wad.entries}
        wad_size = src.stat().st_size

        def orig_size(u: _Unit) -> int:
            if u.t.kind == "nifblock":
                return csize(u.proj.src / u.proj.flat(u.t, ".dds"))
            if u.t.kind == "nifraw":
                return csize(u.proj.src / u.proj.flat(u.t, ".blk"))
            return stored[u.t.name]

        with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
            news = list(pool.map(lambda u: csize(u.path), us))
            origs = list(pool.map(orig_size, us))
        for u, n, o in zip(us, news, origs):
            u.new_c, u.orig_c = n, o
        # Quality preset: anything above the preset's scale is re-encoded down to it first.
        over_cap = [u for u in us if u.factor > max_factor and "@" not in u.t.name]
        if over_cap:
            texconv = texconv or find_tool("texconv.exe", tools)
            by_proj: dict[int, list[_Unit]] = {}
            for u in over_cap:
                by_proj.setdefault(id(u.proj), []).append(u)
            dropped = set()
            for group in by_proj.values():
                ok = group[0].proj._encode_at(texconv, [u.t for u in group], max_factor)
                for u in group:
                    f = ok.get(f"{u.t.wad}:{u.t.name}")
                    if f is None:
                        dropped.add(id(u))
                    else:
                        u.path, u.factor, u.new_c = f, max_factor, csize(f)
            us = [u for u in us if id(u) not in dropped]
        live = list(us)  # units still replacing their original
        limit = budget
        ovl_name = f"Locale_{locale}-{Path(wad_name).stem}.wad"
        ovl_dest = out_dir / ovl_name

        def entry(u: _Unit) -> str:
            return u.t.name.partition("#")[0]

        for attempt in range(4):
            in_ovl: set[str] = set()   # entries (whole models / textures) moved to the overlay

            def base_est() -> int:
                return wad_size + sum(u.new_c - u.orig_c for u in live if entry(u) not in in_ovl)

            def ovl_est() -> int:
                grow: dict[str, int] = {}
                for u in live:
                    if entry(u) in in_ovl:
                        grow[entry(u)] = grow.get(entry(u), 0) + u.new_c - u.orig_c
                return sum(stored.get(e, 0) + g for e, g in grow.items()) + 4 * 2**20

            if overlays and base_est() > limit:
                # Spill the entries that grow the most until the base fits.
                growth: dict[str, int] = {}
                for u in live:
                    growth[entry(u)] = growth.get(entry(u), 0) + u.new_c - u.orig_c
                for e, _g in sorted(growth.items(), key=lambda kv: kv[1], reverse=True):
                    if base_est() <= limit:
                        break
                    in_ovl.add(e)
            log(f"{wad_name}: est {base_est() / 2**30:.2f} GiB"
                + (f" + overlay {ovl_est() / 2**30:.2f} GiB" if in_ovl else "")
                + f", budget {limit / 2**30:.2f} GiB each, {len(live)} textures")
            # Whatever still doesn't fit (the base without overlays, or an
            # overflowing overlay) steps its largest textures down a level.
            for scope_name, in_scope, est_fn in (
                    ("overlay", lambda u: entry(u) in in_ovl, ovl_est),
                    ("base", lambda u: entry(u) not in in_ovl, base_est)):
                while est_fn() > limit:
                    texconv = texconv or find_tool("texconv.exe", tools)
                    over, saved, plan = est_fn() - limit, 0, []
                    for u in sorted((u for u in live if in_scope(u)), key=lambda u: u.new_c, reverse=True):
                        if saved >= over * 1.05:
                            break
                        if "@" in u.t.name:  # cube faces must stay the same size as each other
                            continue
                        nxt = u.factor // 2
                        gain = u.new_c - (u.orig_c if nxt <= 1 else u.new_c // 4)
                        if gain <= 0:
                            continue
                        plan.append((u, nxt))
                        saved += gain
                    if not plan:
                        raise WadError(f"{wad_name} ({scope_name}) can't fit {limit} bytes even with originals")
                    drop = {id(u) for u, nxt in plan if nxt <= 1}
                    lower: dict[tuple[int, int], list[_Unit]] = {}
                    for u, nxt in plan:
                        if nxt > 1:
                            lower.setdefault((id(u.proj), nxt), []).append(u)
                    for (_, nxt), group in lower.items():
                        ok = group[0].proj._encode_at(texconv, [u.t for u in group], nxt)
                        for u in group:
                            f = ok.get(f"{u.t.wad}:{u.t.name}")
                            if f is None:  # lower scale failed validation: fall back to the original
                                drop.add(id(u))
                                continue
                            u.path, u.factor, u.new_c = f, nxt, csize(f)
                    live = [u for u in live if id(u) not in drop]
                    log(f"  {scope_name}: stepped {len(plan)} down, est {est_fn() / 2**30:.2f} GiB")
            repl: dict[str, Path] = {}
            by_proj: dict[int, tuple[Project, dict[str, dict[str, Path]]]] = {}
            for u in live:
                proj, kinds = by_proj.setdefault(id(u.proj), (u.proj, {}))
                kinds.setdefault(u.t.kind, {})[u.t.name] = u.path
            # Blocks of one model can come from several projects (colour art from
            # one, cube faces or bump data from another): rebuild each model once.
            embedded: dict[str, Path] = {}
            for proj, kinds in by_proj.values():
                repl.update(kinds.get("dds", {}))
                repl.update(kinds.get("img", {}))
                if kinds.get("nif"):
                    repl.update(proj._build_nifs(src, wad_name, kinds["nif"]))
                embedded.update(kinds.get("nifblock", {}))
                embedded.update(kinds.get("nifraw", {}))
            if embedded:
                repl.update(next(iter(by_proj.values()))[0]._build_embedded(src, embedded))
            base_repl = {k: v for k, v in repl.items() if k not in in_ovl}
            ovl_files = sorted((k, v) for k, v in repl.items() if k in in_ovl)
            try:
                repack(src, dest, base_repl, max_bytes=max_bytes)
                if ovl_files:
                    if wad_name.lower() == "root.wad":  # Root's overlay also carries Overrides.xml
                        ovl_files.append(("Overrides.xml", overrides_xml([])))
                    write_wad(ovl_dest, ovl_files, max_bytes=max_bytes)
                    if ovl_dest.stat().st_size > max_bytes:
                        raise WadError(f"{ovl_name} is past the archive limit")
                else:
                    ovl_dest.unlink(missing_ok=True)
                # Rebuilt model files are only an input to this archive; drop them
                # (they reached ~94 GB for the whole game).
                for path in repl.values():
                    if path.parent.name == "nif":
                        path.unlink(missing_ok=True)
                break
            except WadError as exc:
                if "archive limit" not in str(exc) or attempt == 3:
                    raise
                limit -= 128 * 2**20
                log(f"  real archive overshot the estimate; refitting under {limit / 2**30:.2f} GiB")
        counts: dict[int, int] = {}
        for u in live:
            counts[u.factor] = counts.get(u.factor, 0) + 1
        log(f"{wad_name}: wrote {dest.stat().st_size / 2**30:.2f} GiB"
            + (f" + {ovl_name} {ovl_dest.stat().st_size / 2**30:.2f} GiB" if ovl_dest.exists() else "")
            + f"; scale mix {dict(sorted(counts.items(), reverse=True))}, {len(us) - len(live)} kept original")
        target = record_against / wad_name if record_against and (record_against / wad_name).exists() else src
        record_source(dest, target)
        written.append(dest)
        if ovl_dest.exists():
            record_source(ovl_dest, target)  # an overlay is stale exactly when its package is
            written.append(ovl_dest)
        index[wad_name] = [[u.t.name, u.t.kind, u.t.width, u.t.height, u.factor] for u in live]
    # What the mod changes, per archive: before/after viewers and
    # memory estimates read this. Archives packed earlier keep their entries.
    idx_path = out_dir / "textures.json"
    try:
        old = json.loads(idx_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = {}
    old.update(index)
    idx_path.write_text(json.dumps(old, separators=(",", ":")), encoding="utf-8")
    if overlays:
        write_overrides(out_dir, locale)
    return written


def overrides_xml(packages: list[str]) -> bytes:
    """Overrides.xml registering ``Locale_<locale>-<package>.wad`` overlays."""
    items = "".join(
        f'    <m_localeMappingList key="{i}">\n      <Class Name="class LocaleMapping">\n'
        f"        <m_wadFile>{p}</m_wadFile>\n        <m_fullReplacement>0</m_fullReplacement>\n"
        f"      </Class>\n    </m_localeMappingList>\n" for i, p in enumerate(packages))
    return ('<?xml version="1.0" ?>\n<Objects>\n  <Class Name="class LocaleManager">\n'
            f"{items}  </Class>\n</Objects>\n").encode()


def write_overrides(mod_dir: Path, locale: str = "en-US") -> Path | None:
    """(Re)write the Root overlay so its Overrides.xml lists every package overlay in
    ``mod_dir``, keeping any Root textures already in it. Removes it when unused."""
    prefix = f"Locale_{locale}-"
    root_ovl = mod_dir / f"{prefix}Root.wad"
    packages = sorted(p.stem[len(prefix):] for p in mod_dir.glob(f"{prefix}*.wad")
                      if p.name.lower() != root_ovl.name.lower())
    keep: list = []
    if root_ovl.exists():
        with Wad(root_ovl) as w:
            keep = [(e.name, w.read(e)) for e in w.entries if e.name != "Overrides.xml"]
    if not packages and not keep:
        root_ovl.unlink(missing_ok=True)
        return None
    write_wad(root_ovl, keep + [("Overrides.xml", overrides_xml(packages))])
    if (GAME_DATA / "Root.wad").exists():
        record_source(root_ovl, GAME_DATA / "Root.wad")
    return root_ovl
