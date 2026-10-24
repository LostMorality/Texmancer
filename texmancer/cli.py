"""Command line: python -m texmancer <extract|upscale|pack|build|play|restore|status|wad> ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import GAME_DATA


def _wads(a: argparse.Namespace) -> list[Path]:
    game = Path(a.game_data)
    if a.all:
        return sorted(game.glob("*.wad"))
    if not a.wads:
        raise SystemExit("name some WADs (e.g. Mob-WorldData.wad) or pass --all")
    return [game / w for w in a.wads]


def cmd_extract(a: argparse.Namespace) -> int:
    from .graphics import Project
    p = Project(a.work)
    n = p.extract(_wads(a), pattern=a.pattern, min_size=a.min_size, embedded=a.embedded, max_factor=a.max_factor)
    print(f"{n} textures in {a.work}")
    return 0


def _engine(a: argparse.Namespace) -> tuple[str, dict]:
    """Backend and power settings: the given ones, else picked for the best graphics card."""
    from . import hardware
    gpus = hardware.detect()
    gpu = gpus[0] if gpus else None
    backend = a.backend or (gpu.backend if gpu else "torch")
    power = hardware.power_settings(a.power, gpu)
    if power.get("priority") == "below_normal":
        hardware.set_low_priority()
    print(f"{gpu.describe() if gpu else 'no graphics card found'}; backend {backend}, power {a.power}")
    return backend, power


def cmd_upscale(a: argparse.Namespace) -> int:
    from .graphics import Project
    backend, power = _engine(a)
    n = Project(a.work).upscale(scale=a.scale, tools=a.tools, max_size=a.max_size, backend=backend, power=power,
                                torch_model=Path(a.model) if a.model else None)
    print(f"upscaled {n} textures")
    return 0


def cmd_pack(a: argparse.Namespace) -> int:
    from .graphics import Project
    from .graphics.build import PRESETS
    from .graphics.upscale import pack_projects
    preset = PRESETS[a.preset]
    out = pack_projects([Project(w) for w in a.work], [Path(a.game_data)], Path(a.out), tools=a.tools,
                        record_against=Path(a.game_data), max_factor=preset["max_factor"],
                        max_source=preset["max_source"], overlays=not a.no_overlays, locale=a.locale,
                        extended=a.extended)
    print(f"wrote {len(out)} archives to {a.out}")
    return 0


def cmd_build(a: argparse.Namespace) -> int:
    from .graphics.build import build_mod
    backend, power = _engine(a)
    build_mod(Path(a.mod), preset=a.preset, everything=a.all, tools=a.tools, game_data=Path(a.game_data),
              backend=backend, power=power, torch_model=Path(a.model) if a.model else None)
    return 0


def _swapper(a: argparse.Namespace, **kw):
    from .swap import Swapper
    game_data = Path(a.game_data)
    return Swapper(Path(a.mods), game_data, install=game_data.parent.parent, **kw)


def cmd_play(a: argparse.Namespace) -> int:
    s = _swapper(a, patch_client=a.extended_archives)
    print("Start Wizard101 and log in. Mods swap in when the launcher finishes patching;"
          " the originals come back when the game closes. Ctrl+C to stop.")
    s.run(once=a.once)
    return 0


def cmd_restore(a: argparse.Namespace) -> int:
    _swapper(a).restore()
    return 0


def cmd_status(a: argparse.Namespace) -> int:
    s = _swapper(a, log=lambda m: None)
    print("mods swapped in" if s.is_swapped() else "originals in place")
    for m in s.all_mods():
        print(f"  {s.label(m)}: {len(s.wads(m))} archives" + ("  (game updated: rebuild)" if s.is_stale(m) else ""))
    return 0


def cmd_wad(a: argparse.Namespace) -> int:
    from .wad import Wad
    path = Path(a.path)
    if not path.exists() and (Path(a.game_data) / path).exists():
        path = Path(a.game_data) / path
    with Wad(path) as wad:
        if a.action == "ls":
            for e in wad.glob(a.pattern):
                print(f"{e.size:>11}  {'z' if e.is_zip else ' '}  {e.name}")
            print(f"-- {len(wad)} entries, KIWAD v{wad.version}", file=sys.stderr)
        else:
            n, skipped = wad.extract(a.out, a.pattern)
            print(f"extracted {n} files -> {a.out}" + (f" ({skipped} not downloaded yet)" if skipped else ""))
    return 0


def _engine_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--backend", choices=["torch", "ncnn"],
                   help="torch = CUDA (NVIDIA), ncnn = Vulkan (any card); default: picked for your card")
    p.add_argument("--power", default="balanced", choices=["low", "balanced", "max"],
                   help="low leaves the game playable while it works")
    p.add_argument("--model", help="path to RealESRGAN_x4plus_anime_6B.pth (torch backend)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="texmancer", description="AI-upscaled textures for Wizard101.")
    p.add_argument("--game-data", default=str(GAME_DATA), help="Wizard101's Data\\GameData folder")
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="pull textures out of game archives into a work folder")
    e.add_argument("--work", required=True, help="work folder for this set of textures")
    e.add_argument("wads", nargs="*", help="archive names, e.g. Mob-WorldData.wad")
    e.add_argument("--all", action="store_true", help="every archive in GameData")
    e.add_argument("--pattern", default="*", help="entry name filter, e.g. *.dds or *.nif")
    e.add_argument("--min-size", type=int, default=1, help="skip textures whose long edge is smaller")
    e.add_argument("--embedded", action="store_true", help="also textures stored inside model files")
    e.add_argument("--max-factor", type=int, choices=[2, 4],
                   help="largest scale to encode at (new work folders default to 2; 4 is for 4K screens)")
    e.set_defaults(fn=cmd_extract)

    u = sub.add_parser("upscale", help="AI-upscale, validate and encode a work folder")
    u.add_argument("--work", required=True)
    u.add_argument("--scale", type=int, default=4, choices=[2, 3, 4])
    u.add_argument("--max-size", type=int, default=8192, help="largest output edge in pixels")
    u.add_argument("--tools", help="folder containing texconv.exe (and realesrgan-ncnn-vulkan.exe)")
    _engine_args(u)
    u.set_defaults(fn=cmd_upscale)

    k = sub.add_parser("pack", help="pack one or more work folders into a mod folder of archives")
    k.add_argument("--work", action="append", required=True, help="work folder (repeatable)")
    k.add_argument("--out", required=True, help="mod folder to write")
    k.add_argument("--preset", default="balanced", choices=["full", "balanced", "light"],
                   help="balanced = 2x (default), full = 4x (only worth it on 4K screens), light = 2x small art only")
    k.add_argument("--no-overlays", action="store_true", help="don't spill into Locale_* overlay archives")
    k.add_argument("--extended", action="store_true",
                   help="pack archives up to 4 GiB; needs `play --extended-archives` (see README)")
    k.add_argument("--locale", default="en-US", help="game locale, as in the client log's LOCALE line")
    k.add_argument("--tools", help="folder containing texconv.exe")
    k.set_defaults(fn=cmd_pack)

    b = sub.add_parser("build", help="rebuild a mod after a game update, or for another preset")
    b.add_argument("--mod", required=True)
    b.add_argument("--preset", choices=["full", "balanced", "light"])
    b.add_argument("--all", action="store_true", help="rebuild every archive")
    b.add_argument("--tools")
    _engine_args(b)
    b.set_defaults(fn=cmd_build)

    for name, fn, text in (("play", cmd_play, "swap mods in after the launcher patches; restore on exit"),
                           ("restore", cmd_restore, "put the original archives back now"),
                           ("status", cmd_status, "show what's installed and swapped")):
        s = sub.add_parser(name, help=text)
        s.add_argument("--mods", required=True, help="folder holding mod folders")
        if name == "play":
            s.add_argument("--once", action="store_true", help="exit after one game session")
            s.add_argument("--extended-archives", action="store_true",
                           help="patch the game client during play so archives can be up to 4 GiB (see README)")
        s.set_defaults(fn=fn)

    w = sub.add_parser("wad", help="list or extract a KIWAD archive")
    w.add_argument("action", choices=["ls", "extract"])
    w.add_argument("path")
    w.add_argument("pattern", nargs="?", default="*")
    w.add_argument("--out", default="extracted")
    w.set_defaults(fn=cmd_wad)
    return p


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return a.fn(a)
