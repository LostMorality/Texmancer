"""The setup program's worker: builds the HD mod from the player's own install.

    python -m texmancer.setup_pipeline <Texmancer folder>

Settings come from state.json in that folder (written by the setup window).
Stages, each resumable (stopping and reopening carries on where it left off):

1. quick: the interface (GUI, Root) and Wizard City, packed and installed so
   there's something to play with within minutes;
2. rest: every other archive with textures, in chunks of roughly 2 GB of game
   archives so the full-size upscales (deleted after each chunk) never pile up;
3. a final pack of everything into the mod.

Lines starting with "@@" are JSON events for the window; the rest is log text.
"""

from __future__ import annotations

import fnmatch
import json
import shutil
import sys
import time
import traceback
from pathlib import Path

from . import hardware
from .bootstrap import Layout, game_locale
from .graphics.build import PRESETS, write_meta
from .graphics.upscale import Project, pack_projects, write_overrides
from .swap import Swapper
from .wad import Wad

QUICK_WADS = ["GUI-WorldData.wad", "Root.wad", "WizardCity-*.wad"]
_TEXTURE_EXT = (".dds", ".nif", ".png", ".tga", ".jpg", ".jpeg", ".bmp", ".gif")
CHUNK_BYTES = 2 * 2**30
# Source pixels per second per backend at Balanced, used until a real rate is measured.
_GUESS_RATE = {"torch": 1.2e6, "ncnn": 0.8e6}
_POWER_SPEED = {"low": 0.4, "balanced": 1.0, "max": 1.15}


def emit(kind: str, **kw) -> None:
    print("@@" + json.dumps({"type": kind, **kw}), flush=True)


def log(text: str) -> None:
    print(text, flush=True)


class Pipeline:
    def __init__(self, base: Path):
        self.lay = Layout(base)
        self.state = self.lay.load_state()
        self.install = Path(self.state["install"])
        self.game_data = self.install / "Data" / "GameData"
        self.quality = int(self.state.get("quality", 2))
        self.preset = "full" if self.quality == 4 else "balanced"
        self.backend = self.state.get("backend", "torch")
        gpus = hardware.detect()
        self.gpu = gpus[0] if gpus else None
        self.power_level = self.state.get("power", "balanced")
        self.power = hardware.power_settings(self.power_level, self.gpu)
        if self.power.get("priority") == "below_normal":
            hardware.set_low_priority()
        self.locale = self.state.get("locale") or game_locale(self.install)
        self.tools = str(self.lay.tools)
        # Measured AI speed, source pixels per second (kept across runs for the estimate).
        self.rate = float(self.state.get("rate") or _GUESS_RATE.get(self.backend, 1e6)
                          * _POWER_SPEED.get(self.power_level, 1.0))
        self.swapper = Swapper(self.lay.mods, self.game_data, install=self.install, log=log)

    # -- bookkeeping ----------------------------------------------------------

    def done(self, step: str) -> bool:
        return step in self.state.get("done", [])

    def mark(self, step: str) -> None:
        self.state.setdefault("done", []).append(step)
        self.lay.save_state(self.state)

    def wait_for_originals(self, why: str, wads: list[str] | None = None) -> None:
        """Reading or replacing archives while the mod is swapped in would read the
        mod instead of the game (or yank files the game has open). With ``wads``,
        only waits if one of those is swapped in, so the rest of the game can be
        read while someone plays with the quick part."""
        def blocked() -> bool:
            if not self.swapper.is_swapped():
                return False
            if wads is None:
                return True
            live = {e.removeprefix("+").split("/", 1)[-1].lower() for e in self.swapper.swapped_entries()}
            return bool(live & {w.lower() for w in wads})
        announced = False
        while blocked():
            if not announced:
                emit("status", text=f"Waiting for Wizard101 to close before {why}")
                announced = True
            time.sleep(10)

    def eta_seconds(self, pixels_left: int) -> int:
        # Decoding, encoding and packing add roughly a quarter on top of the AI step.
        return int(pixels_left / max(self.rate, 1.0) * 1.25)

    # -- stages ---------------------------------------------------------------

    def texture_wads(self) -> list[Path]:
        """Every archive holding at least one image or model (skips sound, music and
        the mod's own Locale_ overlays)."""
        out = []
        for w in sorted(self.game_data.glob("*.wad")):
            if w.name.lower().startswith("locale_"):
                continue
            try:
                with Wad(w) as wad:
                    if any(e.name.lower().endswith(_TEXTURE_EXT) for e in wad.entries):
                        out.append(w)
            except Exception as exc:  # an archive mid-download or damaged: leave it out
                log(f"skipping {w.name}: {exc}")
        return out

    def plan(self) -> list[dict]:
        """Projects to build, in order: [{"name", "label", "wads"}]."""
        if "plan" in self.state:
            return self.state["plan"]
        emit("status", text="Looking through the game's archives")
        wads = self.texture_wads()
        quick = [w for w in wads if any(fnmatch.fnmatch(w.name.lower(), p.lower()) for p in QUICK_WADS)]
        rest = [w for w in wads if w not in quick]
        plan = [{"name": "quick", "label": "the interface and Wizard City", "wads": [w.name for w in quick]}]
        chunk: list[Path] = []
        size = 0
        for w in rest:
            chunk.append(w)
            size += w.stat().st_size
            if size >= CHUNK_BYTES:
                plan.append({"name": f"rest-{len(plan):02d}", "label": "", "wads": [p.name for p in chunk]})
                chunk, size = [], 0
        if chunk:
            plan.append({"name": f"rest-{len(plan):02d}", "label": "", "wads": [p.name for p in chunk]})
        n = len(plan) - 1
        for i, p in enumerate(plan[1:], 1):
            p["label"] = f"the rest of the game, part {i} of {n}"
        self.state["plan"] = plan
        self.lay.save_state(self.state)
        return plan

    def project(self, p: dict) -> Project:
        return Project(self.lay.work / p["name"])

    def extract(self, p: dict) -> None:
        step = f"{p['name']}:extract"
        if self.done(step):
            return
        self.wait_for_originals("reading the game's archives", p["wads"])
        emit("status", text=f"Copying textures out of {p['label']}")
        wads = [self.game_data / n for n in p["wads"] if (self.game_data / n).exists()]
        proj = self.project(p)
        total = 0
        for i, w in enumerate(wads):
            # One archive at a time so progress moves; extract() merges into the manifest.
            total = proj.extract([w], pattern="*", min_size=1, embedded=True, max_factor=self.quality)
            emit("progress", stage=f"Copying textures out of {p['label']}", done=i + 1, total=len(wads))
        log(f"{p['name']}: {total} textures")
        self.mark(step)

    def pixels(self, p: dict) -> int:
        return sum(t.width * t.height for t in self.project(p).load() if t.kind != "nifraw")

    def upscale(self, p: dict, later_pixels: int) -> None:
        step = f"{p['name']}:upscale"
        if self.done(step):
            return
        proj = self.project(p)
        label = f"Upscaling {p['label']}"
        emit("status", text=label)
        first: list[tuple[int, float]] = []  # (pixels, time) at the first AI batch

        def progress(done: int, total: int) -> None:
            # Timed from the first batch: decoding and checking come before it.
            if not first:
                first.append((done, time.monotonic()))
            px0, t0 = first[0]
            elapsed = time.monotonic() - t0
            # Images go small to large, and small ones are slower per pixel, so the
            # measurement only replaces the guess once a fair share is through.
            if elapsed > 30 and done - px0 > total * 0.02:
                self.rate = (done - px0) / elapsed
            emit("progress", stage=label, done=done, total=total,
                 eta=self.eta_seconds(total - done + later_pixels))

        proj.upscale(scale=4, tools=self.tools, backend=self.backend, power=self.power, log=log,
                     torch_model=self.lay.model, progress=progress)
        # The full-size PNGs are only an input to the encoder; a finished DDS
        # counts as done without them, so free the space now.
        shutil.rmtree(proj.up, ignore_errors=True)
        self.state["rate"] = self.rate
        self.mark(step)

    def pack(self, plan: list[dict], step: str, label: str) -> None:
        if self.done(step):
            return
        projects = [self.project(p) for p in plan if self.done(f"{p['name']}:upscale")]
        # Packing reads the original archives and then replaces the mod's.
        self.wait_for_originals("packing the new textures")
        emit("status", text=f"Packing {label}")
        emit("progress", stage=f"Packing {label}", done=0, total=0)
        staging = self.lay.staging
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        written = pack_projects(projects, [self.game_data], staging, tools=self.tools, log=log,
                                record_against=self.game_data, max_factor=PRESETS[self.preset]["max_factor"],
                                max_source=PRESETS[self.preset]["max_source"], locale=self.locale)
        self.wait_for_originals("installing the new textures")
        mod = self.lay.mod
        mod.mkdir(parents=True, exist_ok=True)
        keep = {f.name for f in staging.iterdir()}
        for f in list(mod.glob("*.wad")) + list(mod.glob("*.source.json")):
            if f.name not in keep:  # e.g. an overlay the new pack no longer needs
                f.unlink()
        for f in staging.iterdir():
            dest = mod / f.name
            dest.unlink(missing_ok=True)
            f.replace(dest)
        shutil.rmtree(staging.parent, ignore_errors=True)
        write_overrides(mod, self.locale)
        try:
            count = sum(len(v) for v in json.loads((mod / "textures.json").read_text(encoding="utf-8")).values())
        except (OSError, ValueError):
            count = 0
        write_meta(mod, {"label": f"HD textures ({self.quality}x)", "textures": count,
                         "build": {"projects": [str(p.root) for p in projects], "preset": self.preset,
                                   "locale": self.locale}})
        log(f"installed {len(written)} archives, {count:,} textures, into {mod}")
        self.mark(step)

    def run(self) -> None:
        plan = self.plan()
        quick, rest = plan[0], plan[1:]
        self.extract(quick)
        rest_guess = 0 if self.state.get("quick_only") else self.state.get("rest_pixels", 0)
        self.upscale(quick, rest_guess)
        self.pack(plan[:1], "quick:pack", "the interface and Wizard City")
        emit("quick_ready")
        if self.state.get("quick_only"):
            emit("done", quick_only=True)
            return
        for p in rest:
            self.extract(p)
        pending = [p for p in rest if not self.done(f"{p['name']}:upscale")]
        sizes = {p["name"]: self.pixels(p) for p in pending}
        self.state["rest_pixels"] = sum(sizes.values())
        self.lay.save_state(self.state)
        for i, p in enumerate(pending):
            self.upscale(p, sum(sizes[q["name"]] for q in pending[i + 1:]))
        self.pack(plan, "all:pack", "the whole game")
        emit("done", quick_only=False)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print(__doc__)
        return 2
    try:
        Pipeline(Path(argv[0])).run()
    except Exception as exc:  # reported to the window, with the details in the log
        traceback.print_exc()
        emit("error", text=str(exc) or type(exc).__name__)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
