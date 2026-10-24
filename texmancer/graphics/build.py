"""Rebuild a texture mod in place: after a game update (only the archives KI
changed) or for a different quality preset (every archive).

The mod folder's mod.json says which upscale projects feed it:
    "build": {"projects": ["C:/path/to/work/shared", ...], "preset": "full"}
Each project's project.json says how it was extracted. Progress is printed as
"STEP n/N text" and "PROGRESS done/total" lines so a front end can show it.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from .. import GAME_DATA
from ..swap import source_matches
from .upscale import Project, pack_projects

PRESETS = {
    "full": {"label": "Full 4×", "max_factor": 4, "max_source": None},
    "balanced": {"label": "Balanced 2×", "max_factor": 2, "max_source": None},
    "light": {"label": "Light", "max_factor": 2, "max_source": 256},
}


def read_meta(mod_dir: Path) -> dict:
    try:
        return json.loads((mod_dir / "mod.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_meta(mod_dir: Path, meta: dict) -> None:
    (mod_dir / "mod.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False), encoding="utf-8")


def stale_wads(mod_dir: Path, game_data: Path = GAME_DATA) -> list[str]:
    """Mod archives whose game original changed since the mod was built."""
    out = []
    for w in sorted(mod_dir.glob("*.wad")):
        live = game_data / w.name
        if live.exists() and source_matches(w, live) is False:
            out.append(w.name)
    return out


def build_mod(mod_dir: Path, preset: str | None = None, everything: bool = False, tools: str | None = None,
              game_data: Path = GAME_DATA, log=print) -> list[Path]:
    """Re-extract, re-upscale (only what changed) and repack. Returns the archives written."""
    meta = read_meta(mod_dir)
    cfg = meta.get("build") or {}
    projects = [Project(p) for p in cfg.get("projects", [])]
    if not projects:
        raise ValueError(f"{mod_dir.name} has no build recipe in mod.json (\"build\": {{\"projects\": [...]}})")
    preset = preset or cfg.get("preset", "full")
    if preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose {', '.join(PRESETS)}")
    changed_preset = preset != cfg.get("preset", "full")
    targets = None if (everything or changed_preset) else set(stale_wads(mod_dir, game_data))
    if targets is not None and not targets:
        log("STEP 1/1 Nothing to rebuild: every archive matches the installed game")
        return []

    steps = 2 * len(projects) + 2
    step = 0

    def say(text: str) -> None:
        nonlocal step
        step += 1
        log(f"STEP {step}/{steps} {text}")

    # 1. re-extract the archives in scope (changed textures drop their old outputs)
    for proj in projects:
        pc = proj.config()
        names = pc.get("wads", [])
        pool = sorted(game_data.glob("*.wad")) if "*" in names else [game_data / n for n in names]
        scope = [w for w in pool if w.exists() and (targets is None or w.name in targets)]
        say(f"Reading {len(scope)} archive(s) for {proj.root.name}")
        if scope and (targets is not None or everything):
            proj.extract(scope, pattern=pc.get("pattern", "*.dds"), min_size=pc.get("min_size", 64),
                         embedded=pc.get("embedded", False))
    # 2. upscale whatever is new or changed (everything else is already valid)
    for proj in projects:
        say(f"Upscaling new or changed textures in {proj.root.name}")
        proj.upscale(scale=4, tools=tools, backend="torch", log=log)
    # 3. pack into a staging folder next to the mod, then move into place
    say(f"Packing ({PRESETS[preset]['label']})")
    staging = mod_dir.parent.parent / "build" / mod_dir.name
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    old_index = mod_dir / "textures.json"
    if old_index.exists() and targets is not None:
        shutil.copy2(old_index, staging / "textures.json")
    for f in mod_dir.glob("Locale_*.wad"):  # overlays are rewritten with their package
        if targets is None or any(f.name.endswith("-" + t) for t in targets):
            (staging / f.name).unlink(missing_ok=True)
    written = pack_projects(projects, [game_data], staging, tools=tools, log=log, record_against=game_data,
                            wads=targets, max_factor=PRESETS[preset]["max_factor"],
                            max_source=PRESETS[preset]["max_source"],
                            extended=cfg.get("extended", False), locale=cfg.get("locale", "en-US"))
    say("Installing the rebuilt archives")
    if targets is None:  # whole-mod rebuild: archives no longer produced must go too
        keep = {p.name for p in written}
        for w in mod_dir.glob("*.wad"):
            if w.name not in keep:
                w.unlink()
                w.with_name(w.name + ".source.json").unlink(missing_ok=True)
    for f in staging.iterdir():
        dest = mod_dir / f.name
        dest.unlink(missing_ok=True)
        f.replace(dest)
    shutil.rmtree(staging, ignore_errors=True)
    from .upscale import write_overrides
    write_overrides(mod_dir, cfg.get("locale", "en-US"))  # list every overlay the mod now has
    try:
        index = json.loads((mod_dir / "textures.json").read_text(encoding="utf-8"))
        meta["textures"] = sum(len(v) for v in index.values())
    except (OSError, ValueError):
        pass
    meta.setdefault("build", {})["preset"] = preset
    write_meta(mod_dir, meta)
    log(f"DONE rebuilt {len(written)} archive(s)")
    return written
