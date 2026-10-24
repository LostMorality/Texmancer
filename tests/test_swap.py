import json
import os
from pathlib import Path

import pytest

from texmancer.swap import LogTail, Swapper, record_source
from texmancer.wad import Wad, write_wad


def payload(p):
    with Wad(p) as w:
        return w.read("a")


@pytest.fixture
def layout(tmp_path: Path):
    """GameData with Root.wad + Shared.wad; one mod 'HD' replacing both."""
    game, mods = tmp_path / "GameData", tmp_path / "mods"
    hd = mods / "HD"
    game.mkdir()
    hd.mkdir(parents=True)
    for name in ("Root.wad", "Shared.wad"):
        write_wad(game / name, [("a", b"original" * 10)])
        os.utime(game / name, (1_700_000_000, 1_700_000_000))
        write_wad(hd / name, [("a", b"modded" * 10)])
    return game, mods


def snapshot(game):
    return {p.name: (p.read_bytes(), p.stat().st_mtime_ns) for p in game.glob("*.wad")}


def test_mod_swaps_all_its_wads_and_restores_exactly(layout):
    game, mods = layout
    before = snapshot(game)
    s = Swapper(mods, game, log=lambda m: None, game_running=lambda: False)
    s.validate()
    s.swap_in()
    assert s.is_swapped()
    assert payload(game / "Root.wad") == payload(game / "Shared.wad") == b"modded" * 10
    s.swap_in()  # idempotent
    s.restore()
    assert not s.is_swapped() and snapshot(game) == before
    assert payload(mods / "HD" / "Root.wad") == b"modded" * 10  # mod files back in their folder


def test_one_stale_wad_skips_the_whole_mod(layout):
    game, mods = layout
    for name in ("Root.wad", "Shared.wad"):
        record_source(mods / "HD" / name, game / name)
    write_wad(game / "Shared.wad", [("a", b"patched by KI" * 10)])  # game update to one archive
    s = Swapper(mods, game, log=lambda m: None, game_running=lambda: False)
    assert s.is_stale(mods / "HD")
    s.swap_in()
    assert not s.is_swapped()
    assert payload(game / "Root.wad") == b"original" * 10  # untouched, no half-mod


def test_only_selected_mods_swap(layout):
    game, mods = layout
    s = Swapper(mods, game, log=lambda m: None, game_running=lambda: False, only=set())
    s.swap_in()
    assert not s.is_swapped() and payload(game / "Root.wad") == b"original" * 10


def test_same_wad_in_two_enabled_mods_is_refused(layout):
    game, mods = layout
    (mods / "Other").mkdir()
    write_wad(mods / "Other" / "Root.wad", [("a", b"x")])
    with pytest.raises(ValueError):
        Swapper(mods, game, log=lambda m: None, game_running=lambda: False).swap_in()
    assert payload(game / "Root.wad") == b"original" * 10


def test_restores_old_flat_state(tmp_path: Path):
    """State written by the pre-folder version ("Root.wad") must still restore."""
    game, mods = tmp_path / "GameData", tmp_path / "mods"
    (mods / "originals").mkdir(parents=True)
    game.mkdir()
    write_wad(mods / "originals" / "Root.wad", [("a", b"original" * 10)])
    write_wad(game / "Root.wad", [("a", b"modded" * 10)])
    (mods / "swap_state.json").write_text(json.dumps({"swapped": ["Root.wad"]}))
    Swapper(mods, game, log=lambda m: None, game_running=lambda: False).restore()
    assert payload(game / "Root.wad") == b"original" * 10
    assert payload(mods / "Root.wad") == b"modded" * 10


def test_validate_rejects_unknown_wad(layout):
    game, mods = layout
    write_wad(mods / "HD" / "Nope.wad", [("x", b"1")])
    with pytest.raises(FileNotFoundError):
        Swapper(mods, game, log=lambda m: None, game_running=lambda: False).validate()


def test_logtail_handles_recreated_log(tmp_path: Path):
    log = tmp_path / "l.txt"
    log.write_text("old line\n" * 50)
    t = LogTail(log)
    assert t.new_lines() == []
    log.write_text("fresh\nGame patched successful\npartial")
    assert t.new_lines() == ["fresh", "Game patched successful"]
    with open(log, "a") as f:
        f.write(" line\n")
    assert t.new_lines() == ["partial line"]


def test_added_overlay_archive_goes_in_and_comes_back_out(layout):
    game, mods = layout
    write_wad(mods / "HD" / "Locale_en-US-Shared.wad", [("a", b"overflow" * 10)])
    before = snapshot(game)
    s = Swapper(mods, game, log=lambda m: None, game_running=lambda: False)
    s.validate()  # an overlay has no counterpart in the game, and that's allowed
    assert not s.is_stale(mods / "HD")
    s.swap_in()
    assert payload(game / "Locale_en-US-Shared.wad") == b"overflow" * 10
    s.restore()
    assert snapshot(game) == before  # the overlay is gone, originals untouched
    assert (mods / "HD" / "Locale_en-US-Shared.wad").exists()


def test_skip_leaves_chosen_archives_out(layout):
    game, mods = layout
    s = Swapper(mods, game, log=lambda m: None, game_running=lambda: False,
                skip=lambda mod, wad: wad == "Shared.wad")
    s.swap_in()
    assert payload(game / "Root.wad") == b"modded" * 10
    assert payload(game / "Shared.wad") == b"original" * 10
    s.restore()


def _fake_client() -> bytes:
    read = b"\x8b\x51\x48\x85\xd2\x0f\x88\x42\x01\x00\x00"
    save = b"\x8b\x49\x48\x85\xc9\x0f\x88\xd9\x01\x00\x00"
    return b"\x90" * 64 + read + b"\x90" * 32 + save + b"\x90" * 32 + read + b"\x90" * 64


def test_client_patch_lifts_all_three_checks_and_refuses_unknown_builds():
    from texmancer.swap import patch_client_bytes
    exe = _fake_client()
    out = patch_client_bytes(exe)
    assert out is not None and len(out) == len(exe)
    assert b"\x0f\x88" not in out and out.count(b"\x66\x0f\x1f\x44\x00\x00") == 3
    assert patch_client_bytes(exe + b"\x8b\x51\x48\x85\xd2\x0f\x88\x00\x00\x00\x00") is None  # extra site
    assert patch_client_bytes(b"\x90" * 100) is None


def test_patched_client_is_restored_with_the_archives(layout, tmp_path):
    game, mods = layout
    install = tmp_path / "install"
    (install / "Bin").mkdir(parents=True)
    exe = install / "Bin" / "WizardGraphicalClient.exe"
    exe.write_bytes(_fake_client())
    s = Swapper(mods, game, log=lambda m: None, game_running=lambda: False, install=install,
                backup_path=tmp_path / "backups", patch_client=True)
    s._patch_client()
    assert s.client_patched and b"\x0f\x88" not in exe.read_bytes()
    s.swap_in()
    s.restore()
    assert exe.read_bytes() == _fake_client() and not s.client_patched


def test_overlay_cache_vouches_for_live_base(tmp_path):
    from texmancer.swap import overlay_cache
    base = tmp_path / "Equipment-WorldData.wad"
    base.write_bytes(b"x" * 1234)
    cache = bytes(range(24))
    out = overlay_cache(cache, base)
    assert out[:4] == cache[:4] and out[8:16] == cache[8:16]
    assert int.from_bytes(out[4:8], "little") == 1234
    assert int.from_bytes(out[16:], "little") == base.stat().st_mtime_ns // 1_000_000
    assert overlay_cache(b"short", base) is None


def test_restore_drops_an_archive_the_game_replaced(tmp_path):
    game, mods = tmp_path / "GameData", tmp_path / "Mods"
    game.mkdir(); (mods / "HD").mkdir(parents=True)
    (game / "Root.wad").write_bytes(b"original")
    (mods / "HD" / "Root.wad").write_bytes(b"modded archive, much bigger")
    s = Swapper(mods, game, log=lambda m: None, game_running=lambda: False)
    s.install = tmp_path
    s.swap_in()
    (game / "Root.wad").write_bytes(b"redownload")  # the client overwrote it mid-session
    s.restore()
    assert (game / "Root.wad").read_bytes() == b"original"
    assert not (mods / "HD" / "Root.wad").exists()
    assert s.replaced == ["HD/Root.wad"]
