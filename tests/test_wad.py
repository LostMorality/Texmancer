import os
from pathlib import Path

import pytest

from texmancer import GAME_DATA
from texmancer.wad import Wad, WadError, repack, write_wad


def test_roundtrip(tmp_path: Path):
    files = [("a.txt", b"hello" * 100), ("dir/b.bin", os.urandom(64)), ("empty", b"")]
    dest = tmp_path / "t.wad"
    write_wad(dest, files)
    with Wad(dest) as w:
        assert [e.name for e in w.entries] == [n for n, _ in files]
        for n, data in files:
            assert w.read(n) == data
        assert w.get_entry("a.txt").is_zip          # compressible
        assert not w.get_entry("dir/b.bin").is_zip  # random bytes stored raw


def test_repack_replaces_and_preserves(tmp_path: Path):
    src = tmp_path / "s.wad"
    write_wad(src, [("x", b"1" * 50), ("y", b"2" * 50)])
    out = tmp_path / "o.wad"
    repack(src, out, {"Y": b"new", "z": b"added"})
    with Wad(out) as w:
        assert w.read("x") == b"1" * 50 and w.read("y") == b"new" and w.read("z") == b"added"


def test_path_traversal_refused(tmp_path: Path):
    bad = tmp_path / "bad.wad"
    write_wad(bad, [("../evil.txt", b"x")])
    with Wad(bad) as w, pytest.raises(WadError):
        w.extract(tmp_path / "out")


def test_not_a_wad(tmp_path: Path):
    p = tmp_path / "junk"
    p.write_bytes(b"nope" * 10)
    with pytest.raises(WadError):
        Wad(p)


@pytest.mark.skipif(not (GAME_DATA / "Root.wad").exists(), reason="game not installed")
def test_live_root_wad():
    with Wad(GAME_DATA / "Root.wad") as w:
        assert w.version == 1 and len(w) > 1000
        data = w.read(w.entries[0])
        assert len(data) == w.entries[0].size


def test_new_entries_get_ki_crc(tmp_path: Path):
    from texmancer.wad import ki_crc32
    assert ki_crc32(b"123456789") == 771566984  # KI variant check vector (wizpatch)
    dest = tmp_path / "c.wad"
    write_wad(dest, [("a.txt", b"hello" * 100), ("r", os.urandom(50))])
    with Wad(dest) as w:
        for e in w.entries:
            assert e.crc == ki_crc32(w.read_raw(e))


@pytest.mark.skipif(not (GAME_DATA / "Root.wad").exists(), reason="game not installed")
def test_live_entry_crcs_match_ki_crc():
    from texmancer.wad import ki_crc32
    with Wad(GAME_DATA / "Root.wad") as w:
        checked = [e for e in w.entries[:200] if w.is_downloaded(e)]
        assert checked and all(ki_crc32(w.read_raw(e)) == e.crc for e in checked)


def test_refuses_to_pass_the_archive_limit(tmp_path: Path):
    with pytest.raises(WadError, match="archive limit"):
        write_wad(tmp_path / "big.wad", [("a", os.urandom(200)), ("b", os.urandom(200))], max_bytes=300)
    assert not (tmp_path / "big.wad").exists() and not (tmp_path / "big.wad.tmp").exists()
    # the same data fits under a larger (extended) limit
    write_wad(tmp_path / "big.wad", [("a", os.urandom(200)), ("b", os.urandom(200))], max_bytes=1000)
    assert (tmp_path / "big.wad").exists()
