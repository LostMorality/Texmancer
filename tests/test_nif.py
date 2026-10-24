import pytest

from texmancer import GAME_DATA
from texmancer.nif import NifError, TextureNif
from texmancer.wad import Wad


def _texture_nifs(limit=300):
    out = []
    for wad_name in ("WizardCity-WC_Hub.wad", "_Shared-WorldData.wad"):
        p = GAME_DATA / wad_name
        if not p.exists():
            continue
        with Wad(p) as w:
            for e in w.entries:
                n = e.name.lower()
                if n.startswith("textures/") and n.endswith(".nif") and w.is_downloaded(e):
                    out.append((e.name, w.read(e)))
                    if len(out) >= limit:
                        return out
    return out


needs_game = pytest.mark.skipif(not (GAME_DATA / "_Shared-WorldData.wad").exists(), reason="game not installed")


@needs_game
def test_roundtrip_is_byte_identical():
    nifs = _texture_nifs()
    assert nifs
    for name, d in nifs:
        assert TextureNif.parse(d).serialize() == d, name


@needs_game
def test_dxt_via_dds_is_lossless():
    for name, d in _texture_nifs():
        t = TextureNif.parse(d)
        if t.pixels.pixel_format in (4, 5, 6):
            assert t.with_dds(t.pixels.to_dds()).serialize() == d, name


@needs_game
def test_replacing_pixels_resizes_block_and_mips():
    name, d = next((n, d) for n, d in _texture_nifs() if TextureNif.parse(d).pixels.pixel_format == 4)
    t = TextureNif.parse(d)
    w, h = t.pixels.width * 2, t.pixels.height * 2
    # Fake a 2x DXT1 DDS with a full mip chain.
    import struct
    mips, size, ww, hh = 0, 0, w, h
    while True:
        size += max(1, (ww + 3) // 4) * max(1, (hh + 3) // 4) * 8
        mips += 1
        if ww == 1 and hh == 1:
            break
        ww, hh = max(1, ww // 2), max(1, hh // 2)
    head = bytearray(t.pixels.to_dds()[:128])
    struct.pack_into("<II", head, 12, h, w)
    struct.pack_into("<I", head, 28, mips)
    new = TextureNif.parse(t.with_dds(bytes(head) + b"\x11" * size).serialize())
    assert (new.pixels.width, new.pixels.height, len(new.pixels.mips)) == (w, h, mips)
    assert new.header.block_sizes[0] == len(new.pixels.serialize())


def test_rejects_non_nif():
    with pytest.raises(NifError):
        TextureNif.parse(b"not a nif at all" * 4)
