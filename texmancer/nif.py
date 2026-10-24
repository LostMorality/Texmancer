"""Gamebryo 20.6.0.0 / 20.3.0.9 texture NIFs: one NiPixelData block holding a whole mip chain.

Wizard101 stores most world, monster and treasure-card textures this way
(``Textures/<name>__<hash>.nif`` inside zone/mob WADs) rather than as .dds.
Layout per NifTools' nif.xml for v20.6.0.0, user version 0:

    header   "Gamebryo File Format, Version 20.6.0.0\\n" | u32 version | u8 endian | u32 user_version
             | u32 num_blocks | u16 num_block_types | SizedString[types] | u16[block type idx]
             | u32[block sizes] | u32 num_strings | u32 max_strlen | SizedString[strings] | u32 num_groups
             | u32[groups]
    block    NiPixelData (see PixelData)
    footer   u32 num_roots | i32[roots]

The pixel data's mips are contiguous top-down, same as DDS, so conversion is
lossless in both directions.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field, replace

# Versions W101 ships with block sizes in the header (20.2.0.5+), so blocks can
# be split without understanding every type. The 20.1.0.3 files are test
# proxies with no usable textures.
VERSIONS = {b"Gamebryo File Format, Version 20.6.0.0\n": 0x14060000,
            b"Gamebryo File Format, Version 20.3.0.9\n": 0x14030009,
            b"Gamebryo File Format, Version 20.2.0.8\n": 0x14020008,
            b"Gamebryo File Format, Version 20.2.0.7\n": 0x14020007}
# Blocks holding a texture's pixels. The persistent variant (used for textures
# baked into models) adds a padded pixel count and a platform id.
PIXEL_BLOCKS = (b"NiPixelData", b"NiPersistentSrcTextureRendererData")
_SRGB_SINCE = 0x14030004

# PixelFormat enum (nif.xml) -> DDS FourCC / bit layout
FMT_RGB, FMT_RGBA, FMT_DXT1, FMT_DXT3, FMT_DXT5 = 0, 1, 4, 5, 6
_FOURCC = {FMT_DXT1: b"DXT1", FMT_DXT3: b"DXT3", FMT_DXT5: b"DXT5"}
_FMT_FROM_FOURCC = {v: k for k, v in _FOURCC.items()}
_BLOCK_BYTES = {FMT_DXT1: 8, FMT_DXT3: 16, FMT_DXT5: 16}


class NifError(Exception):
    pass


@dataclass
class Header:
    header_string: bytes
    version: int
    endian: int
    user_version: int
    block_types: list[bytes]
    block_type_index: list[int]
    block_sizes: list[int]
    strings: list[bytes]
    max_string_length: int
    groups: list[int]

    def serialize(self) -> bytes:
        out = bytearray(self.header_string)
        out += struct.pack("<IBII", self.version, self.endian, self.user_version, len(self.block_type_index))
        out += struct.pack("<H", len(self.block_types))
        for t in self.block_types:
            out += struct.pack("<I", len(t)) + t
        out += struct.pack(f"<{len(self.block_type_index)}H", *self.block_type_index)
        out += struct.pack(f"<{len(self.block_sizes)}I", *self.block_sizes)
        out += struct.pack("<II", len(self.strings), self.max_string_length)
        for s in self.strings:
            out += struct.pack("<I", len(s)) + s
        out += struct.pack("<I", len(self.groups)) + struct.pack(f"<{len(self.groups)}I", *self.groups)
        return bytes(out)


@dataclass
class Channel:
    type: int
    convention: int
    bits: int
    signed: int


@dataclass
class Mip:
    width: int
    height: int
    offset: int


@dataclass
class PixelData:
    pixel_format: int
    bits_per_pixel: int
    renderer_hint: int
    extra_data: int
    flags: int
    tiling: int
    srgb: int
    channels: list[Channel]
    palette_ref: int
    bytes_per_pixel: int
    mips: list[Mip]
    num_faces: int
    data: bytes
    version: int = 0x14060000
    persistent: bool = False
    platform: int = 0

    _HEAD = struct.Struct("<IBIIBIB")
    _HEAD_NO_SRGB = struct.Struct("<IBIIBI")
    _CHAN = struct.Struct("<IIBB")

    @property
    def width(self) -> int:
        return self.mips[0].width

    @property
    def height(self) -> int:
        return self.mips[0].height

    @classmethod
    def parse(cls, buf: bytes, version: int = 0x14060000, persistent: bool = False) -> "PixelData":
        o = 0
        if version >= _SRGB_SINCE:
            fmt, bpp, hint, extra, flags, tiling, srgb = cls._HEAD.unpack_from(buf, o)
            o += cls._HEAD.size
        else:
            (fmt, bpp, hint, extra, flags, tiling), srgb = cls._HEAD_NO_SRGB.unpack_from(buf, o), 0
            o += cls._HEAD_NO_SRGB.size
        chans = []
        for _ in range(4):
            chans.append(Channel(*cls._CHAN.unpack_from(buf, o)))
            o += cls._CHAN.size
        palette, nmips, bytes_pp = struct.unpack_from("<iII", buf, o)
        o += 12
        if nmips > 32:
            raise NifError(f"implausible mip count {nmips}")
        mips = []
        for _ in range(nmips):
            mips.append(Mip(*struct.unpack_from("<III", buf, o)))
            o += 12
        platform = 0
        if persistent:
            num_pixels, _pad, num_faces, platform = struct.unpack_from("<IIII", buf, o)
            o += 16
        else:
            num_pixels, num_faces = struct.unpack_from("<II", buf, o)
            o += 8
        total = num_pixels * num_faces
        if o + total != len(buf):
            raise NifError(f"pixel data size {total} doesn't fill the block ({len(buf) - o} left)")
        return cls(fmt, bpp, hint, extra, flags, tiling, srgb, chans, palette, bytes_pp, mips,
                   num_faces, buf[o:o + total], version, persistent, platform)

    def serialize(self) -> bytes:
        if self.version >= _SRGB_SINCE:
            out = bytearray(self._HEAD.pack(self.pixel_format, self.bits_per_pixel, self.renderer_hint,
                                            self.extra_data, self.flags, self.tiling, self.srgb))
        else:
            out = bytearray(self._HEAD_NO_SRGB.pack(self.pixel_format, self.bits_per_pixel, self.renderer_hint,
                                                    self.extra_data, self.flags, self.tiling))
        for c in self.channels:
            out += self._CHAN.pack(c.type, c.convention, c.bits, c.signed)
        out += struct.pack("<iII", self.palette_ref, len(self.mips), self.bytes_per_pixel)
        for m in self.mips:
            out += struct.pack("<III", m.width, m.height, m.offset)
        if len(self.data) % self.num_faces:
            raise NifError("pixel data doesn't divide evenly into faces")
        per_face = len(self.data) // self.num_faces
        if self.persistent:
            out += struct.pack("<IIII", per_face, per_face, self.num_faces, self.platform)
        else:
            out += struct.pack("<II", per_face, self.num_faces)
        return bytes(out) + self.data

    @property
    def is_color(self) -> bool:
        """Plain colour art worth upscaling: DXT, or uncompressed R/G/B(/A) with
        padding. Excludes bump/normal maps (U/V offset, luma, vector channels)
        and cube maps, which an image model would corrupt."""
        if self.num_faces != 1:
            return False
        if self.pixel_format in _FOURCC:
            return True
        return self.pixel_format in (FMT_RGB, FMT_RGBA) and all(c.type in (0, 1, 2, 3, 14, 19) for c in self.channels)

    @property
    def has_alpha(self) -> bool:
        if self.pixel_format in (FMT_DXT3, FMT_DXT5):
            return True
        return any(c.type == 3 and c.bits for c in self.channels)

    # -- DDS conversion -----------------------------------------------------

    def to_dds(self) -> bytes:
        """Single-face DXT1/3/5 or uncompressed RGB(A) -> DDS bytes (all mips)."""
        if self.num_faces != 1:
            raise NifError("cube maps aren't supported")
        DDSD = 0x1 | 0x2 | 0x4 | 0x1000 | 0x20000  # caps|height|width|pixelformat|mipmapcount
        caps = 0x1000 | (0x400008 if len(self.mips) > 1 else 0)
        if self.pixel_format in _FOURCC:
            pf = struct.pack("<II4s5I", 32, 0x4, _FOURCC[self.pixel_format], 0, 0, 0, 0, 0)
            linear = max(1, (self.width + 3) // 4) * max(1, (self.height + 3) // 4) * _BLOCK_BYTES[self.pixel_format]
            DDSD |= 0x80000
        elif self.pixel_format in (FMT_RGB, FMT_RGBA):
            # W101 ships uncompressed textures in layouts texconv rejects (24-bit
            # R-G-B byte order, assorted 16-bit). Normalise to plain 32-bit BGRA.
            alpha = self.has_alpha
            pf = struct.pack("<II4s5I", 32, 0x40 | (0x1 if alpha else 0), b"\0\0\0\0", 32,
                             0x00FF0000, 0x0000FF00, 0x000000FF, 0xFF000000 if alpha else 0)
            linear = self.width * 4
            DDSD |= 0x8
            head = struct.pack("<4sIIIIIII44s", b"DDS ", 124, DDSD, self.height, self.width, linear, 0,
                               len(self.mips), b"\0" * 44)
            return head + pf + struct.pack("<IIIII", caps, 0, 0, 0, 0) + self._to_bgra8()
        else:
            raise NifError(f"unsupported pixel format {self.pixel_format}")
        head = struct.pack("<4sIIIIIII44s", b"DDS ", 124, DDSD, self.height, self.width, linear, 0,
                           len(self.mips), b"\0" * 44)
        head += pf + struct.pack("<IIIII", caps, 0, 0, 0, 0)
        return head + self.data

    def _to_bgra8(self) -> bytes:
        """Every mip of an uncompressed texture as 8-bit BGRA, unpacked by channel masks."""
        import numpy as np

        bits = self.bits_per_pixel or self.bytes_per_pixel * 8
        if bits not in (16, 24, 32):
            raise NifError(f"unsupported uncompressed depth {bits}")
        bpp = bits // 8
        masks = _channel_masks(self.channels)  # R, G, B, A
        out = []
        for m in self.mips:
            n = m.width * m.height
            if m.offset + n * bpp > len(self.data):
                raise NifError("mip runs past the pixel data")
            raw = np.frombuffer(self.data, dtype=np.uint8, count=n * bpp, offset=m.offset).reshape(n, bpp)
            v = np.zeros(n, dtype=np.uint32)
            for k in range(bpp):
                v |= raw[:, k].astype(np.uint32) << np.uint32(8 * k)
            chans = []
            for idx, mask in enumerate(masks):
                if not mask:
                    chans.append(np.full(n, 255 if idx == 3 else 0, dtype=np.uint8))
                    continue
                shift = (mask & -mask).bit_length() - 1
                top = (1 << bin(mask).count("1")) - 1
                val = (v & np.uint32(mask)) >> np.uint32(shift)
                chans.append(((val.astype(np.uint64) * 255 + top // 2) // top).astype(np.uint8))
            r, g, b, a = chans
            out.append(np.stack([b, g, r, a], axis=1).tobytes())
        return b"".join(out)

    def with_dds(self, dds: bytes) -> "PixelData":
        """A copy of this block holding ``dds`` (same pixel format family)."""
        if dds[:4] != b"DDS ":
            raise NifError("not a DDS file")
        height, width = struct.unpack_from("<II", dds, 12)
        nmips = struct.unpack_from("<I", dds, 28)[0] or 1
        pf_flags = struct.unpack_from("<I", dds, 80)[0]
        if not pf_flags & 0x4:
            raise NifError("only DXT DDS can be written back (got uncompressed)")
        fmt = _FMT_FROM_FOURCC.get(dds[84:88])
        if fmt is None:
            raise NifError(f"unsupported FourCC {dds[84:88]!r}")
        block = _BLOCK_BYTES[fmt]
        mips, off, w, h = [], 0, width, height
        for _ in range(nmips):
            mips.append(Mip(w, h, off))
            off += max(1, (w + 3) // 4) * max(1, (h + 3) // 4) * block
            w, h = max(1, w // 2), max(1, h // 2)
        data = dds[128:128 + off]
        if len(data) != off:
            raise NifError(f"DDS holds {len(data)} bytes of pixels, mips need {off}")
        chans = self.channels if self.pixel_format in _FOURCC else [Channel(4, 4, 0, 0)] + [Channel(19, 5, 0, 0)] * 3
        return replace(self, pixel_format=fmt, bits_per_pixel=0, bytes_per_pixel=0, channels=chans,
                       mips=mips, num_faces=1, data=data)


    # -- cube maps ----------------------------------------------------------

    def face(self, k: int) -> "PixelData":
        """One face of a cube map as a single-face texture (faces are stored
        one after another, each with its full mip chain)."""
        per = len(self.data) // self.num_faces
        return replace(self, num_faces=1, data=self.data[k * per:(k + 1) * per])

    def with_faces(self, faces: list["PixelData"]) -> "PixelData":
        """This cube map rebuilt from per-face textures that share one layout."""
        first = faces[0]
        if any((f.pixel_format, f.width, f.height, len(f.mips), len(f.data)) !=
               (first.pixel_format, first.width, first.height, len(first.mips), len(first.data)) for f in faces):
            raise NifError("cube faces differ in size or format")
        return replace(first, num_faces=len(faces), data=b"".join(f.data for f in faces),
                       version=self.version, persistent=self.persistent, platform=self.platform)

    # -- non-colour data (bump/luma maps) -----------------------------------

    @property
    def is_raw_resizable(self) -> bool:
        """Uncompressed single-face data that isn't plain colour (signed bump
        offsets, luma). Not for an image model, but resampled exactly."""
        return (self.num_faces == 1 and self.pixel_format not in _FOURCC and not self.is_color
                and (self.bits_per_pixel or self.bytes_per_pixel * 8) in (8, 16, 24, 32)
                and sum(c.bits for c in self.channels) == (self.bits_per_pixel or self.bytes_per_pixel * 8))

    def resampled(self, factor: int) -> "PixelData":
        """A ``factor``x copy of a raw texture: each channel Lanczos-resampled in
        its own numeric range (signed channels as signed) and block-corrected so it
        shrinks back to the source exactly, full mip chain
        rebuilt by 2x2 averaging, same pixel format."""
        import numpy as np
        from PIL import Image

        bits = self.bits_per_pixel or self.bytes_per_pixel * 8
        bpp = bits // 8
        m0 = self.mips[0]
        n = m0.width * m0.height
        raw = np.frombuffer(self.data, dtype=np.uint8, count=n * bpp, offset=m0.offset).reshape(n, bpp)
        v = np.zeros(n, dtype=np.uint64)
        for k in range(bpp):
            v |= raw[:, k].astype(np.uint64) << np.uint64(8 * k)
        chans, shift = [], 0
        for c in self.channels:
            if c.bits:
                x = ((v >> np.uint64(shift)) & np.uint64((1 << c.bits) - 1)).astype(np.int64)
                if c.signed:
                    x = np.where(x >= 1 << (c.bits - 1), x - (1 << c.bits), x)
                chans.append((c, shift, x.reshape(m0.height, m0.width).astype(np.float32)))
            shift += c.bits
        W, H = m0.width * factor, m0.height * factor
        levels = []
        arrs = []
        for _, _, x in chans:
            up = np.asarray(Image.fromarray(x, "F").resize((W, H), Image.LANCZOS), dtype=np.float32)
            # Add back each block's error so a box downscale returns the source
            # exactly: smooth like Lanczos, without its drift on noisy bump data.
            err = x - up.reshape(m0.height, factor, m0.width, factor).mean(axis=(1, 3))
            arrs.append(up + np.repeat(np.repeat(err, factor, axis=0), factor, axis=1))
        while True:
            levels.append(arrs)
            h, w = arrs[0].shape
            if w == 1 and h == 1:
                break
            nw, nh = max(1, w // 2), max(1, h // 2)
            arrs = [np.asarray(Image.fromarray(a, "F").resize((nw, nh), Image.BOX), dtype=np.float32) for a in arrs]
            if len(levels) >= max(len(self.mips), 1) and len(self.mips) == 1:
                break
        out, mips, off = bytearray(), [], 0
        for arrs in levels:
            h, w = arrs[0].shape
            word = np.zeros(h * w, dtype=np.uint64)
            for (c, sh, _), a in zip(chans, arrs):
                lo, hi = (-(1 << (c.bits - 1)), (1 << (c.bits - 1)) - 1) if c.signed else (0, (1 << c.bits) - 1)
                q = np.clip(np.rint(a.reshape(-1)), lo, hi).astype(np.int64) & ((1 << c.bits) - 1)
                word |= q.astype(np.uint64) << np.uint64(sh)
            level = np.stack([((word >> np.uint64(8 * k)) & np.uint64(0xFF)).astype(np.uint8) for k in range(bpp)], axis=1)
            mips.append(Mip(w, h, off))
            out += level.tobytes()
            off += h * w * bpp
        return replace(self, mips=mips, data=bytes(out), num_faces=1)


def _channel_masks(chans: list[Channel]) -> tuple[int, int, int, int]:
    """RGBA bit masks from nif channel descriptors (type 0=R 1=G 2=B 3=A)."""
    masks = [0, 0, 0, 0]
    shift = 0
    for c in chans:
        if c.type in (0, 1, 2, 3) and c.bits:
            masks[c.type] = ((1 << c.bits) - 1) << shift
        shift += c.bits
    return tuple(masks)


@dataclass
class TextureNif:
    header: Header
    pixels: PixelData
    footer: bytes = field(repr=False)

    @classmethod
    def parse(cls, buf: bytes) -> "TextureNif":
        hs = next((h for h in VERSIONS if buf.startswith(h)), None)
        if hs is None:
            raise NifError("not a Gamebryo 20.3.0.9/20.6.0.0 file")
        o = len(hs)
        version, endian, user_version, nblocks = struct.unpack_from("<IBII", buf, o)
        o += 13
        if version != VERSIONS[hs] or endian != 1 or user_version != 0:
            raise NifError(f"unsupported header: version {version:#x} endian {endian} user {user_version}")
        ntypes = struct.unpack_from("<H", buf, o)[0]
        o += 2
        types = []
        for _ in range(ntypes):
            n = struct.unpack_from("<I", buf, o)[0]
            types.append(buf[o + 4:o + 4 + n])
            o += 4 + n
        idx = list(struct.unpack_from(f"<{nblocks}H", buf, o))
        o += 2 * nblocks
        sizes = list(struct.unpack_from(f"<{nblocks}I", buf, o))
        o += 4 * nblocks
        nstr, maxlen = struct.unpack_from("<II", buf, o)
        o += 8
        strings = []
        for _ in range(nstr):
            n = struct.unpack_from("<I", buf, o)[0]
            strings.append(buf[o + 4:o + 4 + n])
            o += 4 + n
        ngroups = struct.unpack_from("<I", buf, o)[0]
        groups = list(struct.unpack_from(f"<{ngroups}I", buf, o + 4))
        o += 4 + 4 * ngroups
        header = Header(hs, version, endian, user_version, types, idx, sizes, strings, maxlen, groups)
        if nblocks != 1 or types[idx[0] & 0x7FFF] not in PIXEL_BLOCKS:
            raise NifError(f"not a single-texture NIF: {[types[i & 0x7FFF].decode() for i in idx]}")
        pixels = PixelData.parse(buf[o:o + sizes[0]], version,
                                 types[idx[0] & 0x7FFF] == b"NiPersistentSrcTextureRendererData")
        return cls(header, pixels, buf[o + sizes[0]:])

    def serialize(self) -> bytes:
        block = self.pixels.serialize()
        header = replace(self.header, block_sizes=[len(block)])
        return header.serialize() + block + self.footer

    def with_dds(self, dds: bytes) -> "TextureNif":
        return replace(self, pixels=self.pixels.with_dds(dds))


def is_texture_nif(buf: bytes) -> bool:
    try:
        TextureNif.parse(buf)
        return True
    except (NifError, struct.error):
        return False


@dataclass
class NifFile:
    """Any 20.3.0.9/20.6.0.0 NIF as header + raw blocks + footer.

    Blocks reference each other by index, never by byte offset, so a block can
    be resized freely as long as its size in the header is updated. That's all
    texture replacement inside model/scene files needs.
    """

    header: Header
    blocks: list[bytes]
    footer: bytes = field(repr=False)

    @classmethod
    def parse(cls, buf: bytes) -> "NifFile":
        hs = next((h for h in VERSIONS if buf.startswith(h)), None)
        if hs is None:
            raise NifError("not a Gamebryo 20.3.0.9/20.6.0.0 file")
        o = len(hs)
        version, endian, user_version, nblocks = struct.unpack_from("<IBII", buf, o)
        o += 13
        if version != VERSIONS[hs] or endian != 1 or user_version != 0:
            raise NifError(f"unsupported header: version {version:#x} endian {endian} user {user_version}")
        ntypes = struct.unpack_from("<H", buf, o)[0]
        o += 2
        types = []
        for _ in range(ntypes):
            n = struct.unpack_from("<I", buf, o)[0]
            types.append(buf[o + 4:o + 4 + n])
            o += 4 + n
        idx = list(struct.unpack_from(f"<{nblocks}H", buf, o))
        o += 2 * nblocks
        sizes = list(struct.unpack_from(f"<{nblocks}I", buf, o))
        o += 4 * nblocks
        nstr, maxlen = struct.unpack_from("<II", buf, o)
        o += 8
        strings = []
        for _ in range(nstr):
            n = struct.unpack_from("<I", buf, o)[0]
            strings.append(buf[o + 4:o + 4 + n])
            o += 4 + n
        ngroups = struct.unpack_from("<I", buf, o)[0]
        groups = list(struct.unpack_from(f"<{ngroups}I", buf, o + 4))
        o += 4 + 4 * ngroups
        blocks = []
        for size in sizes:
            if o + size > len(buf):
                raise NifError("block runs past end of file")
            blocks.append(buf[o:o + size])
            o += size
        header = Header(hs, version, endian, user_version, types, idx, sizes, strings, maxlen, groups)
        return cls(header, blocks, buf[o:])

    def block_type(self, i: int) -> bytes:
        # The top bit of a block type index is a flag in 20.x files; mask it off.
        return self.header.block_types[self.header.block_type_index[i] & 0x7FFF]

    def pixel_blocks(self) -> list[int]:
        return [i for i in range(len(self.blocks)) if self.block_type(i) in PIXEL_BLOCKS]

    def texture_names(self) -> dict[int, str]:
        """NiPixelData block index -> the file name its NiSourceTexture gives it
        (embedded textures keep their authoring name, e.g. HD_WC_Door01_JT.tga)."""
        names: dict[int, str] = {}
        for i, blk in enumerate(self.blocks):
            if self.block_type(i) != b"NiSourceTexture":
                continue
            # NiObjectNET: name (string idx), extra data list, controller; then
            # use_external u8, file name (string idx), [unknown link if external], pixel data link.
            o = 4
            nextra = struct.unpack_from("<I", blk, o)[0]
            o += 4 + 4 * nextra + 4
            use_external = blk[o]
            o += 1
            fname = struct.unpack_from("<i", blk, o)[0]
            o += 4
            if use_external:
                continue
            pix = struct.unpack_from("<i", blk, o)[0]
            if 0 <= fname < len(self.header.strings) and 0 <= pix < len(self.blocks):
                names[pix] = self.header.strings[fname].decode("latin1")
        return names

    def pixels(self, i: int) -> PixelData:
        t = self.block_type(i)
        if t not in PIXEL_BLOCKS:
            raise NifError(f"block {i} is {t!r}, not a pixel block")
        return PixelData.parse(self.blocks[i], self.header.version, t == b"NiPersistentSrcTextureRendererData")

    def set_pixels(self, i: int, px: PixelData) -> None:
        if self.block_type(i) not in PIXEL_BLOCKS:
            raise NifError(f"block {i} is not a pixel block")
        self.blocks[i] = px.serialize()

    def serialize(self) -> bytes:
        header = replace(self.header, block_sizes=[len(b) for b in self.blocks])
        return header.serialize() + b"".join(self.blocks) + self.footer
