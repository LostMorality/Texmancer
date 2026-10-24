"""KIWAD archive reader/writer for Wizard101 / Pirate101.

Layout (little-endian), verified against the live W.1.610 client data:

    header   b"KIWAD" | u32 version | u32 entry_count | [u8 flags if version >= 2]
    entry    u32 offset | u32 size | u32 zsize | u8 is_zip | u32 crc | u32 name_len | name (NUL-terminated)

``size`` is the uncompressed length, ``zsize`` the stored (zlib) length when
``is_zip`` is set and 0xFFFFFFFF otherwise. ``crc`` is KingsIsle's CRC-32 of the
*stored* bytes: the reflected IEEE polynomial with initial value 0 and no final
XOR (found in wizpatch; verified on 400/400 Root.wad entries). The writer
computes it for new entries and copies it for untouched ones.
"""

from __future__ import annotations

import fnmatch
import os
import struct
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable, Iterator

MAGIC = b"KIWAD"
_HEADER = struct.Struct("<II")
_ENTRY = struct.Struct("<IIIBII")
_NO_ZSIZE = 0xFFFFFFFF  # KI's zsize for stored (uncompressed) entries
# The client reads entry offsets as *signed* 32-bit: anything at or past 2 GiB
# comes back negative ("Bad position (-2023466685)") and fails to load, even
# though the format field is unsigned (past it the client logs "WAD_ELEM Bad position").
MAX_WAD_BYTES = 0x7FFFFFFF
# With the client's offset check lifted (swap.patch_client_bytes) archives can use
# the full unsigned 32-bit range; 0xFFFFFFFF itself reads as an error.
EXTENDED_WAD_BYTES = 0xFFFFFFFE


def ki_crc32(data: bytes) -> int:
    """KingsIsle CRC-32: zlib's table, but init 0 and no final XOR."""
    return zlib.crc32(data, 0xFFFFFFFF) ^ 0xFFFFFFFF


class WadError(Exception):
    pass


class EntryNotDownloaded(WadError):
    """The client streams assets on demand; until an asset is fetched its slot
    in the archive is reserved but zero-filled."""


@dataclass(slots=True)
class WadEntry:
    name: str
    offset: int
    size: int
    zsize: int
    is_zip: bool
    crc: int

    @property
    def stored_size(self) -> int:
        return self.zsize if self.is_zip else self.size


class Wad:
    """Random-access reader. Use as a context manager."""

    def __init__(self, path: str | os.PathLike[str]):
        self.path = Path(path)
        self._fh: BinaryIO = open(self.path, "rb")
        try:
            self.version, self.flags, self.entries = self._read_journal(self._fh)
        except Exception:
            self._fh.close()
            raise
        self._index = {e.name.lower(): e for e in self.entries}

    @staticmethod
    def _read_journal(fh: BinaryIO) -> tuple[int, int, list[WadEntry]]:
        if fh.read(5) != MAGIC:
            raise WadError("not a KIWAD archive")
        raw = fh.read(_HEADER.size)
        if len(raw) != _HEADER.size:
            raise WadError("truncated header")
        version, count = _HEADER.unpack(raw)
        flags = fh.read(1)[0] if version >= 2 else 0
        file_len = os.fstat(fh.fileno()).st_size
        entries: list[WadEntry] = []
        for i in range(count):
            raw = fh.read(_ENTRY.size)
            if len(raw) != _ENTRY.size:
                raise WadError(f"truncated entry table at entry {i}")
            off, size, zsize, is_zip, crc, name_len = _ENTRY.unpack(raw)
            if name_len > 4096:
                raise WadError(f"implausible name length {name_len} at entry {i}")
            name = fh.read(name_len).rstrip(b"\0").decode("utf-8", "replace")
            entry = WadEntry(name, off, size, zsize, bool(is_zip), crc)
            if off + entry.stored_size > file_len:
                raise WadError(f"entry {name!r} points past end of file")
            entries.append(entry)
        return version, flags, entries

    def close(self) -> None:
        self._fh.close()

    def __enter__(self) -> "Wad":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __len__(self) -> int:
        return len(self.entries)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self._index

    def get_entry(self, name: str) -> WadEntry:
        try:
            return self._index[name.lower()]
        except KeyError:
            raise KeyError(f"{name!r} not in {self.path.name}") from None

    def glob(self, pattern: str) -> Iterator[WadEntry]:
        pat = pattern.lower()
        return (e for e in self.entries if fnmatch.fnmatchcase(e.name.lower(), pat))

    def read_raw(self, entry: WadEntry) -> bytes:
        """Stored bytes exactly as they sit in the archive."""
        self._fh.seek(entry.offset)
        data = self._fh.read(entry.stored_size)
        if len(data) != entry.stored_size:
            raise WadError(f"short read for {entry.name!r}")
        return data

    def read(self, name_or_entry: str | WadEntry) -> bytes:
        entry = self.get_entry(name_or_entry) if isinstance(name_or_entry, str) else name_or_entry
        raw = self.read_raw(entry)
        if not entry.is_zip:
            return raw
        if raw[:1] == b"\0" and not any(raw):
            raise EntryNotDownloaded(f"{entry.name!r} has not been downloaded by the client yet")
        data = zlib.decompress(raw)
        if len(data) != entry.size:
            raise WadError(f"{entry.name!r}: decompressed {len(data)} bytes, expected {entry.size}")
        return data

    def is_downloaded(self, entry: WadEntry) -> bool:
        """Cheap check: a streamed-in zlib entry starts with a 0x78 header byte."""
        if not entry.is_zip or entry.zsize == 0:
            return True
        self._fh.seek(entry.offset)
        return self._fh.read(1) != b"\0"

    def extract(self, out_dir: str | os.PathLike[str], pattern: str = "*") -> tuple[int, int]:
        """Returns (extracted, skipped_not_downloaded)."""
        out = Path(out_dir).resolve()
        count = skipped = 0
        for entry in self.glob(pattern):
            dest = (out / entry.name).resolve()
            # Archive names are untrusted: refuse anything that escapes out_dir.
            if out not in dest.parents:
                raise WadError(f"refusing path traversal in entry name {entry.name!r}")
            try:
                data = self.read(entry)
            except EntryNotDownloaded:
                skipped += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            count += 1
        return count, skipped


def write_wad(
    dest: str | os.PathLike[str],
    files: Iterable[tuple[str, bytes | Path | WadEntry]],
    source: Wad | None = None,
    compress_level: int = 6,
    version: int = 1,
    flags: int = 0,
    crcs: dict[str, int] | None = None,
    max_bytes: int = MAX_WAD_BYTES,
) -> int:
    """Write a KIWAD archive (no larger than ``max_bytes``).

    ``files`` yields ``(name, payload)``. ``bytes`` or a ``Path`` (read lazily,
    so multi-GB replacement sets never sit in memory uncompressed) is new data
    and gets compressed on a thread pool (zlib releases the GIL); a ``WadEntry``
    from ``source`` is copied verbatim (same compression, same crc).
    Writes to a temp file and renames, so a failed pack never clobbers ``dest``.
    """
    dest = Path(dest)
    crcs = crcs or {}
    slots: list[tuple[str, bytes | Path | None, tuple | None]] = []
    for name, payload in files:
        if isinstance(payload, WadEntry):
            if source is None:
                raise ValueError("WadEntry payload requires source=")
            slots.append((name, None, (source.read_raw(payload), payload.size, payload.is_zip, payload.crc)))
        else:
            slots.append((name, payload, None))

    def pack(slot):
        name, payload, copied = slot
        if copied is not None:
            return (name, *copied)
        data = payload.read_bytes() if isinstance(payload, Path) else payload
        packed = zlib.compress(data, compress_level)
        stored, is_zip = (packed, True) if len(packed) < len(data) else (data, False)
        return name, stored, len(data), is_zip, crcs.get(name.lower(), ki_crc32(stored))

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as pool:
        items: list[tuple[str, bytes, int, bool, int]] = list(pool.map(pack, slots))

    names = [n.encode("utf-8") + b"\0" for n, *_ in items]
    table_len = 5 + _HEADER.size + (1 if version >= 2 else 0)
    table_len += sum(_ENTRY.size + len(n) for n in names)

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        _write_body(tmp, items, names, table_len, version, flags, max_bytes)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, dest)
    return len(items)


def _write_body(tmp: Path, items, names, table_len: int, version: int, flags: int,
                max_bytes: int = MAX_WAD_BYTES) -> None:
    with open(tmp, "wb") as fh:
        fh.write(MAGIC + _HEADER.pack(version, len(items)))
        if version >= 2:
            fh.write(bytes([flags]))
        offset = table_len
        for (name, stored, size, is_zip, crc), bname in zip(items, names):
            if offset + len(stored) > max_bytes:
                label = "2 GiB" if max_bytes <= MAX_WAD_BYTES else "4 GiB"
                raise WadError(f"archive would pass the client's {label} archive limit at {name!r}")
            zsize = len(stored) if is_zip else _NO_ZSIZE
            fh.write(_ENTRY.pack(offset, size, zsize, int(is_zip), crc, len(bname)) + bname)
            offset += len(stored)
        for _, stored, *_ in items:
            fh.write(stored)


def repack(src: str | os.PathLike[str], dest: str | os.PathLike[str], replacements: dict[str, bytes | Path],
           keep_crc: bool = False, max_bytes: int = MAX_WAD_BYTES) -> int:
    """Copy ``src`` to ``dest`` swapping in ``replacements`` (name -> new bytes or a file to read).

    ``keep_crc`` writes the original entry's crc for replaced files instead of 0,
    for testing whether the patcher trusts the table rather than recomputing.
    """
    repl = {k.lower(): (k, v) for k, v in replacements.items()}
    with Wad(src) as wad:
        crcs = {e.name.lower(): e.crc for e in wad.entries} if keep_crc else {}

        def gen():
            for e in wad.entries:
                hit = repl.pop(e.name.lower(), None)
                yield (e.name, hit[1]) if hit else (e.name, e)
            for name, data in repl.values():  # brand-new files
                yield name, data
        return write_wad(dest, gen(), source=wad, version=wad.version, flags=wad.flags, crcs=crcs,
                         max_bytes=max_bytes)


def pack_dir(root: str | os.PathLike[str], dest: str | os.PathLike[str]) -> int:
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    return write_wad(dest, ((p.relative_to(root).as_posix(), p.read_bytes()) for p in files))
