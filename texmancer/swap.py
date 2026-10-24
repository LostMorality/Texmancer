"""Swap modified WADs in after the launcher's patch pass, restore them when the game exits.

The launcher re-downloads any archive whose size/mtime no longer matches
PatchInfo/CRC_<pkg>.dat, and it only checks after login. So the window is:

    launcher logs "Game patched successful"  ->  swap mods in
    WizardGraphicalClient.exe exits (or the launcher closes without PLAY)  ->  swap originals back

Swaps are renames on the same volume, so the original file keeps its exact
size and mtime and the next patch pass finds nothing to repair. A state file
records what is swapped so a crash can be undone with ``restore``.
"""

from __future__ import annotations

import ctypes
import re
import json
import os
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from . import DEFAULT_INSTALL, GAME_DATA
from .wad import Wad

PATCH_DONE = "Game patched successful"
# Archives a mod may add that the game doesn't ship: locale overlays the client
# checks before the real package (Locale_<locale>-<package>.wad).
ADDED_PREFIX = "locale_"


def is_added(wad_name: str) -> bool:
    return wad_name.lower().startswith(ADDED_PREFIX)


# The client reads archive offsets as signed 32-bit ("Bad position" past 2 GiB).
# Each check is `mov reg,[rcx+0x48]; test reg,reg; js fail`; turning the js into a
# 6-byte NOP makes it unsigned, so archives work up to 4 GiB. Located by signature,
# never by offset, and only applied when every site is found exactly as expected.
_OFFSET_CHECKS = ((re.compile(rb"\x8b\x51\x48\x85\xd2(\x0f\x88....)", re.S), 2),   # read + commit
                  (re.compile(rb"\x8b\x49\x48\x85\xc9(\x0f\x88....)", re.S), 1))   # save entry
_NOP6 = b"\x66\x0f\x1f\x44\x00\x00"
EXTENDED_LIMIT = 0xFFFFFFFF - 1


def patch_client_bytes(exe: bytes) -> bytes | None:
    """The client with its 2 GiB archive checks lifted, or None if this build doesn't
    look exactly like the one the patch was made for."""
    out = bytearray(exe)
    for pat, want in _OFFSET_CHECKS:
        hits = list(pat.finditer(exe))
        if len(hits) != want:
            return None
        for m in hits:
            out[m.start(1):m.end(1)] = _NOP6
    return bytes(out)
SOURCE_SUFFIX = ".source.json"


_FINGERPRINTS: dict[tuple[str, int, int], str] = {}


def table_fingerprint(wad_path: Path) -> str:
    """Hash of a WAD's entry table. It changes when KingsIsle patches the archive,
    but not when the client streams content into it during play (which does
    change size/mtime), so it's the right "game updated?" signal. Cached per
    (path, size, mtime): parsing ~1,200 tables on every UI refresh was slow."""
    import hashlib

    st = wad_path.stat()
    key = (str(wad_path).lower(), st.st_size, st.st_mtime_ns)
    hit = _FINGERPRINTS.get(key)
    if hit is None:
        with Wad(wad_path) as w:
            rows = [(e.name, e.offset, e.size, e.zsize, e.is_zip, e.crc) for e in w.entries]
        hit = _FINGERPRINTS[key] = hashlib.sha256(repr(rows).encode()).hexdigest()
    return hit


def record_source(mod: Path, original: Path) -> None:
    """Remember which original a mod was built from. Swapping a mod built for an
    older game version over newer content would break it."""
    mod.with_name(mod.name + SOURCE_SUFFIX).write_text(json.dumps({"table": table_fingerprint(original)}))


def source_matches(mod: Path, live: Path) -> bool | None:
    """None when the mod has no record (built before records existed)."""
    rec = mod.with_name(mod.name + SOURCE_SUFFIX)
    if not rec.exists():
        return None
    want = json.loads(rec.read_text())
    if "table" in want:
        return table_fingerprint(live) == want["table"]
    st = live.stat()  # older size+mtime records
    return st.st_size == want["size"] and st.st_mtime_ns == want["mtime_ns"]
LAUNCHER_EXE = "wizardlauncher.exe"
CLIENT_EXE = "wizardgraphicalclient.exe"
STATE_NAME = "swap_state.json"

_k32 = ctypes.WinDLL("kernel32", use_last_error=True)
_TH32CS_SNAPPROCESS = 0x2
_SYNCHRONIZE = 0x00100000
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_WAIT_OBJECT_0 = 0
_WAIT_TIMEOUT = 0x102
_INVALID_HANDLE = wintypes.HANDLE(-1).value
_FILE_NOTIFY_CHANGE_SIZE = 0x8
_FILE_NOTIFY_CHANGE_LAST_WRITE = 0x10
_FILE_NOTIFY_CHANGE_FILE_NAME = 0x1


class _PROCESSENTRY32W(ctypes.Structure):
    _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD),
                ("cntThreads", wintypes.DWORD), ("th32ParentProcessID", wintypes.DWORD),
                ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                ("szExeFile", ctypes.c_wchar * 260)]


_k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
_k32.OpenProcess.restype = wintypes.HANDLE
_k32.FindFirstChangeNotificationW.restype = wintypes.HANDLE
_k32.FindFirstChangeNotificationW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL, wintypes.DWORD]
_k32.FindNextChangeNotification.argtypes = [wintypes.HANDLE]
_k32.FindCloseChangeNotification.argtypes = [wintypes.HANDLE]
_k32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_k32.WaitForMultipleObjects.argtypes = [wintypes.DWORD, ctypes.POINTER(wintypes.HANDLE), wintypes.BOOL, wintypes.DWORD]
_k32.CloseHandle.argtypes = [wintypes.HANDLE]
_k32.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]


def find_pids(exe: str) -> list[int]:
    snap = _k32.CreateToolhelp32Snapshot(_TH32CS_SNAPPROCESS, 0)
    if snap == _INVALID_HANDLE:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        entry = _PROCESSENTRY32W()
        entry.dwSize = ctypes.sizeof(entry)
        pids = []
        ok = _k32.Process32FirstW(snap, ctypes.byref(entry))
        while ok:
            if entry.szExeFile.lower() == exe:
                pids.append(entry.th32ProcessID)
            ok = _k32.Process32NextW(snap, ctypes.byref(entry))
        return pids
    finally:
        _k32.CloseHandle(snap)


_PROCESS_TERMINATE = 0x0001


def kill_process(pid: int) -> bool:
    h = _k32.OpenProcess(_PROCESS_TERMINATE, False, pid)
    if not h:
        return False
    try:
        return bool(_k32.TerminateProcess(h, 1))
    finally:
        _k32.CloseHandle(h)


def wait_process_exit(pid: int, timeout_ms: int) -> bool:
    """True once the process is gone (or already was)."""
    h = _k32.OpenProcess(_SYNCHRONIZE | _PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not h:
        return True
    try:
        return _k32.WaitForSingleObject(h, timeout_ms) == _WAIT_OBJECT_0
    finally:
        _k32.CloseHandle(h)


class LogTail:
    """Follows a log that the launcher recreates each start (handles truncation)."""

    def __init__(self, path: Path, from_end: bool = True):
        self.path = path
        self.pos = path.stat().st_size if (from_end and path.exists()) else 0

    def new_lines(self) -> list[str]:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        if size < self.pos:
            self.pos = 0
        if size == self.pos:
            return []
        with open(self.path, "rb") as fh:
            fh.seek(self.pos)
            chunk = fh.read(size - self.pos)
        # Only consume complete lines; a partial last line is re-read next time.
        end = chunk.rfind(b"\n") + 1
        self.pos += end
        return chunk[:end].decode("utf-8", "replace").splitlines()


@dataclass
class Swapper:
    mods_dir: Path
    game_data: Path = GAME_DATA
    install: Path = DEFAULT_INSTALL
    log: Callable[[str], None] = print
    only: set[str] | None = None            # mod folder names to use; None = all
    # (mod folder, WAD name) -> True to leave that archive out (per-area toggles).
    skip: Callable[[str, str], bool] = lambda mod, wad: False
    stop: threading.Event = field(default_factory=threading.Event)
    on_state: Callable[[str], None] = lambda s: None
    # Runs right after each launcher patch pass, before PLAY: config tweaks the
    # patcher would otherwise revert (it rewrites Bin files) are re-applied here.
    after_patch: Callable[[], None] = lambda: None
    # Lift the client's 2 GiB-per-archive limit for this session (needed by mods
    # with archives past 2 GiB); the original exe comes back with the archives.
    patch_client: bool = False
    client_patched: bool = False
    game_running: Callable[[], bool] = lambda: bool(find_pids(CLIENT_EXE))
    backup_path: Path | None = None         # library's Backups\; default mods_dir/originals
    state_file: Path | None = None          # default mods_dir/swap_state.json

    # A mod is a folder in mods_dir holding one or more replacement WADs (plus
    # an optional mod.json {"label": ...}). Its WADs swap in and out together.

    @property
    def state_path(self) -> Path:
        return self.state_file or self.mods_dir / STATE_NAME

    @property
    def backup_dir(self) -> Path:
        return self.backup_path or self.mods_dir / "originals"

    def all_mods(self) -> list[Path]:
        if not self.mods_dir.is_dir():
            return []
        return sorted(d for d in self.mods_dir.iterdir()
                      if d.is_dir() and d.resolve() != self.backup_dir.resolve() and any(d.glob("*.wad")))

    def mods(self) -> list[Path]:
        return [d for d in self.all_mods() if self.only is None or d.name in self.only]

    @staticmethod
    def wads(mod: Path) -> list[Path]:
        return sorted(p for p in mod.glob("*.wad") if p.is_file())

    @staticmethod
    def label(mod: Path) -> str:
        try:
            return json.loads((mod / "mod.json").read_text(encoding="utf-8"))["label"]
        except (OSError, ValueError, KeyError):
            return mod.name

    def is_stale(self, mod: Path) -> bool:
        return any(source_matches(w, self.game_data / w.name) is False for w in self.wads(mod)
                   if not is_added(w.name))

    def usable_wads(self) -> list[Path]:
        """WADs of the selected mods. A mod with any archive the game has since
        updated is skipped whole, so you never get half of a mod."""
        out: list[Path] = []
        seen: dict[str, str] = {}
        for mod in self.mods():
            if self.is_stale(mod):
                self.log(f"skipping mod {mod.name!r}: the game updated an archive since it was built; rebuild it")
                continue
            for w in self.wads(mod):
                if self.skip(mod.name, w.name):
                    continue
                if w.stat().st_size > 0x7FFFFFFF and not self.client_patched:
                    self.log(f"leaving out {mod.name}/{w.name}: it's past 2 GiB and the client isn't patched")
                    continue
                if w.name in seen:
                    raise ValueError(f"{w.name} is in both mod {seen[w.name]!r} and {mod.name!r}; enable only one")
                seen[w.name] = mod.name
                out.append(w)
        return out

    def validate(self) -> None:
        checked, total = 0, 0
        if not self.all_mods():
            raise FileNotFoundError(f"no mod folders with *.wad files in {self.mods_dir}")
        for mod in self.mods():
            for w in self.wads(mod):
                if self.skip(mod.name, w.name):
                    continue
                if not (self.game_data / w.name).exists() and not is_added(w.name):
                    raise FileNotFoundError(f"{mod.name}/{w.name} has no counterpart in {self.game_data}")
                with Wad(w):  # parses the whole table; raises on a corrupt archive
                    checked += 1
                    total += w.stat().st_size
        if checked:
            self.log(f"checked {checked:,} mod archive(s), {total / 2**30:.2f} GiB: all readable")
        if os.path.splitdrive(self.mods_dir.resolve())[0].lower() != os.path.splitdrive(self.game_data.resolve())[0].lower():
            raise OSError("mods dir must be on the same drive as the game so swaps are instant renames")

    def is_swapped(self) -> bool:
        return self.state_path.exists()

    def swap_in(self) -> None:
        if self.is_swapped():
            self.log("already swapped in; skipping")
            return
        if self.game_running():
            raise RuntimeError("game client is running; refusing to swap archives under it")
        self.backup_dir.mkdir(exist_ok=True)
        done: list[str] = []  # paths relative to mods_dir, e.g. "HD Textures/Root.wad"
        try:
            for w in self.usable_wads():
                live = self.game_data / w.name
                if is_added(w.name):
                    # A new archive (locale overlay): nothing of the game's to back up.
                    if live.exists():
                        raise FileExistsError(f"{live.name} already exists in the game folder; refusing to overwrite it")
                    os.replace(w, live)
                    done.append("+" + w.relative_to(self.mods_dir).as_posix())
                    self._write_state(done)
                    continue
                os.replace(live, self.backup_dir / w.name)
                try:
                    os.replace(w, live)
                except OSError:
                    os.replace(self.backup_dir / w.name, live)
                    raise
                done.append(w.relative_to(self.mods_dir).as_posix())
                self._write_state(done)
        except Exception:
            self._restore(done)
            raise
        self.log(f"swapped in: {', '.join(done)}" if done else "no usable mods; playing unmodded")

    @property
    def client_exe(self) -> Path:
        return self.install / "Bin" / "WizardGraphicalClient.exe"

    def _patch_client(self) -> None:
        exe, backup = self.client_exe, self.backup_dir / "WizardGraphicalClient.exe"
        if self.game_running():
            self.log("client already running; not patching it")
            return
        self.backup_dir.mkdir(exist_ok=True)
        if backup.exists():  # a previous session didn't restore: that copy is the original
            os.replace(backup, exe)
        patched = patch_client_bytes(exe.read_bytes())
        if patched is None:
            self.log("client build not recognised; leaving it unpatched (archives over 2 GiB stay out)")
            return
        tmp = exe.with_suffix(".patched.tmp")
        tmp.write_bytes(patched)
        os.replace(exe, backup)
        try:
            os.replace(tmp, exe)
        except OSError:
            os.replace(backup, exe)
            raise
        self.client_patched = True
        self.log("client patched for archives up to 4 GiB (original restored when the game closes)")

    def _unpatch_client(self) -> None:
        backup = self.backup_dir / "WizardGraphicalClient.exe"
        if backup.exists():
            os.replace(backup, self.client_exe)
            self.client_patched = False
            self.log("original game client restored")

    def restore(self) -> None:
        if not self.is_swapped():
            self._unpatch_client()
            self.log("nothing swapped in")
            return
        entries = json.loads(self.state_path.read_text())["swapped"]
        self._restore(entries)
        self._unpatch_client()
        # The client "patches" every overlay package it registers and leaves CRC caches.
        for f in (self.install / "PatchInfo").glob("CRC_Locale_*.dat"):
            f.unlink(missing_ok=True)
        self.log(f"restored originals: {', '.join(Path(e).name for e in entries)}")

    def _restore(self, entries: list[str]) -> None:
        # Entries are "Mod/Name.wad"; a bare "Name.wad" is the old flat layout.
        for rel in reversed(entries):
            if rel.startswith("+"):  # an added archive goes home; there's no original
                rel = rel[1:]
                home = self.mods_dir / rel
                home.parent.mkdir(parents=True, exist_ok=True)
                live = self.game_data / Path(rel).name
                if live.exists():
                    os.replace(live, home)
                continue
            name = Path(rel).name
            live, orig, home = self.game_data / name, self.backup_dir / name, self.mods_dir / rel
            if not orig.exists():
                raise FileNotFoundError(f"backup {orig} missing; repair with the launcher (it re-downloads)")
            home.parent.mkdir(parents=True, exist_ok=True)
            os.replace(live, home)
            os.replace(orig, live)
        self.state_path.unlink(missing_ok=True)

    def _write_state(self, entries: list[str]) -> None:
        self.state_path.write_text(json.dumps({"swapped": entries, "time": time.time()}))

    # -- watcher ------------------------------------------------------------

    def run(self, once: bool = False) -> None:
        """Wait for patch completion, swap in, wait for the game to end, restore.

        Stops when ``self.stop`` is set (or Ctrl+C on the CLI). Originals are
        never left swapped in unless the game is still running at stop time.
        """
        self.validate()
        if self.is_swapped():
            self.log("found a leftover swap from a previous run; restoring first")
            self.restore()
        banks = [self.install / "PatchClient" / b for b in ("BankA", "BankB")]
        logs = [b / "WizardLauncher.Log.txt" for b in banks]
        tails = [LogTail(p) for p in logs]
        # Launcher already open (maybe already patched): re-read its current
        # log from the top. The launcher recreates the log each start, so only
        # the newest one belongs to the running launcher.
        if find_pids(LAUNCHER_EXE) and not find_pids(CLIENT_EXE):
            newest = max((p for p in logs if p.exists()), key=lambda p: p.stat().st_mtime, default=None)
            for t in tails:
                if t.path == newest:
                    t.pos = 0
        watches = [_k32.FindFirstChangeNotificationW(str(b), False, _FILE_NOTIFY_CHANGE_SIZE
                                                     | _FILE_NOTIFY_CHANGE_LAST_WRITE | _FILE_NOTIFY_CHANGE_FILE_NAME)
                   for b in banks if b.is_dir()]
        if any(h == _INVALID_HANDLE for h in watches):
            raise ctypes.WinError(ctypes.get_last_error())
        arr = (wintypes.HANDLE * len(watches))(*watches)
        try:
            while not self.stop.is_set():
                self.on_state("waiting_login")
                self.log("waiting for the launcher to finish patching (log in now)...")
                if not self._wait_for_patch(tails, arr):
                    break
                try:
                    self.after_patch()
                except Exception as exc:  # a tweak failing must never block the mods
                    self.log(f"post-patch tweak failed: {exc!r}")
                if self.patch_client and any(w.stat().st_size > 0x7FFFFFFF for m in self.mods()
                                             for w in self.wads(m) if not self.skip(m.name, w.name)):
                    self._patch_client()
                self.swap_in()
                self.on_state("mods_active")
                self._wait_for_session_end()
                self.restore()
                self.on_state("restored")
                if once:
                    return
        except KeyboardInterrupt:
            self.log("stopping")
        finally:
            for h in watches:
                _k32.FindCloseChangeNotification(h)
            if self.is_swapped():
                if find_pids(CLIENT_EXE):
                    self.log("game still running: originals NOT restored; run `texmancer swap restore` after it exits")
                else:
                    self.restore()
            self.on_state("stopped")

    def _wait_for_patch(self, tails: list[LogTail], arr) -> bool:
        """True once patching finished; False if stopped or the launcher closed first."""
        seen_launcher = False
        while not self.stop.is_set():
            for t in tails:
                if any(PATCH_DONE in line for line in t.new_lines()):
                    self.log("launcher patch pass finished")
                    return True
            running = bool(find_pids(LAUNCHER_EXE))
            if running:
                seen_launcher = True
            elif seen_launcher:
                self.log("launcher closed before logging in")
                return False
            # Directory change notification wakes us on log writes; the timeout
            # keeps stop/launcher-exit checks responsive.
            r = _k32.WaitForMultipleObjects(len(arr), arr, False, 1000)
            if _WAIT_OBJECT_0 <= r < _WAIT_OBJECT_0 + len(arr):
                _k32.FindNextChangeNotification(arr[r - _WAIT_OBJECT_0])
            elif r != _WAIT_TIMEOUT:  # WAIT_FAILED: fall back to a plain 1 s tick
                time.sleep(1)
        return False

    def _guard_launchers(self, allowed: set[int]) -> None:
        """A launcher started after the swap would patch the swapped-in archives
        back to KI's versions (and its login kicks the running session). Close it
        before it gets that far; the launcher already open at swap time is fine."""
        for pid in find_pids(LAUNCHER_EXE):
            if pid not in allowed and kill_process(pid):
                self.log(f"closed a second Wizard101 launcher (pid {pid}) so it can't patch over the active mods")

    def _wait_for_session_end(self) -> None:
        self.log("mods active; press PLAY. Originals come back when the game (or launcher) closes.")
        allowed = set(find_pids(LAUNCHER_EXE))
        announced = False
        while True:
            self._guard_launchers(allowed)
            clients = find_pids(CLIENT_EXE)
            if clients:
                if not announced:
                    self.on_state("playing")
                    self.log(f"game running (pid {clients[0]}); waiting for it to exit")
                    announced = True
                for pid in clients:
                    while not wait_process_exit(pid, 1000):
                        self._guard_launchers(allowed)
                # The launcher sometimes relaunches the client; make sure it's really gone.
                if not find_pids(CLIENT_EXE):
                    self.log("game exited")
                    return
                continue
            if self.stop.is_set():
                return
            launchers = find_pids(LAUNCHER_EXE)
            if not launchers:
                self.log("launcher closed without starting the game")
                return
            wait_process_exit(launchers[0], 1000)
