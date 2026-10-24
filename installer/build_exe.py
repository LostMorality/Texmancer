"""Build dist/Texmancer-Setup.exe (PyInstaller one-file, windowed).

    python -m pip install pyinstaller
    python installer/build_exe.py

The exe holds the setup window plus a copy of the texmancer package, which it
deploys into the runtime it downloads. After building, every file inside the
exe is unpacked and scanned for this PC's user name, home folder and computer
name; the build fails if any turn up.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUILD = ROOT / "build"
DIST = ROOT / "dist"
NAME = "Texmancer-Setup"


def personal_markers() -> list[re.Pattern[bytes]]:
    """Ways this PC could leak into a build: any absolute path under a Users
    folder, the home folder, and the user or computer name as a path part
    (\\name\\, /name/, \\\\COMPUTER). Bare words aren't checked: short names
    are often ordinary words that the standard library uses too."""
    names = {os.environ.get("USERNAME", ""), os.environ.get("COMPUTERNAME", ""), Path.home().name}
    names = {n for n in names if len(n) >= 2}
    sep = rb"[\\/]"
    pats = [rb"[A-Za-z]:" + sep + rb"+Users" + sep + rb"+[^\\/\x00\r\n\"']+", re.escape(str(Path.home()).encode())]
    pats += [sep + re.escape(n.encode()) + sep for n in names]
    # The same in UTF-16 (Windows API strings in the bootloader and DLLs).
    sep16 = rb"[\\/]\x00"
    pats += [sep16 + re.escape(n.encode("utf-16-le")) + sep16 for n in names]
    return [re.compile(p, re.I) for p in pats]


def scan(exe: Path) -> list[str]:
    """Every member of the exe (and of its PYZ module archive), decompressed."""
    from PyInstaller.archive.readers import CArchiveReader, ZlibArchiveReader

    markers = personal_markers()
    hits: list[str] = []

    def check(label: str, data: bytes) -> None:
        for m in markers:
            found = m.search(data)
            if found:
                hits.append(f"{label}: contains {found.group(0)!r}")
                return

    check("exe", exe.read_bytes())
    arch = CArchiveReader(str(exe))
    tmp = BUILD / "scan"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    for name in arch.toc:
        data = arch.extract(name)
        check(name, data)
        if name.endswith(".pyz"):
            pyz = tmp / "mods.pyz"
            pyz.write_bytes(data)
            z = ZlibArchiveReader(str(pyz))
            for mod in z.toc:
                try:
                    code = z.extract(mod, raw=True)
                except TypeError:  # older PyInstaller
                    code = z.extract(mod)
                if isinstance(code, tuple):
                    code = code[1]
                if isinstance(code, bytes):
                    try:
                        code = zlib.decompress(code)
                    except zlib.error:
                        pass
                    check(f"{name}/{mod}", code)
    return hits


def main() -> int:
    import PyInstaller.__main__

    pkg = BUILD / "pkg"
    shutil.rmtree(BUILD, ignore_errors=True)
    shutil.copytree(ROOT / "texmancer", pkg / "texmancer", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    PyInstaller.__main__.run([
        str(ROOT / "installer" / "setup_app.py"),
        "--name", NAME, "--onefile", "--windowed", "--noconfirm", "--clean",
        "--distpath", str(DIST), "--workpath", str(BUILD / "pyi"), "--specpath", str(BUILD),
        "--paths", str(ROOT),
        "--add-data", f"{pkg / 'texmancer'}{os.pathsep}texmancer_src",
        # The window needs only the standard library; keep the big packages out.
        *[a for m in ("numpy", "PIL", "torch", "spandrel", "matplotlib", "scipy", "pandas")
          for a in ("--exclude-module", m)],
    ])
    exe = DIST / f"{NAME}.exe"
    hits = scan(exe)
    if hits:
        print("\nPersonal info found in the build:\n  " + "\n  ".join(hits[:20]), file=sys.stderr)
        return 1
    print(f"\n{exe} ({exe.stat().st_size / 2**20:.1f} MB): no personal info found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
