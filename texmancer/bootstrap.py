"""First-run setup for the one-click installer: finds the game and fetches the
Python runtime, packages, tools and AI model from their official sources.

Standard library only (it runs inside the setup exe before anything else is
installed). Every file with a fixed release is checked against a pinned
SHA-256; Python packages come from PyPI and the official PyTorch index.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.request
import winreg
import zipfile
from pathlib import Path
from typing import Callable

from . import DEFAULT_INSTALL

PY_VERSION = "3.11.9"
PY_TAG = "311"

# name -> (url, sha256)
DOWNLOADS = {
    "python": (f"https://www.python.org/ftp/python/{PY_VERSION}/python-{PY_VERSION}-embed-amd64.zip",
               "009d6bf7e3b2ddca3d784fa09f90fe54336d5b60f0e0f305c37f400bf83cfd3b"),
    "pip": ("https://files.pythonhosted.org/packages/3f/06/3e4420302363fbb71be3c7f35ce042f54bfca330e3e66b3a2e3c4cca579f/"
            "pip-26.2.2-py3-none-any.whl", "fd7f4dfa8190e014013b9f08f3e973f4e4c26ffd179a204895533a7254813a70"),
    "texconv": ("https://github.com/microsoft/DirectXTex/releases/download/may2026/texconv.exe",
                "dcfdec10244e02cf5037fba089c55fb7e1326b1c8181742d77d15fa5cb5eef06"),
    "model": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
              "f872d837d3c90ed2e05227bed711af5671a6fd1c9f7d7e91c911a61f155e99da"),
    "ncnn": ("https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/"
             "realesrgan-ncnn-vulkan-20220424-windows.zip",
             "abc02804e17982a3be33675e4d471e91ea374e65b70167abc09e31acb412802d"),
}

BASE_PACKAGES = ["numpy==2.4.6", "pillow==12.3.0"]
TORCH_PACKAGES = ["spandrel==0.4.2"]
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

Log = Callable[[str], None]
Progress = Callable[[int, int], None]


class SetupError(RuntimeError):
    pass


def is_install(path: Path) -> bool:
    return (path / "Data" / "GameData" / "Root.wad").is_file()


def find_install() -> Path | None:
    """The Wizard101 folder: the default location, then the uninstall entries
    (KingsIsle and Steam installs register one), then Steam's default library."""
    cands = [DEFAULT_INSTALL]
    for root, sub in ((winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
                      (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
                      (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall")):
        try:
            key = winreg.OpenKey(root, sub)
        except OSError:
            continue
        with key:
            for i in range(winreg.QueryInfoKey(key)[0]):
                try:
                    with winreg.OpenKey(key, winreg.EnumKey(key, i)) as app:
                        name = str(winreg.QueryValueEx(app, "DisplayName")[0])
                        if "wizard101" in name.lower():
                            cands.append(Path(str(winreg.QueryValueEx(app, "InstallLocation")[0])))
                except OSError:
                    continue
    pf86 = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)"))
    cands.append(pf86 / "Steam" / "steamapps" / "common" / "Wizard101")
    return next((c for c in cands if is_install(c)), None)


def game_locale(install: Path) -> str:
    """The client's locale from its own log (``LOCALE = en-US``), else en-US."""
    try:
        text = (install / "Bin" / "WizardClient.log").read_text(encoding="utf-8", errors="replace")
    except OSError:
        return "en-US"
    m = re.findall(r"LOCALE = ([A-Za-z]{2}-[A-Za-z]{2})", text)
    return m[-1] if m else "en-US"


def default_base(install: Path) -> Path:
    """Where Texmancer keeps its files. The mods have to be on the game's drive
    (swaps are instant renames), so it's %LOCALAPPDATA% only if that's there."""
    local = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local") / "Texmancer"
    if os.path.splitdrive(str(local))[0].lower() == os.path.splitdrive(str(install))[0].lower():
        return local
    return Path(os.path.splitdrive(str(install))[0] + "\\") / "Texmancer"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name: str, dest: Path, progress: Progress | None = None) -> Path:
    """Fetch DOWNLOADS[name] to ``dest`` unless a copy with the right hash is there."""
    url, digest = DOWNLOADS[name]
    if dest.is_file() and _sha256(dest) == digest:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "Texmancer-Setup"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp, open(part, "wb") as out:
            total = int(resp.headers.get("Content-Length") or 0)
            done = 0
            while block := resp.read(1 << 20):
                out.write(block)
                done += len(block)
                if progress:
                    progress(done, total)
    except OSError as exc:
        part.unlink(missing_ok=True)
        raise SetupError(f"couldn't download {url}: {exc}") from exc
    if _sha256(part) != digest:
        part.unlink(missing_ok=True)
        raise SetupError(f"{url} didn't match its expected checksum; not using it")
    part.replace(dest)
    return dest


class Layout:
    """Texmancer's folder: runtime\\ (Python), app\\ (this package), tools\\,
    models\\, work\\ (per-stage projects), mods\\HD (the mod), logs\\."""

    def __init__(self, base: Path):
        self.base = Path(base)
        self.runtime = self.base / "runtime"
        self.app = self.base / "app"
        self.tools = self.base / "tools"
        self.models = self.base / "models"
        self.work = self.base / "work"
        self.mods = self.base / "mods"
        self.mod = self.mods / "HD"
        self.staging = self.base / "build" / "HD"
        self.logs = self.base / "logs"
        self.state_file = self.base / "state.json"

    @property
    def python(self) -> Path:
        return self.runtime / "python.exe"

    @property
    def model(self) -> Path:
        return self.models / "RealESRGAN_x4plus_anime_6B.pth"

    def load_state(self) -> dict:
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def save_state(self, state: dict) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        tmp = self.state_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=1), encoding="utf-8")
        tmp.replace(self.state_file)


def deploy_app(layout: Layout, package_dir: Path) -> None:
    """Copy this package (from the setup exe) into app\\texmancer for the runtime."""
    dest = layout.app / "texmancer"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(package_dir, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))


def _pip(layout: Layout, args: list[str], log: Log) -> None:
    cmd = [str(layout.python), "-m", "pip", "install", "--no-warn-script-location", "--disable-pip-version-check",
           "--progress-bar", "off", *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace",
                            creationflags=_NO_WINDOW)
    tail: list[str] = []
    for line in proc.stdout:  # type: ignore[union-attr]
        line = line.rstrip()
        if line:
            tail = (tail + [line])[-15:]
            if line.startswith(("Collecting", "Downloading", "Installing", "Successfully")):
                log(line)
    if proc.wait() != 0:
        raise SetupError("installing Python packages failed:\n" + "\n".join(tail))


def _has_packages(layout: Layout, modules: list[str]) -> bool:
    if not layout.python.exists():
        return False
    code = "import " + ", ".join(modules)
    return subprocess.run([str(layout.python), "-c", code], capture_output=True, creationflags=_NO_WINDOW).returncode == 0


def ensure_runtime(layout: Layout, backend: str, torch_build: tuple[str, str] | None,
                   log: Log, progress: Progress | None = None) -> None:
    """Install whatever is missing; quick when everything is already there."""
    dl = layout.base / "downloads"
    if not layout.python.exists():
        log("Downloading Python (python.org)")
        z = download("python", dl / "python-embed.zip", progress)
        layout.runtime.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(layout.runtime)
        # The embeddable build ignores site-packages until its ._pth file says otherwise.
        (layout.runtime / f"python{PY_TAG}._pth").write_text(
            f"python{PY_TAG}.zip\n.\nLib\\site-packages\n..\\app\nimport site\n", encoding="ascii")
    if not _has_packages(layout, ["pip"]):
        log("Setting up pip (pypi.org)")
        whl = download("pip", dl / "pip-26.2.2-py3-none-any.whl", progress)
        # A wheel is a zip laid out for site-packages (dist-info included). pip
        # won't install itself from its own wheel path, and the embeddable
        # Python ignores PYTHONPATH, so unpack it directly.
        with zipfile.ZipFile(whl) as zf:
            zf.extractall(layout.runtime / "Lib" / "site-packages")
        if not _has_packages(layout, ["pip"]):
            raise SetupError("setting up pip failed")
    if backend == "torch":
        if not torch_build:
            raise SetupError("this graphics card has no CUDA build; use the Vulkan backend")
        index, version = torch_build
        if not _has_packages(layout, ["torch", "spandrel", "numpy", "PIL"]):
            log(f"Installing PyTorch {version} for NVIDIA (download.pytorch.org, about 3 GB)")
            _pip(layout, [f"torch=={version}", "torchvision", "--index-url", f"https://download.pytorch.org/whl/{index}"],
                 log)
            log("Installing image packages (pypi.org)")
            _pip(layout, BASE_PACKAGES + TORCH_PACKAGES, log)
        if not layout.model.exists() or _sha256(layout.model) != DOWNLOADS["model"][1]:
            log("Downloading the AI model (Real-ESRGAN on GitHub)")
            download("model", layout.model, progress)
    else:
        if not _has_packages(layout, ["numpy", "PIL"]):
            log("Installing image packages (pypi.org)")
            _pip(layout, BASE_PACKAGES, log)
        exe = layout.tools / "realesrgan-ncnn-vulkan" / "realesrgan-ncnn-vulkan.exe"
        if not exe.exists():
            log("Downloading Real-ESRGAN for Vulkan (Real-ESRGAN on GitHub)")
            z = download("ncnn", dl / "realesrgan-ncnn-vulkan.zip", progress)
            with zipfile.ZipFile(z) as zf:
                for m in zf.namelist():  # just the program and its models, not the demo video
                    if not m.endswith("/") and (m.endswith((".exe", ".dll")) or m.startswith("models/")):
                        target = layout.tools / "realesrgan-ncnn-vulkan" / m
                        target.parent.mkdir(parents=True, exist_ok=True)
                        target.write_bytes(zf.read(m))
    texconv = layout.tools / "texconv.exe"
    if not texconv.exists():
        log("Downloading texconv (Microsoft DirectXTex on GitHub)")
        download("texconv", texconv, progress)
    shutil.rmtree(dl, ignore_errors=True)


def free_bytes(path: Path) -> int:
    p = Path(path)
    while not p.exists() and p.parent != p:
        p = p.parent
    return shutil.disk_usage(p).free
