"""Graphics card detection and the Low / Balanced / Max power settings.

Standard library only: the setup program imports this before any of the heavy
packages are installed. DXGI lists every adapter whatever the vendor;
nvidia-smi adds the CUDA compute capability and live free memory on NVIDIA.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from ctypes import wintypes
from dataclasses import dataclass

NVIDIA, AMD, INTEL = 0x10DE, 0x1002, 0x8086
_VENDORS = {NVIDIA: "NVIDIA", AMD: "AMD", 0x1022: "AMD", INTEL: "Intel"}

# PyTorch's current CUDA wheels need Turing (7.5) or newer and a recent driver;
# anything older uses Vulkan. (index, torch version, minimum driver)
MIN_CUDA_CAPABILITY = (7, 5)
TORCH_BUILDS = (("cu130", "2.14.1", 580), ("cu128", "2.11.0", 570), ("cu126", "2.14.1", 560))


@dataclass
class Gpu:
    name: str
    vendor: str
    vram: int                      # dedicated memory, bytes
    compute_cap: tuple[int, int] | None = None
    free: int | None = None        # bytes free right now (NVIDIA only)
    driver: float | None = None    # NVIDIA driver major.minor, e.g. 581.57

    @property
    def torch_build(self) -> tuple[str, str] | None:
        """(wheel index, torch version) that runs on this card, or None for Vulkan."""
        if self.vendor != "NVIDIA" or not self.compute_cap or self.compute_cap < MIN_CUDA_CAPABILITY                 or not self.driver:
            return None
        for index, version, min_driver in TORCH_BUILDS:
            # The cu126 build has no Blackwell (12.x) kernels.
            if self.driver >= min_driver and not (index == "cu126" and self.compute_cap[0] >= 12):
                return index, version
        return None

    @property
    def backend(self) -> str:
        """"torch" (CUDA) or "ncnn" (Vulkan, any vendor)."""
        return "torch" if self.torch_build else "ncnn"

    def describe(self) -> str:
        how = "CUDA" if self.backend == "torch" else "Vulkan"
        return f"{self.name} ({self.vram / 2**30:.0f} GB, uses {how})"


class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _DXGI_ADAPTER_DESC1(ctypes.Structure):
    _fields_ = [("Description", wintypes.WCHAR * 128), ("VendorId", wintypes.UINT), ("DeviceId", wintypes.UINT),
                ("SubSysId", wintypes.UINT), ("Revision", wintypes.UINT), ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t), ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", _LUID), ("Flags", wintypes.UINT)]


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


_IID_IDXGIFactory1 = _GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
_DXGI_ADAPTER_FLAG_SOFTWARE = 2
_DXGI_ERROR_NOT_FOUND = 0x887A0002


def _vcall(obj: ctypes.c_void_p, index: int, restype, argtypes: tuple = (), *args):
    """Call COM method ``index`` of ``obj``'s vtable."""
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    fn = ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])
    return fn(obj, *args)


def dxgi_adapters() -> list[Gpu]:
    """Every hardware adapter DXGI reports (skips the Microsoft Basic Render Driver)."""
    try:
        dxgi = ctypes.WinDLL("dxgi")
    except OSError:
        return []
    factory = ctypes.c_void_p()
    create = dxgi.CreateDXGIFactory1
    create.restype = ctypes.HRESULT
    create.argtypes = [ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
    try:
        create(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory))
    except OSError:
        return []
    out: list[Gpu] = []
    try:
        i = 0
        while True:
            adapter = ctypes.c_void_p()
            # IDXGIFactory1::EnumAdapters1 is vtable slot 12.
            hr = _vcall(factory, 12, ctypes.c_long, (wintypes.UINT, ctypes.POINTER(ctypes.c_void_p)),
                        i, ctypes.byref(adapter))
            if hr & 0xFFFFFFFF == _DXGI_ERROR_NOT_FOUND or hr < 0:
                break
            try:
                desc = _DXGI_ADAPTER_DESC1()
                # IDXGIAdapter1::GetDesc1 is slot 10.
                if _vcall(adapter, 10, ctypes.c_long, (ctypes.POINTER(_DXGI_ADAPTER_DESC1),), ctypes.byref(desc)) >= 0 \
                        and not desc.Flags & _DXGI_ADAPTER_FLAG_SOFTWARE:
                    out.append(Gpu(desc.Description.strip(), _VENDORS.get(desc.VendorId, f"0x{desc.VendorId:04X}"),
                                   desc.DedicatedVideoMemory))
            finally:
                _vcall(adapter, 2, ctypes.c_ulong)  # Release
            i += 1
    finally:
        _vcall(factory, 2, ctypes.c_ulong)
    return out


def nvidia_smi() -> list[Gpu]:
    exe = shutil.which("nvidia-smi")
    if not exe:
        return []
    try:
        res = subprocess.run([exe, "--query-gpu=name,memory.total,memory.free,compute_cap,driver_version",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=15,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError):
        return []
    if res.returncode != 0:
        return []
    out = []
    for line in res.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5:
            continue
        name, total, free, cap, driver = parts
        try:
            major, _, minor = cap.partition(".")
            drv_major, _, drv_minor = driver.partition(".")
            out.append(Gpu(name, "NVIDIA", int(float(total)) * 2**20, (int(major), int(minor or 0)),
                           int(float(free)) * 2**20, float(f"{drv_major}.{drv_minor.split('.')[0] or 0}")))
        except ValueError:
            continue
    return out


def detect() -> list[Gpu]:
    """Graphics cards, best first (CUDA-capable, then most memory)."""
    gpus = dxgi_adapters()
    smi = nvidia_smi()
    for g in gpus:
        if g.vendor != "NVIDIA":
            continue
        match = next((s for s in smi if s.name.lower() in g.name.lower() or g.name.lower() in s.name.lower()), None)
        if match:
            g.compute_cap, g.free, g.driver = match.compute_cap, match.free, match.driver
            g.vram = g.vram or match.vram
            smi.remove(match)
    gpus += smi  # nvidia-smi saw a card DXGI didn't (rare: headless compute cards)
    return sorted(gpus, key=lambda g: (g.backend != "torch", -g.vram))


# What each power level means. Low keeps the game playable while it runs: small
# tiles, a share of the card's memory, and it rests the GPU half the time.
POWER = {
    "low": {"label": "Low", "hint": "Slowest. Leaves room to play while it works."},
    "balanced": {"label": "Balanced", "hint": "Good speed. The PC stays usable for light tasks."},
    "max": {"label": "Max", "hint": "Fastest. Uses the whole graphics card; don't play at the same time."},
}


def power_settings(level: str, gpu: Gpu | None) -> dict:
    """Upscaler options for ``level`` on ``gpu``. Keys:

    tile        model input tile edge (VRAM per step grows with its square)
    batch       how many tiles' worth of small images go through at once
    mem         share of the card's memory torch may take (CUDA only)
    rest        seconds of GPU rest per second of work (0 = flat out)
    workers     file decode/encode threads
    ncnn_tile / ncnn_jobs   realesrgan-ncnn-vulkan -t and -j values

    Vulkan tiles stay at 256 or auto: at 64 and 128 the anime model's output
    drifted on ~1 in 6 textures (caught by validation, but then plain-resized).
    """
    if level not in POWER:
        raise ValueError(f"unknown power level {level!r}; choose {', '.join(POWER)}")
    gib = (gpu.vram if gpu else 4 * 2**30) / 2**30
    if level == "low":
        return {"tile": 192, "batch": 1, "mem": 0.25, "rest": 1.0, "workers": 2, "priority": "below_normal",
                "ncnn_tile": 256, "ncnn_jobs": "1:1:1"}
    if level == "balanced":
        return {"tile": 384 if gib >= 6 else 256, "batch": 2, "mem": 0.6, "rest": 0.0, "workers": 4,
                "priority": "below_normal", "ncnn_tile": 256, "ncnn_jobs": "1:2:2"}
    return {"tile": 512 if gib >= 10 else 384, "batch": 8 if gib >= 10 else 4, "mem": 0.9, "rest": 0.0,
            "workers": 8, "priority": "normal", "ncnn_tile": 0, "ncnn_jobs": "2:4:4"}


def set_low_priority() -> None:
    """Below-normal CPU priority for this process, so the game wins any contention."""
    k32 = ctypes.WinDLL("kernel32")
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass(k32.GetCurrentProcess(), 0x00004000)  # BELOW_NORMAL_PRIORITY_CLASS


if __name__ == "__main__":
    for g in detect():
        print(g.describe(), g)
