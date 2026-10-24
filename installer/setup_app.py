"""Texmancer-Setup.exe: one window that sets everything up, builds the HD mod
from the player's own Wizard101 install, and runs the swapper while they play.

The window itself only needs the standard library. Everything heavy runs in
a Python runtime it downloads from python.org into the Texmancer folder.
"""

from __future__ import annotations

import ctypes
import json
import os
import queue
import subprocess
import sys
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from texmancer import __version__, bootstrap, hardware

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
SETTINGS = Path(os.environ.get("APPDATA") or Path.home()) / "Texmancer" / "setup.json"

# Disk use, GB: runtime + tools by backend; work files + mod by sharpness. Measured
# on the quick stage (0.93 GB of game archives -> 7.3 GB at 2x) and scaled to the
# 11.2 GB of archives that hold textures, plus room for one chunk's upscales.
RUNTIME_GB = {"torch": 4, "ncnn": 1}
NEED_GB = {2: {"quick": 8, "full": 105}, 4: {"quick": 22, "full": 270}}

ACCENT, ACCENT_DARK, BG, MUTED = "#6b3fd4", "#4f2ba3", "#f7f5fb", "#5c5670"


def package_dir() -> Path:
    """The texmancer package to deploy: bundled in the exe, or next to this file."""
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / "texmancer_src"  # type: ignore[attr-defined]
    return Path(__file__).resolve().parent.parent / "texmancer"


def fmt_eta(seconds: float) -> str:
    if seconds < 90:
        return "about a minute left"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"about {minutes} minutes left"
    h, m = divmod(minutes, 60)
    return f"about {h} h {m:02d} min left" if h < 24 else f"about {h / 24:.1f} days left"


def kill_tree(proc: subprocess.Popen | None) -> None:
    if proc and proc.poll() is None:
        subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"], capture_output=True, creationflags=_NO_WINDOW)


class App:
    def __init__(self, root: tk.Tk, base: Path | None = None):
        """``base`` (the --base option) pins the Texmancer folder and keeps the
        window's settings inside it, e.g. for a test build in a scratch folder."""
        self.root = root
        self.settings_file = base / "setup.json" if base else SETTINGS
        self.events: queue.Queue = queue.Queue()
        self.worker: subprocess.Popen | None = None
        self.player: subprocess.Popen | None = None
        self.running = False
        self.gpus: list[hardware.Gpu] = []
        self.settings = self._load_settings()
        self.install = self._initial_install()
        self.base = base or (Path(self.settings["base"]) if self.settings.get("base") else
                             (bootstrap.default_base(self.install) if self.install else None))

        root.title(f"Texmancer {__version__}")
        root.configure(bg=BG)
        root.minsize(620, 560)
        root.protocol("WM_DELETE_WINDOW", self.on_close)
        self.power = tk.StringVar(value=self.settings.get("power", "balanced"))
        self.quality = tk.IntVar(value=int(self.settings.get("quality", 2)))
        self.mode = tk.StringVar(value=self.settings.get("mode", "all"))
        self._build()
        threading.Thread(target=self._detect, daemon=True).start()
        self._refresh()
        # Enter starts (or continues), like the big button.
        self.start_btn.focus_set()
        root.bind("<Return>", lambda _e: None if self.running or str(self.start_btn["state"]) == "disabled"
                  else self.on_start())
        root.after(100, self._pump)

    # -- settings -------------------------------------------------------------

    def _load_settings(self) -> dict:
        try:
            return json.loads(self.settings_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_settings(self) -> None:
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self.settings.update(install=str(self.install) if self.install else None,
                             base=str(self.base) if self.base else None, power=self.power.get(),
                             quality=self.quality.get(), mode=self.mode.get())
        self.settings_file.write_text(json.dumps(self.settings, indent=1), encoding="utf-8")

    def _initial_install(self) -> Path | None:
        saved = self.settings.get("install")
        if saved and bootstrap.is_install(Path(saved)):
            return Path(saved)
        return bootstrap.find_install()

    @property
    def layout(self) -> bootstrap.Layout | None:
        return bootstrap.Layout(self.base) if self.base else None

    # -- window ---------------------------------------------------------------

    def _build(self) -> None:
        r = self.root
        style = ttk.Style(r)
        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG)
        style.configure("TRadiobutton", background=BG)
        style.configure("Muted.TLabel", background=BG, foreground=MUTED)
        style.configure("Head.TLabel", background=BG, font=("Segoe UI", 20, "bold"))
        style.configure("Sub.TLabel", background=BG, foreground=MUTED, font=("Segoe UI", 10))
        style.configure("Sect.TLabel", background=BG, font=("Segoe UI", 10, "bold"))
        style.configure("Big.Horizontal.TProgressbar", thickness=18)

        outer = ttk.Frame(r, padding=(22, 16))
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Texmancer", style="Head.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Sharper textures for Wizard101, made on your own PC from your own game files.",
                  style="Sub.TLabel").pack(anchor="w", pady=(0, 10))

        grid = ttk.Frame(outer)
        grid.pack(fill="x")
        grid.columnconfigure(1, weight=1)

        ttk.Label(grid, text="Wizard101", style="Sect.TLabel").grid(row=0, column=0, sticky="nw", pady=4, padx=(0, 14))
        f = ttk.Frame(grid)
        f.grid(row=0, column=1, sticky="ew")
        self.install_lbl = ttk.Label(f, wraplength=420)
        self.install_lbl.pack(side="left", fill="x", expand=True)
        self.install_btn = ttk.Button(f, text="Change…", command=self.pick_install)
        self.install_btn.pack(side="right")

        ttk.Label(grid, text="Graphics card", style="Sect.TLabel").grid(row=1, column=0, sticky="nw", pady=4)
        self.gpu_lbl = ttk.Label(grid, text="Checking…", wraplength=460)
        self.gpu_lbl.grid(row=1, column=1, sticky="w", pady=4)

        ttk.Label(grid, text="Power", style="Sect.TLabel").grid(row=2, column=0, sticky="nw", pady=4)
        f = ttk.Frame(grid)
        f.grid(row=2, column=1, sticky="w", pady=4)
        self.option_widgets: list[ttk.Radiobutton] = []
        for key, info in hardware.POWER.items():
            rb = ttk.Radiobutton(f, text=f"{info['label']}: {info['hint']}", value=key, variable=self.power,
                                 command=self._refresh)
            rb.pack(anchor="w")
            self.option_widgets.append(rb)

        ttk.Label(grid, text="Sharpness", style="Sect.TLabel").grid(row=3, column=0, sticky="nw", pady=4)
        f = ttk.Frame(grid)
        f.grid(row=3, column=1, sticky="w", pady=4)
        self.quality_widgets = [
            ttk.Radiobutton(f, text="2x (recommended for 1080p and 1440p screens)", value=2, variable=self.quality,
                            command=self._refresh),
            ttk.Radiobutton(f, text="4x (only helps on 4K screens; about 4 times the disk space)", value=4,
                            variable=self.quality, command=self._refresh)]
        for rb in self.quality_widgets:
            rb.pack(anchor="w")

        ttk.Label(grid, text="What to make", style="Sect.TLabel").grid(row=4, column=0, sticky="nw", pady=4)
        f = ttk.Frame(grid)
        f.grid(row=4, column=1, sticky="w", pady=4)
        self.mode_widgets = [
            ttk.Radiobutton(f, text="Everything: the menus and Wizard City first (playable sooner), then the rest",
                            value="all", variable=self.mode, command=self._refresh),
            ttk.Radiobutton(f, text="Quick only: just the menus and Wizard City", value="quick",
                            variable=self.mode, command=self._refresh)]
        for rb in self.mode_widgets:
            rb.pack(anchor="w")

        f = ttk.Frame(outer)
        f.pack(fill="x", pady=(8, 6))
        self.disk_lbl = ttk.Label(f, style="Muted.TLabel", wraplength=470)
        self.disk_lbl.pack(side="left", fill="x", expand=True)
        self.base_btn = ttk.Button(f, text="Change…", command=self.pick_base)
        self.base_btn.pack(side="right")

        self.start_btn = tk.Button(outer, text="Start", font=("Segoe UI", 16, "bold"), fg="white", bg=ACCENT,
                                   activebackground=ACCENT_DARK, activeforeground="white", relief="flat",
                                   cursor="hand2", padx=30, pady=8, command=self.on_start)
        self.start_btn.pack(fill="x", pady=(4, 10))

        self.stage_lbl = ttk.Label(outer, text="", font=("Segoe UI", 10, "bold"), wraplength=560)
        self.stage_lbl.pack(anchor="w")
        self.bar = ttk.Progressbar(outer, style="Big.Horizontal.TProgressbar", maximum=1000)
        self.bar.pack(fill="x", pady=4)
        self.eta_lbl = ttk.Label(outer, text="", style="Muted.TLabel")
        self.eta_lbl.pack(anchor="w")

        row = ttk.Frame(outer)
        row.pack(fill="x", pady=(10, 0))
        self.play_btn = ttk.Button(row, text="Play with HD textures", command=self.on_play)
        self.play_btn.pack(side="left")
        self.restore_btn = ttk.Button(row, text="Put original files back", command=self.on_restore)
        self.restore_btn.pack(side="left", padx=8)
        self.log_btn = ttk.Button(row, text="Show details", command=self.toggle_log)
        self.log_btn.pack(side="right")

        self.log_frame = ttk.Frame(outer)
        self.log_text = tk.Text(self.log_frame, height=12, wrap="word", font=("Consolas", 9), relief="flat")
        sb = ttk.Scrollbar(self.log_frame, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=sb.set, state="disabled")
        self.log_text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        ttk.Label(outer, text="Changing game files is against KingsIsle's Terms of Service; use at your own risk. "
                              "No game files are downloaded or shared: everything is made from your install.",
                  style="Muted.TLabel", wraplength=560, font=("Segoe UI", 8)).pack(side="bottom", anchor="w",
                                                                                  pady=(10, 0))

    def toggle_log(self) -> None:
        if self.log_frame.winfo_ismapped():
            self.log_frame.pack_forget()
            self.log_btn.configure(text="Show details")
        else:
            self.log_frame.pack(fill="both", expand=True, pady=(8, 0))
            self.log_btn.configure(text="Hide details")

    def log(self, text: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", text + "\n")
        lines = int(self.log_text.index("end-1c").split(".")[0])
        if lines > 3000:
            self.log_text.delete("1.0", f"{lines - 3000}.0")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        lay = self.layout
        if lay:
            try:
                lay.logs.mkdir(parents=True, exist_ok=True)
                with open(lay.logs / "setup.log", "a", encoding="utf-8") as f:
                    f.write(time.strftime("%Y-%m-%d %H:%M:%S ") + text + "\n")
            except OSError:
                pass

    # -- state ----------------------------------------------------------------

    def _detect(self) -> None:
        try:
            gpus = hardware.detect()
        except Exception as exc:  # detection must never stop the window from opening
            gpus = []
            self.events.put(("log", f"graphics card check failed: {exc!r}"))
        self.events.put(("gpus", gpus))

    @property
    def gpu(self) -> hardware.Gpu | None:
        return self.gpus[0] if self.gpus else None

    def started_state(self) -> dict:
        lay = self.layout
        return lay.load_state() if lay else {}

    def _refresh(self) -> None:
        state = self.started_state()
        begun = bool(state.get("done"))
        if begun and "quality" in state:  # the work so far was made at this sharpness
            self.quality.set(int(state["quality"]))
        self.install_lbl.configure(text=str(self.install) if self.install else
                                   "Not found. Click Change… and pick your Wizard101 folder.")
        for w in self.option_widgets:
            w.configure(state="disabled" if self.running else "normal")
        for w in self.quality_widgets:
            w.configure(state="disabled" if self.running or begun else "normal")
        for w in self.mode_widgets:
            w.configure(state="disabled" if self.running else "normal")
        self.install_btn.configure(state="disabled" if self.running or begun else "normal")
        self.base_btn.configure(state="disabled" if self.running or begun else "normal")
        mod_ready = bool(self.layout and any(self.layout.mod.glob("*.wad")))
        swapped = bool(self.layout and (self.layout.mods / "swap_state.json").exists())
        self.play_btn.configure(state="normal" if mod_ready and not self.player_running() else "disabled")
        self.restore_btn.configure(state="normal" if swapped and not self.player_running() else "disabled")
        if self.running:
            self.start_btn.configure(text="Stop", bg="#9a3a52", activebackground="#7a2a40")
        else:
            finished = "all:pack" in state.get("done", []) or (self.mode.get() == "quick"
                                                               and "quick:pack" in state.get("done", []))
            text = "Done" if finished else ("Continue" if begun else "Start")
            self.start_btn.configure(text=text, bg=ACCENT, activebackground=ACCENT_DARK,
                                     state="disabled" if finished or not self.install or not self.gpus else "normal")
        if self.base:
            try:
                free = bootstrap.free_bytes(self.base) / 2**30
            except OSError:
                free = 0
            need = NEED_GB[self.quality.get()]["quick" if self.mode.get() == "quick" else "full"]
            need += RUNTIME_GB[self.gpu.backend if self.gpu else "torch"]
            warn = "  Not enough space!" if free < need and not begun else ""
            self.disk_lbl.configure(text=f"Files go in {self.base}. Free space there: {free:,.0f} GB; "
                                         f"this needs about {need} GB.{warn}")

    def player_running(self) -> bool:
        return bool(self.player and self.player.poll() is None)

    # -- actions --------------------------------------------------------------

    def pick_install(self) -> None:
        d = filedialog.askdirectory(title="Pick your Wizard101 folder (the one with Wizard101.exe)")
        if not d:
            return
        p = Path(d)
        for cand in (p, p.parent, p.parent.parent):  # they may pick Data or GameData
            if bootstrap.is_install(cand):
                self.install = cand
                self.base = bootstrap.default_base(cand)
                self._save_settings()
                self._refresh()
                return
        messagebox.showerror("Texmancer", "That folder doesn't look like Wizard101 "
                                          "(it needs Data\\GameData\\Root.wad inside).")

    def pick_base(self) -> None:
        if not self.install:
            return
        d = filedialog.askdirectory(title="Pick where Texmancer keeps its files")
        if not d:
            return
        p = Path(d)
        drive = os.path.splitdrive(str(self.install))[0]
        if os.path.splitdrive(str(p))[0].lower() != drive.lower():
            messagebox.showerror("Texmancer", f"Pick a folder on the same drive as Wizard101 ({drive}). "
                                              "Swapping the textures in and out is instant only on the same drive.")
            return
        self.base = p if p.name.lower() == "texmancer" else p / "Texmancer"
        self._save_settings()
        self._refresh()

    def on_start(self) -> None:
        if self.running:
            if messagebox.askyesno("Texmancer", "Stop now? Next time you open Texmancer it carries on "
                                                "where it stopped."):
                self.stop_worker()
            return
        if not self.install or not self.gpu:
            return
        self._save_settings()
        lay = self.layout
        state = lay.load_state()
        state.update(install=str(self.install), power=self.power.get(), backend=self.gpu.backend,
                     quick_only=self.mode.get() == "quick", locale=bootstrap.game_locale(self.install))
        state.setdefault("quality", self.quality.get())
        lay.save_state(state)
        self.running = True
        self._refresh()
        self.stage_lbl.configure(text="Getting ready")
        threading.Thread(target=self._run, args=(lay,), daemon=True).start()

    def _run(self, lay: bootstrap.Layout) -> None:
        put = self.events.put
        try:
            bootstrap.deploy_app(lay, package_dir())
            gpu = self.gpu
            bootstrap.ensure_runtime(
                lay, gpu.backend, gpu.torch_build, log=lambda m: put(("stage", m)),
                progress=lambda d, t: put(("progress", {"done": d, "total": t})))
            cmd = [str(lay.python), "-u", "-m", "texmancer.setup_pipeline", str(lay.base)]
            self.worker = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                           encoding="utf-8", errors="replace", creationflags=_NO_WINDOW,
                                           cwd=str(lay.base))
            for line in self.worker.stdout:  # type: ignore[union-attr]
                line = line.rstrip("\n")
                if line.startswith("@@"):
                    try:
                        ev = json.loads(line[2:])
                    except ValueError:
                        continue
                    put((ev.pop("type"), ev))
                elif line:
                    put(("log", line))
            code = self.worker.wait()
            put(("exit", code))
        except Exception as exc:
            put(("error", {"text": str(exc)}))
            put(("exit", 1))

    def stop_worker(self) -> None:
        kill_tree(self.worker)

    def _pump(self) -> None:
        try:
            while True:
                kind, data = self.events.get_nowait()
                self._handle(kind, data)
        except queue.Empty:
            pass
        self.root.after(100, self._pump)

    def _handle(self, kind: str, data) -> None:
        if kind == "gpus":
            self.gpus = data
            if self.gpu:
                extra = "" if self.gpu.backend == "torch" else " (NVIDIA CUDA isn't available for this card)" \
                    if self.gpu.vendor == "NVIDIA" else ""
                self.gpu_lbl.configure(text=self.gpu.describe() + extra)
            else:
                self.gpu_lbl.configure(text="No supported graphics card found. Texmancer needs an NVIDIA, AMD "
                                            "or Intel graphics card with up-to-date drivers.")
            self._refresh()
        elif kind == "log":
            self.log(data)
        elif kind in ("stage", "status"):
            text = data if isinstance(data, str) else data.get("text", "")
            self.stage_lbl.configure(text=text)
            self.log(text)
        elif kind == "progress":
            total, done = data.get("total") or 0, data.get("done") or 0
            if data.get("stage"):
                self.stage_lbl.configure(text=data["stage"])
            if total:
                self.bar.configure(mode="determinate")
                self.bar.stop()
                self.bar["value"] = 1000 * done / total
            else:
                self.bar.configure(mode="indeterminate")
                self.bar.start(15)
            self.eta_lbl.configure(text=fmt_eta(data["eta"]) if data.get("eta") is not None else "")
        elif kind == "quick_ready":
            self.log("The menus and Wizard City are ready to play.")
            self._refresh()
            if self.mode.get() == "all" and not self.player_running():
                self.eta_lbl.configure(text="The menus and Wizard City are ready. You can play now "
                                            "(Low power works best while playing).")
        elif kind == "done":
            self.bar.stop()
            self.bar.configure(mode="determinate")
            self.bar["value"] = 1000
            self.stage_lbl.configure(text="All done!")
            self.eta_lbl.configure(text="")
        elif kind == "error":
            self.log("ERROR: " + data.get("text", ""))
            messagebox.showerror("Texmancer", f"Something went wrong:\n\n{data.get('text', '')}\n\n"
                                              "Click Continue to try again. The details are in the log "
                                              f"({self.layout.logs / 'setup.log' if self.layout else ''}).")
        elif kind == "exit":
            self.running = False
            self.worker = None
            self.bar.stop()
            state = self.started_state()
            self._refresh()
            finished = "all:pack" in state.get("done", []) or (state.get("quick_only")
                                                               and "quick:pack" in state.get("done", []))
            if data == 0 and finished and not self.player_running():
                if messagebox.askyesno("Texmancer", "Your HD textures are ready.\n\nStart playing with them now?"):
                    self.on_play()
            elif data != 0 and not finished:
                self.stage_lbl.configure(text="Stopped. Click Continue to carry on.")
                self.eta_lbl.configure(text="")
        elif kind == "player_exit":
            self.log("Stopped swapping; the original game files are back.")
            self._refresh()

    def on_play(self) -> None:
        lay = self.layout
        if not lay or self.player_running():
            return
        cmd = [str(lay.python), "-u", "-m", "texmancer", "--game-data", str(self.install / "Data" / "GameData"),
               "play", "--mods", str(lay.mods)]
        self.player = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding="utf-8", errors="replace", creationflags=_NO_WINDOW)
        self._refresh()
        messagebox.showinfo("Texmancer", "Now open Wizard101 and log in like you always do.\n\n"
                                         "Texmancer swaps the HD textures in after the launcher finishes "
                                         "updating, and puts the original files back when you close the game.\n\n"
                                         "Keep this window open while you play.")

        def pipe() -> None:
            for line in self.player.stdout:  # type: ignore[union-attr]
                if line.strip():
                    self.events.put(("log", "[play] " + line.rstrip()))
            self.player.wait()
            self.events.put(("player_exit", None))
        threading.Thread(target=pipe, daemon=True).start()

    def on_restore(self) -> None:
        lay = self.layout
        if not lay:
            return
        cmd = [str(lay.python), "-m", "texmancer", "--game-data", str(self.install / "Data" / "GameData"),
               "restore", "--mods", str(lay.mods)]
        res = subprocess.run(cmd, capture_output=True, text=True, creationflags=_NO_WINDOW)
        self.log((res.stdout + res.stderr).strip() or "original files restored")
        self._refresh()

    def on_close(self) -> None:
        if self.player_running():
            if game_running():
                messagebox.showwarning("Texmancer", "Close Wizard101 first, so Texmancer can put the original "
                                                    "game files back.")
                return
            kill_tree(self.player)
            self.on_restore()
        if self.running:
            if not messagebox.askyesno("Texmancer", "Texmancer is still working. Stop and close?\n\n"
                                                    "Next time you open it, it carries on where it stopped."):
                return
            self.stop_worker()
        self.root.destroy()


def game_running() -> bool:
    out = subprocess.run(["tasklist", "/FI", "IMAGENAME eq WizardGraphicalClient.exe", "/NH"], capture_output=True,
                         text=True, creationflags=_NO_WINDOW).stdout
    return "WizardGraphicalClient.exe" in out


def main() -> None:
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        pass
    import argparse
    ap = argparse.ArgumentParser(prog="Texmancer-Setup")
    ap.add_argument("--base", help="Texmancer folder to use (settings are kept inside it too)")
    args = ap.parse_args()
    root = tk.Tk()
    App(root, Path(args.base).resolve() if args.base else None)
    root.mainloop()


if __name__ == "__main__":
    main()
