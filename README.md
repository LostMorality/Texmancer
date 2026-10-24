# Texmancer

**Makes Wizard101 look sharper by upscaling the game's textures with AI, right on your own PC.**

## How to use it

1. **Download** `Texmancer-Setup.exe` from the [Releases page](https://github.com/LostMorality/Texmancer/releases/latest).
2. **Run it.** It finds Wizard101 on its own. If it can't, click **Change…** and pick your Wizard101 folder.
3. **Click Start.** The menus and Wizard City are done first, usually within an hour on a recent graphics card. The rest of the game keeps going in the background, and the window shows how long is left. You can close it any time. Next time you open it, it carries on where it stopped.
4. **Click "Play with HD textures"**, then open Wizard101 and log in like you always do. Keep the Texmancer window open while you play. When you close the game, your original files go back.

That's it. Nothing to install by hand.

## Is this safe / allowed?

**It's against KingsIsle's rules.** Their Terms of Use don't allow modifying the game or making derivative works of it, and that includes the images in it. Texmancer does both, on your own PC. Use it at your own risk: your account is your responsibility.

Here is exactly what it does and doesn't do, so you can decide for yourself:

- **It only changes how textures look on your own screen.** Other players see nothing different.
- **It doesn't touch gameplay, the network, KingsIsle's servers, other players or your account.** There's no automation and no advantage of any kind. Everything plays exactly the same.
- **It ships no game files.** Texmancer doesn't include or download any of Wizard101's art, upscaled or not. Everything is made on your PC from the copy of the game you already have.
- **The game is never left modified.** The HD files are swapped in only while you play, and your original files go back as soon as you close the game. If anything goes wrong, **Put original files back** restores them.
- **One optional extra changes the game's exe** ([4 GB archives](#optional-4-gb-archives-off-by-default)). It's off by default and only matters for 4x builds. While you play, it swaps in a copy of the game's exe with three file-size checks switched off, and puts the original back when the game closes. The setup window never turns it on.

Texmancer isn't affiliated with or endorsed by KingsIsle Entertainment.

### What you need

- Windows 10 or 11
- A graphics card from **NVIDIA, AMD or Intel** with up-to-date drivers. Texmancer checks which one you have and sets itself up for it.
- Free disk space on the same drive as Wizard101. For the whole game at 2x, plan on about **110 GB**. Quick only (menus and Wizard City) needs about 12 GB. The window shows what's needed and what's free.
- An internet connection the first time. Setup downloads about 3 GB on NVIDIA cards (PyTorch) or about 50 MB on AMD/Intel cards.

### The choices in the window

- **Power**
  - **Low** is slowest. It rests the graphics card half the time, so you can play while it works.
  - **Balanced** is the default.
  - **Max** uses the whole card. Don't play while it runs.
- **Sharpness**
  - **2x** is the default and looks great on 1080p and 1440p screens.
  - **4x** only helps on 4K screens. It takes about 4 times the disk space and is much slower to build.
- **What to make**
  - **Everything** does the menus and Wizard City first, then the rest of the game.
  - **Quick only** stops after the menus and Wizard City.

### Common questions

- **The game updated. Is my mod broken?** No. Texmancer won't swap in a part of the mod that the update changed, so the game always works. To bring those parts back, rebuild them (see [After a game update](#after-a-game-update)).
- **How do I put everything back?** Close the game. Texmancer restores the originals by itself. If something went wrong, click **Put original files back**.
- **How do I remove Texmancer?** Click **Put original files back**, then delete its folder (shown in the window, usually `%LOCALAPPDATA%\Texmancer`) and `%APPDATA%\Texmancer`.
- **My antivirus complains about the exe.** Tools packed with PyInstaller sometimes get flagged by mistake. You can build the exe yourself from this source (see below), or use the command line instead.

---

## Technical details

Everything below is for people who want to run Texmancer from source or understand how it works.

### What the setup program does

`Texmancer-Setup.exe` is a small PyInstaller program (its window only needs Python's standard library). On first start it:

- finds the install: the default `C:\ProgramData\KingsIsle Entertainment\Wizard101`, then the Windows uninstall entries (KingsIsle and Steam installs register one), then Steam's default library, otherwise it asks;
- detects the graphics card through DXGI (any vendor) and `nvidia-smi` (CUDA compute capability, driver version, free memory);
- downloads only from official sources, each pinned to a SHA-256 where the file is a fixed release:
  - Python 3.11 embeddable from python.org
  - pip from PyPI
  - PyTorch from download.pytorch.org (NVIDIA only; `cu130`, `cu128` or `cu126` depending on the driver)
  - numpy, Pillow and spandrel from PyPI
  - the `RealESRGAN_x4plus_anime_6B` model from the official Real-ESRGAN GitHub release
  - `realesrgan-ncnn-vulkan` from the same release (AMD, Intel and older NVIDIA cards)
  - `texconv.exe` from Microsoft's DirectXTex release
- keeps everything in one folder on the game's drive. The mods have to sit on the same drive, because swaps are instant renames.

It then runs `python -m texmancer.setup_pipeline <folder>` in that runtime, in stages:

1. **Quick:** `GUI-WorldData.wad`, `Root.wad` and every `WizardCity-*.wad` are extracted, upscaled and packed into `mods\HD`, so there's something to play within minutes.
2. **Rest:** every other archive that holds textures (sound and music archives are skipped), in chunks of about 2 GB of game archives. Each chunk's full-size upscaled PNGs are deleted once its final textures are encoded, so they never pile up.
3. **Final pack** of everything into the mod.

Every stage is recorded in `state.json`, and every step underneath resumes from what's on disk, so stopping at any point loses at most the image batch in progress. The time estimate comes from the measured upscaling speed, in source pixels per second.

### Backends

| Card | Backend | Notes |
|---|---|---|
| NVIDIA, Turing (GTX 16 / RTX 20) or newer | PyTorch CUDA, fp16, batched | Fastest |
| AMD, Intel, older NVIDIA | `realesrgan-ncnn-vulkan` (Vulkan) | Same model; in testing it ran at about 80% of CUDA speed |

`torch-directml` was tried for AMD/Intel as well. It produced correct results, but it ran at half the speed of the Vulkan build and needs an old torch. The Vulkan build is also a 50 MB download instead of several hundred MB, so that's what's used.

Power levels map to settings in `texmancer/hardware.py`:

| | Low | Balanced | Max |
|---|---|---|---|
| CUDA tile / batch | 192 px / 1 tile | 384 px / 2 tiles | 512 px / 8 tiles (≥10 GB cards) |
| CUDA memory share | 25% | 60% | 90% |
| Vulkan `-t` / `-j` | 256 / 1:1:1 | 256 / 1:2:2 | auto / 2:4:4 |
| GPU rest | 1 s per 1 s of work | none | none |
| CPU priority | below normal | below normal | normal |

### Running from source

```bash
# NVIDIA: PyTorch with CUDA (cu130 needs driver 580+; cu128 for 570+)
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install numpy pillow spandrel
```

Download **texconv.exe** from [DirectXTex releases](https://github.com/microsoft/DirectXTex/releases) into a tools folder. On AMD or Intel, also put [realesrgan-ncnn-vulkan](https://github.com/xinntao/Real-ESRGAN/releases) there (`numpy pillow` are the only Python packages that path needs). On NVIDIA, save **RealESRGAN_x4plus_anime_6B.pth** from [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) to `%USERPROFILE%\models\`, or pass `--model`.

```bash
# 1. Copy the textures out of the game (new work folders encode at 2x; --max-factor 4 for 4K screens)
python -m texmancer extract --work work/everything --all --embedded

# 2. Upscale them (the long step). The backend is picked for your card; --power low|balanced|max
python -m texmancer upscale --work work/everything --tools C:/tools

# 3. Pack them into a mod folder (balanced = 2x is the default; --preset full for 4x)
python -m texmancer pack --work work/everything --out mods/HD --tools C:/tools

# 4. Play: leave this running, then open Wizard101 and log in
python -m texmancer play --mods mods
```

Each step can be stopped and started again. The AI always upscales 4x, because a 4x result scaled down is sharper than upscaling straight to 2x. `max_factor` in the work folder's `project.json` (set by `extract --max-factor`) caps the scale it's encoded at.

Other commands:

- `status --mods mods` shows what's installed.
- `restore --mods mods` puts the originals back now.
- `build --mod mods/HD` rebuilds after a game update (see below).
- `wad ls|extract` lists or unpacks an archive.

To free disk space after a build, you can delete the `up` folder in your work folder. It holds the full-size upscaled PNGs, usually the biggest part. The finished textures in `dds` are kept, and rebuilds still only redo what changed.

### After a game update

KingsIsle updates replace some of the game's files. The swapper skips any part of a mod whose original changed, so the game never runs on outdated textures. The setup program writes a build recipe into `mods\HD\mod.json`, so from the command line:

```bash
python -m texmancer build --mod mods/HD
```

Only the archives that changed are extracted and upscaled again. A hand-made mod needs its own `mod.json` with `{"label": "HD Textures", "build": {"projects": ["work/everything"], "preset": "balanced"}}`.

### Optional: 4 GB archives (off by default)

The game won't read any archive file bigger than 2 GB. Texmancer gets around that by splitting large parts of the game across two files. At 4x, a few very large parts (monsters, shared world art, gear) are still too big even then, so some of their textures get packed at 2x.

If you want those at 4x too, use `pack --extended` and `play --extended-archives`. This raises the limit to 4 GB. **It works by changing the game's exe file**, not just textures. While you play, Texmancer swaps in a copy of the exe with three size checks switched off, and puts the original back when the game closes. It checks the exe first and won't touch any version it doesn't recognise. Changing the exe carries more risk than changing textures, so only turn this on if you're OK with that.

### How it works

- **Finds every texture.** That covers loose images (DDS, PNG, TGA, JPG, BMP, GIF), texture files, and textures inside 3D model files (Gamebryo NIF, versions 20.2.0.7 to 20.6.0.0), including cube maps. Bump maps hold surface-shape data, not pictures, so they're resized with plain math instead of the AI.
- **Checks every result.** Each upscaled texture is shrunk back down and compared to the original. If it drifted too far, Texmancer tries a safer method, ending with a plain resize, so no texture ships looking different from the original art. See-through edges that were sharp stay sharp.
- **Runs fast.** Textures of the same size go through the AI together, disk reads and writes overlap with the GPU work, and duplicates are only upscaled once.
- **Handles the size limit.** The game reads file offsets as signed 32-bit numbers, which caps each archive at 2 GB. When a part outgrows that, the extra textures go into a second file, `Locale_en-US-<part>.wad`, which the game checks first. That file is registered through an `Overrides.xml` inside `Locale_en-US-Root.wad`.
- **Swaps safely.** The launcher re-downloads any archive that doesn't match its records, and it only checks after login. So the mod is swapped in after the launcher logs "Game patched successful" and swapped out when the game exits. Both are same-drive renames, so the originals keep their exact size and date.

| File | What it does |
|---|---|
| `installer/setup_app.py` | The setup window (`Texmancer-Setup.exe`) |
| `installer/build_exe.py` | Builds the exe and checks it for personal info |
| `texmancer/bootstrap.py` | Finds the game and downloads the runtime, tools and model |
| `texmancer/hardware.py` | Graphics card detection and the power levels |
| `texmancer/setup_pipeline.py` | The staged quick-then-everything build |
| `texmancer/wad.py` | Reads and writes the game's `.wad` archives |
| `texmancer/nif.py` | Reads and rebuilds textures inside the game's 3D model files |
| `texmancer/graphics/upscale.py` | The main steps: extract, upscale, check, encode, pack |
| `texmancer/graphics/torch_upscale.py` | The AI upscaler on CUDA |
| `texmancer/graphics/ncnn_upscale.py` | The AI upscaler on Vulkan |
| `texmancer/graphics/validate.py` | The "does it still look like the original?" check |
| `texmancer/graphics/build.py` | Rebuilds the mod after game updates |
| `texmancer/swap.py` | Swaps the mod in while you play and restores the originals after |

### Building the exe

```bash
pip install pyinstaller
python installer/build_exe.py
```

The result is `dist/Texmancer-Setup.exe`; `--out DIR` puts `build/` and `dist/` somewhere else. The build script unpacks every file inside the exe and fails if it finds the building PC's user name, home folder or computer name.

## Credits

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause) · [spandrel](https://github.com/chaiNNer-org/spandrel) (MIT) · [DirectXTex / texconv](https://github.com/microsoft/DirectXTex) (MIT) · [PyTorch](https://pytorch.org) (BSD-3-Clause) · Gamebryo file layouts from [NifTools nif.xml](https://github.com/niftools/nifxml).

Not affiliated with or endorsed by KingsIsle Entertainment. Wizard101 is a trademark of KingsIsle Entertainment, Inc.

## License

MIT. See [LICENSE](LICENSE).
