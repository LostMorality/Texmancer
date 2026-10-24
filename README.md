# Texmancer

**Makes Wizard101 look sharper by upscaling every texture in the game with AI, right on your own PC.**

Wizard101's textures are small and look blurry on modern screens. Texmancer takes each one, makes it 4 times bigger with an AI upscaler, makes sure it still looks like the original, and puts it back so the game uses it. It does this for everything: worlds, monsters, gear, housing, cards, spell effects and the menus.

Nothing from the game is included here or downloaded. Texmancer only works with the copy of Wizard101 already installed on your computer.

> **Use at your own risk.** Changing game files is against KingsIsle's Terms of Service. Texmancer only changes what textures look like on your screen, but your account is your responsibility.

## What you need

- A Windows PC with an **NVIDIA graphics card**
- **Python 3.11 or newer** ([python.org](https://www.python.org/downloads/))
- **Lots of free disk space.** The full 4x build is around 70 GB, and the working files need more space than that while it builds.
- A few hours. Upscaling the whole game takes a while, even on a fast card.

## Setup (one time)

1. Install the Python packages. The first command installs PyTorch with NVIDIA support:

   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   pip install numpy pillow spandrel
   ```

2. Download **texconv.exe** from [DirectXTex releases](https://github.com/microsoft/DirectXTex/releases) and put it in a folder, for example `C:/tools/texconv`.

3. Download the AI model **RealESRGAN_x4plus_anime_6B.pth** from [Real-ESRGAN releases](https://github.com/xinntao/Real-ESRGAN/releases) and save it as `C:/Users/<you>/models/RealESRGAN_x4plus_anime_6B.pth`.

## Making the mod

Run these from the Texmancer folder, one at a time. Each step can be stopped and started again. It picks up where it left off.

```bash
# 1. Copy every texture out of the game
python -m texmancer extract --work work/everything --all --embedded

# 2. Upscale them (this is the long step)
python -m texmancer upscale --work work/everything --tools C:/tools/texconv

# 3. Pack them into a mod folder
python -m texmancer pack --work work/everything --out mods/HD --tools C:/tools/texconv
```

Low on space? Use `pack --preset balanced` in step 3. It packs everything at 2x instead of 4x, which takes about a quarter of the space.

## Playing with it

```bash
python -m texmancer play --mods mods
```

Leave that running, then open Wizard101 and log in like you normally do. Texmancer waits for the launcher to finish updating, swaps the HD textures in, and puts the original files back when you close the game. Your install is never left changed.

Other commands:

- `python -m texmancer status --mods mods` shows whether the mod is installed.
- `python -m texmancer restore --mods mods` puts the original files back right away.

## After a game update

KingsIsle updates replace some of the game's files. To bring the mod back up to date, add this file at `mods/HD/mod.json`:

```json
{"name": "HD Textures", "build": {"projects": ["work/everything"], "preset": "full"}}
```

then run:

```bash
python -m texmancer build --mod mods/HD
```

Only the parts of the game that changed get upscaled again, so this is much faster than the first build.

## Optional: room for even more 4x textures (off by default)

The game won't read any archive file bigger than 2 GB. Texmancer gets around that by splitting large parts of the game across two files. A few very large parts (monsters, shared world art, gear) are still too big even then, so some of their textures get packed at 2x.

If you want those at 4x too, use `pack --extended` and `play --extended-archives`. This raises the limit to 4 GB. **It works by changing the game's exe file**, not just textures. While you play, Texmancer swaps in a copy of the exe with three size checks switched off, and puts the original back when the game closes. It checks the exe first and won't touch any version it doesn't recognise. Changing the exe carries more risk than changing textures, so only turn this on if you're OK with that.

## How it works (for the curious)

- **Finds every texture.** That means loose image files (DDS, PNG, TGA, JPG, BMP, GIF), texture files, and textures hidden inside 3D model files (Gamebryo NIF, versions 20.2.0.7 to 20.6.0.0), including cube maps. Bump maps hold surface-shape data, not pictures, so they're resized with plain math instead of the AI.
- **Checks every result.** Each upscaled texture is shrunk back down and compared to the original. If it drifted too far, Texmancer tries a safer method, ending with a plain resize. So no texture ever ships looking different from the original art. See-through edges that were sharp stay sharp.
- **Runs fast.** The AI runs on your graphics card in half precision. Textures of the same size are processed together, disk reading and writing happen while the card is busy, and duplicate textures are only upscaled once.
- **Handles the size limit.** The game reads file offsets as signed 32-bit numbers, which caps each archive at 2 GB. When a part outgrows that, the extra textures go into a second file, `Locale_en-US-<part>.wad`, which the game checks first. That file is registered through an `Overrides.xml` inside `Locale_en-US-Root.wad`.

| File | What it does |
|---|---|
| `texmancer/wad.py` | Reads and writes the game's `.wad` archive files |
| `texmancer/nif.py` | Reads and rebuilds textures inside the game's 3D model files |
| `texmancer/graphics/upscale.py` | The main steps: extract, upscale, check, encode, pack |
| `texmancer/graphics/torch_upscale.py` | Runs the AI upscaler on the graphics card |
| `texmancer/graphics/validate.py` | The "does it still look like the original?" check |
| `texmancer/graphics/build.py` | Rebuilds the mod after game updates |
| `texmancer/swap.py` | Swaps the mod in while you play and restores the originals after |

## Credits

[Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (BSD-3-Clause) · [spandrel](https://github.com/chaiNNer-org/spandrel) (MIT) · [DirectXTex / texconv](https://github.com/microsoft/DirectXTex) (MIT) · Gamebryo file layouts from [NifTools nif.xml](https://github.com/niftools/nifxml).

Not affiliated with or endorsed by KingsIsle Entertainment. Wizard101 is a trademark of KingsIsle Entertainment, Inc.

## License

MIT. See [LICENSE](LICENSE).
