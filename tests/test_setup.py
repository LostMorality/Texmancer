import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from texmancer import bootstrap, hardware
from texmancer.graphics.imageops import keep_binary_alpha, resize, with_lanczos_alpha
from texmancer.graphics.upscale import Project, TexInfo
from texmancer.wad import write_wad


def gpu(vendor="NVIDIA", cap=(12, 0), driver=617.4, vram=16):
    return hardware.Gpu("Test card", vendor, vram * 2**30, cap, None, driver)


@pytest.mark.parametrize("card, build", [
    (gpu(), ("cu130", "2.14.1")),
    (gpu(driver=575.0), ("cu128", "2.11.0")),          # Blackwell on an older driver
    (gpu(cap=(8, 6), driver=565.0), ("cu126", "2.14.1")),
    (gpu(cap=(12, 0), driver=565.0), None),             # cu126 has no Blackwell kernels
    (gpu(cap=(6, 1)), None),                            # Pascal: Vulkan
    (gpu(driver=None), None),
    (gpu("AMD", None, None), None),
])
def test_torch_build(card, build):
    assert card.torch_build == build
    assert card.backend == ("torch" if build else "ncnn")


def test_power_levels_scale_with_the_card():
    low, bal, mx = (hardware.power_settings(k, gpu()) for k in ("low", "balanced", "max"))
    assert low["rest"] > 0 and bal["rest"] == mx["rest"] == 0
    assert low["tile"] < bal["tile"] <= mx["tile"]
    assert low["mem"] < bal["mem"] < mx["mem"]
    small = hardware.power_settings("max", gpu(vram=4))
    assert small["tile"] < mx["tile"] and small["batch"] < mx["batch"]
    with pytest.raises(ValueError):
        hardware.power_settings("turbo", None)


def test_project_max_factor_caps_encode(tmp_path):
    p = Project(tmp_path / "w")
    (tmp_path / "w").mkdir()
    t = TexInfo("Textures/a.dds", "A.wad", 256, 256, 1, "DXT1", "BC1_UNORM")
    assert p._factor(t, 4, 8192) == 4  # no project.json: 4x as before
    (tmp_path / "w" / "project.json").write_text(json.dumps({"max_factor": 2}))
    assert Project(tmp_path / "w")._factor(t, 4, 8192) == 2
    assert Project(tmp_path / "w")._factor(TexInfo("b", "A.wad", 8192, 64, 1, "DXT1", "BC1_UNORM"), 4, 8192) == 1


def test_new_projects_default_to_2x(tmp_path):
    wad = tmp_path / "A.wad"
    write_wad(wad, [("readme.txt", b"no textures")])
    new = Project(tmp_path / "new")
    new.extract([wad])
    assert new.config()["max_factor"] == 2 and new.max_factor == 2
    four = Project(tmp_path / "four")
    four.extract([wad], max_factor=4)
    four.extract([wad])  # re-extracting keeps the choice
    assert four.config()["max_factor"] == 4


def test_install_detection_and_locale(tmp_path):
    inst = tmp_path / "Wizard101"
    assert not bootstrap.is_install(inst)
    (inst / "Data" / "GameData").mkdir(parents=True)
    (inst / "Data" / "GameData" / "Root.wad").write_bytes(b"")
    assert bootstrap.is_install(inst)
    assert bootstrap.game_locale(inst) == "en-US"
    (inst / "Bin").mkdir()
    (inst / "Bin" / "WizardClient.log").write_text("x [STAT] CORE_SEER       LOCALE = de-DE\n")
    assert bootstrap.game_locale(inst) == "de-DE"


def test_base_folder_is_on_the_game_drive(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", r"C:\Users\someone\AppData\Local")
    assert bootstrap.default_base(Path(r"C:\Games\Wizard101")) == Path(r"C:\Users\someone\AppData\Local\Texmancer")
    assert bootstrap.default_base(Path(r"D:\Games\Wizard101")) == Path("D:\\Texmancer")


def test_download_rejects_a_wrong_checksum(monkeypatch, tmp_path):
    src = tmp_path / "file.bin"
    src.write_bytes(b"not what was pinned")
    monkeypatch.setitem(bootstrap.DOWNLOADS, "test", (src.as_uri(), "0" * 64))
    with pytest.raises(bootstrap.SetupError, match="checksum"):
        bootstrap.download("test", tmp_path / "out" / "file.bin")
    assert not (tmp_path / "out" / "file.bin").exists()
    import hashlib
    monkeypatch.setitem(bootstrap.DOWNLOADS, "test", (src.as_uri(), hashlib.sha256(src.read_bytes()).hexdigest()))
    assert bootstrap.download("test", tmp_path / "out" / "file.bin").read_bytes() == src.read_bytes()


def test_image_helpers():
    rgba = np.zeros((8, 8, 4), np.uint8)
    rgba[..., :3] = 200
    rgba[2:6, 2:6, 3] = 255  # 1-bit cut-out
    soft = Image.fromarray(rgba, "RGBA").resize((32, 32), Image.BILINEAR)
    hard = np.asarray(keep_binary_alpha(rgba, soft))[..., 3]
    assert set(np.unique(hard)) <= {0, 255}
    assert resize(rgba, 4, "nearest").size == (32, 32)
    swapped = with_lanczos_alpha(rgba, Image.new("RGBA", (32, 32), (10, 20, 30, 0)))
    arr = np.asarray(swapped)
    assert arr[16, 16, 3] > 200 and tuple(arr[0, 0, :3]) == (10, 20, 30)


def test_pipeline_plan_splits_quick_and_rest(tmp_path, monkeypatch):
    from texmancer import setup_pipeline

    inst = tmp_path / "Wizard101"
    gd = inst / "Data" / "GameData"
    gd.mkdir(parents=True)
    tex = [("Textures/a.dds", b"x" * 100)]
    for name in ("Root.wad", "GUI-WorldData.wad", "WizardCity-WC_Hub.wad", "Krokotopia-KT_Hub.wad",
                 "Mob-WorldData.wad", "Locale_en-US-Root.wad"):
        write_wad(gd / name, tex)
    write_wad(gd / "Sound-WorldData.wad", [("a.mp3", b"x")])
    monkeypatch.setattr(setup_pipeline, "CHUNK_BYTES", 1)  # one archive per chunk
    monkeypatch.setattr(hardware, "detect", lambda: [])
    base = tmp_path / "Texmancer"
    bootstrap.Layout(base).save_state({"install": str(inst), "backend": "ncnn"})
    plan = setup_pipeline.Pipeline(base).plan()
    assert plan[0]["name"] == "quick"
    assert sorted(plan[0]["wads"]) == ["GUI-WorldData.wad", "Root.wad", "WizardCity-WC_Hub.wad"]
    rest = [w for p in plan[1:] for w in p["wads"]]
    assert sorted(rest) == ["Krokotopia-KT_Hub.wad", "Mob-WorldData.wad"]  # no sound, no overlays
    assert all(len(p["wads"]) == 1 for p in plan[1:])
    assert setup_pipeline.Pipeline(base).plan() == plan  # remembered for resume
