import numpy as np
import pytest
from PIL import Image

from texmancer.graphics.validate import check_upscale


def _src():
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 255, (32, 32, 3), dtype=np.uint8)
    a = np.zeros((32, 32), np.uint8)
    a[4:28, 4:28] = 255
    return Image.fromarray(np.dstack([rgb, a]), "RGBA")


def test_faithful_upscale_passes():
    src = _src()
    assert check_upscale(src, src.resize((128, 128), Image.NEAREST), 4) is None


def test_blank_output_fails():
    src = _src()
    assert check_upscale(src, Image.new("RGBA", (128, 128)), 4) == "blank output"


def test_wrong_size_fails():
    src = _src()
    assert "size" in check_upscale(src, src.resize((64, 64)), 4)


def test_lost_alpha_fails():
    src = _src()
    out = src.resize((128, 128), Image.NEAREST)
    out.putalpha(0)
    assert "alpha" in check_upscale(src, out, 4)


def test_garbled_color_fails():
    src = _src()
    out = src.resize((128, 128), Image.NEAREST)
    arr = np.asarray(out).copy()
    arr[..., :3] = 255 - arr[..., :3]
    assert "color" in check_upscale(src, Image.fromarray(arr, "RGBA"), 4)


torch = pytest.importorskip("torch")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_tiled_matches_untiled():
    from texmancer.graphics.torch_upscale import DEFAULT_MODEL, TorchUpscaler

    if not DEFAULT_MODEL.exists():
        pytest.skip("model not downloaded")
    # Smooth art-like input (gradients + a disc with soft alpha); noise would be
    # legitimately smoothed away by the model.
    yy, xx = np.mgrid[0:96, 0:96]
    rgb = np.dstack([xx * 2.5, yy * 2.5, 255 - xx * 2]).clip(0, 255).astype(np.uint8)
    a = (255 * ((xx - 48) ** 2 + (yy - 48) ** 2 < 40 ** 2)).astype(np.uint8)
    img = Image.fromarray(np.dstack([rgb, a]), "RGBA")
    whole = TorchUpscaler(tile=4096).upscale(img)
    tiled = TorchUpscaler(tile=40, pad=16).upscale(img)
    diff = np.abs(np.asarray(whole, np.int16) - np.asarray(tiled, np.int16)).mean()
    assert diff < 2.0
    assert check_upscale(img, tiled, 4) is None
