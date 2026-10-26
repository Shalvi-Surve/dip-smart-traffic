import numpy as np
import pytest
from PIL import Image, ImageOps
from dip.spatial import clahe_luminance, equalize_luminance, gamma_correct, median_denoise, unsharp_luminance


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_spatial_ops_support_grayscale_rgb_and_rgba(mode):
    image = Image.new(mode, (24, 20), 90 if mode == "L" else (90, 100, 110, 80) if mode == "RGBA" else (90, 100, 110))
    for result in (gamma_correct(image, 0.8), clahe_luminance(image, tile_size=16),
                   median_denoise(image), unsharp_luminance(image)):
        assert result.mode == "RGB" and result.size == image.size


def test_gamma_and_invalid_parameter():
    img = Image.fromarray(np.full((8, 8), 32, np.uint8), mode="L")
    assert gamma_correct(img, .7).convert("L").getpixel((0, 0)) > 32
    with pytest.raises(ValueError):
        gamma_correct(img, 0)


def test_global_histogram_equalization_is_applied_to_luminance_only():
    ramp = np.full((16, 16), 80, dtype=np.uint8)
    ramp.flat[-16:] = 100
    image = Image.fromarray(ramp, mode="L")
    result = equalize_luminance(image)
    assert result.size == image.size and result.mode == "RGB"
    y, cb, cr = image.convert("RGB").convert("YCbCr").split()
    expected = Image.merge("YCbCr", (ImageOps.equalize(y), cb, cr)).convert("RGB")
    assert np.array_equal(np.asarray(result), np.asarray(expected))
