import numpy as np
import pytest
from PIL import Image
from dip.adaptive import build_pipeline


def test_dark_low_contrast_input_selects_gamma_and_clahe():
    img = Image.fromarray(np.tile(np.arange(32, dtype=np.uint8), (24, 1)) // 3)
    result = build_pipeline(img)
    names = [s["name"] for s in result["stages"]]
    assert "Gamma correction" in names
    assert "CLAHE on luminance" in names
    assert result["final"].size == img.size
    assert all("image" in s and "before" in s and "after" in s for s in result["stages"])


def test_user_selected_denoising_is_labeled():
    image = Image.fromarray(np.random.default_rng(4).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    result = build_pipeline(image, denoise_choice="median")
    stage = next(s for s in result["stages"] if s["name"] == "Median denoising")
    assert stage["parameters"]["selection"] == "user-selected"


@pytest.mark.parametrize("choice,operation", [
    ("gaussian", "Gaussian denoising"),
    ("bilateral", "Bilateral denoising"),
    ("median", "Median denoising"),
    ("frequency", "FFT Gaussian low-pass"),
])
def test_explicit_denoising_methods_are_selected_and_saved(choice, operation):
    pixels = np.full((24, 32), 90, dtype=np.uint8)
    pixels[::4, ::5] = 0
    pixels[2::5, 3::4] = 255
    image = Image.fromarray(pixels)
    result = build_pipeline(image, denoise_choice=choice)
    stage = next(s for s in result["stages"] if s["name"] == operation)
    assert stage["output_kind"] == "photograph"
    assert stage["parameters"]["selection"] == "user-selected"
    assert not np.array_equal(np.asarray(stage["image"]), np.asarray(stage["input_image"]))


def test_automatic_noise_measurement_selects_actual_frequency_filter():
    image = Image.fromarray(np.random.default_rng(9).integers(0, 256, (32, 32, 3), dtype=np.uint8))
    result = build_pipeline(image)
    stage = next(s for s in result["stages"] if s["name"] == "FFT Gaussian low-pass")
    assert stage["parameters"]["rule"] == "noise estimate >= 8"
    assert stage["image"].size == image.size
