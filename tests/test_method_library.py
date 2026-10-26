import numpy as np
import pytest
from PIL import Image

from dip.frequency import gaussian_lowpass_luminance, spectrum_image
from dip.dehaze import dark_channel_prior
from dip.spatial import (bilateral_denoise, clahe_luminance, edge_enhance,
                         edge_map, equalize_luminance, gaussian_denoise,
                         gamma_correct, median_denoise, morphology,
                         normalize_luminance, unsharp_luminance)


def structured_scene(mode="RGB"):
    y, x = np.mgrid[:40, :48]
    gray = np.where(x < 24, 70, 105).astype(np.uint8)
    if mode == "L":
        return Image.fromarray(gray)
    rgb = np.stack([gray, np.roll(gray, 2, axis=1), gray], axis=2)
    image = Image.fromarray(rgb, "RGB")
    if mode == "RGBA":
        image.putalpha(Image.new("L", image.size, 128))
    return image


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_normalization_gamma_global_equalization_and_clahe_return_real_color_images(mode):
    image = structured_scene(mode)
    results = [normalize_luminance(image), gamma_correct(image, .75),
               equalize_luminance(image), clahe_luminance(image, tile_size=16)]
    for output in results:
        assert output.size == image.size and output.mode == "RGB"
    assert not np.array_equal(np.asarray(results[0]), np.asarray(image.convert("RGB")))
    assert not np.array_equal(np.asarray(results[1]), np.asarray(image.convert("RGB")))
    assert not np.array_equal(np.asarray(results[2]), np.asarray(image.convert("RGB")))
    assert not np.array_equal(np.asarray(results[3]), np.asarray(image.convert("RGB")))


@pytest.mark.parametrize("function", [median_denoise, gaussian_denoise, bilateral_denoise,
                                       unsharp_luminance, edge_enhance, gaussian_lowpass_luminance])
def test_spatial_frequency_filters_preserve_geometry_and_compute_output(function):
    rng = np.random.default_rng(12)
    base = np.tile(np.where(np.arange(48) < 24, 70, 150), (40, 1)).astype(np.int16)
    noisy = np.clip(base + rng.integers(-20, 21, base.shape), 0, 255).astype(np.uint8)
    source = Image.fromarray(np.repeat(noisy[:, :, None], 3, axis=2))
    output = function(source)
    assert output.size == source.size and output.mode == "RGB"
    assert not np.array_equal(np.asarray(output), np.asarray(source))


@pytest.mark.parametrize("operator", ["sobel", "prewitt"])
def test_edge_maps_are_actual_labeled_diagnostics_and_edge_enhancement_is_color(operator):
    image = structured_scene()
    edges = edge_map(image, operator)
    enhanced = edge_enhance(image, operator=operator)
    assert edges.mode == "L" and edges.size == image.size and np.asarray(edges).max() > 0
    assert enhanced.mode == "RGB" and enhanced.size == image.size


@pytest.mark.parametrize("operation", ["erosion", "dilation", "opening", "closing"])
def test_morphology_operations_produce_distinct_auxiliary_grayscale_output(operation):
    values = np.zeros((25, 25), dtype=np.uint8)
    values[6:19, 6:19] = 200
    values[12, 12] = 255
    values[9, 9] = 0
    image = Image.fromarray(values)
    output = morphology(image, operation, size=3)
    assert output.mode == "L" and output.size == image.size
    assert not np.array_equal(np.asarray(output), values)


def test_fft_spectrum_is_a_real_diagnostic():
    image = structured_scene()
    spectrum = spectrum_image(image)
    assert spectrum.mode == "L" and spectrum.size == image.size
    assert np.asarray(spectrum).max() > np.asarray(spectrum).min()


def test_filter_parameter_validation():
    image = structured_scene()
    with pytest.raises(ValueError):
        bilateral_denoise(image, diameter=4)
    with pytest.raises(ValueError):
        morphology(image, "closing", size=2)
    with pytest.raises(ValueError):
        edge_map(image, "canny")


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_all_method_families_accept_grayscale_rgb_and_rgba(mode):
    source = structured_scene(mode).resize((16, 16))
    outputs = [normalize_luminance(source), gamma_correct(source, .9),
               equalize_luminance(source), clahe_luminance(source, tile_size=8),
               median_denoise(source), gaussian_denoise(source), bilateral_denoise(source),
               dark_channel_prior(source, window=3)[0], unsharp_luminance(source),
               edge_map(source, "prewitt"), edge_enhance(source, operator="sobel"),
               morphology(source, "opening"), gaussian_lowpass_luminance(source)]
    assert all(output.size == source.size for output in outputs)
    assert all(output.mode in {"RGB", "L"} for output in outputs)
