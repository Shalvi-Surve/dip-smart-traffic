import numpy as np
from PIL import Image
from analysis.quality import calculate_quality, estimate_noise
from analysis.degradation import assess_degradation


def test_quality_metrics_are_pixel_derived_and_dimensions_reported():
    image = Image.fromarray(np.tile(np.arange(32, dtype=np.uint8), (24, 1)), mode="L")
    result = calculate_quality(image)
    assert result["width"] == 32 and result["height"] == 24
    assert result["mean_brightness"] == 15.5
    assert result["contrast_stddev"] > 0
    assert result["sharpness_laplacian_variance"] > 0
    assert set(result["channel_means_rgb"]) == {"red", "green", "blue"}


def test_noise_estimate_is_finite_for_flat_and_noisy_images():
    assert estimate_noise(Image.new("RGB", (20, 20), "gray")) == 0
    rng = np.random.default_rng(2)
    noisy = Image.fromarray(rng.integers(0, 256, (32, 32), dtype=np.uint8))
    assert np.isfinite(estimate_noise(noisy))


def test_degradation_flags_are_derived_from_image_measurements():
    dark = Image.new("L", (12, 12), 15)
    report = assess_degradation(dark)
    assert "underexposed" in report["flags"]
    assert "low_contrast" in report["flags"]
