"""Pixel-derived quality and degradation indicators.

These measurements describe image statistics; none is a perceptual-quality
score or a definitive weather classifier.
"""
import numpy as np
from PIL import Image, ImageFilter


def _gray_array(image):
    return np.asarray(image.convert("L"), dtype=np.float32)


def _local_contrast(gray, tile=64):
    values = []
    for y in range(0, gray.shape[0], tile):
        for x in range(0, gray.shape[1], tile):
            patch = gray[y:y + tile, x:x + tile]
            if patch.size:
                values.append(float(patch.std()))
    return float(np.mean(values)) if values else 0.0


def _edge_stats(gray):
    p = np.pad(gray, 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]
          - p[:-2, :-2] - 2 * p[1:-1, :-2] - p[2:, :-2]) / 8.0
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]
          - p[:-2, :-2] - 2 * p[:-2, 1:-1] - p[:-2, 2:]) / 8.0
    magnitude = np.hypot(gx, gy)
    return (float(magnitude.mean()), float((magnitude >= 4).mean()),
            float((magnitude >= 12).mean()))


def calculate_quality(image):
    """Return brightness, contrast, sharpness, histogram, clipping, color, edges.

    Sharpness is four-neighbour Laplacian variance. It is content and resolution
    dependent; higher values can also reflect noise or oversharpening.
    """
    gray = _gray_array(image)
    if gray.size == 0:
        raise ValueError("Image contains no pixels")
    p = np.pad(gray, 1, mode="edge")
    lap = p[:-2, 1:-1] + p[2:, 1:-1] + p[1:-1, :-2] + p[1:-1, 2:] - 4 * gray
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    channel_means = {name: float(rgb[:, :, i].mean()) for i, name in enumerate(("red", "green", "blue"))}
    channel_stddevs = {name: float(rgb[:, :, i].std()) for i, name in enumerate(("red", "green", "blue"))}
    max_c = rgb.max(axis=2)
    saturation = np.divide(rgb.max(axis=2) - rgb.min(axis=2), max_c,
                           out=np.zeros_like(max_c), where=max_c > 0)
    hist, _ = np.histogram(gray, bins=32, range=(0, 256))
    luminance_p1, luminance_p99 = np.percentile(gray, [1, 99])
    edge_mean, edge_fraction_4, edge_fraction_12 = _edge_stats(gray)
    noise_proxy = estimate_noise(image)
    low_contrast_evidence = float(np.clip((60.0 - gray.std()) / 60.0, 0.0, 1.0))
    gray8 = image.convert("RGB")
    min_channel = np.asarray(gray8, dtype=np.uint8).min(axis=2)
    win = min(15, min(image.width, image.height))
    if win % 2 == 0:
        win -= 1
    if win < 1:
        win = 1
    dark = Image.fromarray(min_channel, mode="L")
    if win > 1:
        dark = dark.filter(ImageFilter.MinFilter(win))
    dark_channel_mean = float(np.asarray(dark, dtype=np.float32).mean() / 255.0)
    low_local_contrast_evidence = float(np.clip((30.0 - _local_contrast(gray)) / 30.0, 0.0, 1.0))
    # Heuristic evidence combines dark-channel lift, global/local contrast, and
    # low saturation. It is not a probability or weather label.
    haze_index = float(np.clip(0.35 * dark_channel_mean + 0.20 * low_contrast_evidence
                               + 0.25 * low_local_contrast_evidence
                               + 0.20 * (1.0 - float(saturation.mean())), 0.0, 1.0))
    return {
        "mean_brightness": float(gray.mean()),
        "contrast_stddev": float(gray.std()),
        "luminance_p1": float(luminance_p1), "luminance_p99": float(luminance_p99),
        "robust_dynamic_range": float(luminance_p99 - luminance_p1),
        "sharpness_laplacian_variance": float(lap.var()),
        "noise_estimate": noise_proxy,
        "width": int(image.width), "height": int(image.height),
        "mode": image.info.get("source_mode", image.mode),
        "channel_means_rgb": channel_means, "channel_stddevs_rgb": channel_stddevs,
        "clipped_shadow_fraction": float((gray <= 5).mean()),
        "clipped_highlight_fraction": float((gray >= 250).mean()),
        "local_contrast_stddev": _local_contrast(gray),
        "edge_gradient_mean": edge_mean,
        "edge_fraction_gradient_ge_4": edge_fraction_4,
        "edge_fraction_gradient_ge_12": edge_fraction_12,
        "mean_saturation": float(saturation.mean()),
        "dark_channel_mean": dark_channel_mean,
        "haze_like_indicator": haze_index,
        "haze_local_contrast_evidence": low_local_contrast_evidence,
        "haze_indicator_note": "Heuristic index using dark-channel lift, global/local contrast and low saturation; not a probability or weather classification.",
        "luminance_histogram_32": hist.astype(int).tolist(),
    }


def estimate_noise(image):
    """Robust fine-grain noise proxy from median-filter residual MAD.

    Edges and texture also contribute to this estimate, especially at small sizes.
    """
    gray = image.convert("L")
    smooth = np.asarray(gray.filter(ImageFilter.MedianFilter(3)), dtype=np.float32)
    residual = np.asarray(gray, dtype=np.float32) - smooth
    med = np.median(residual)
    return float(1.4826 * np.median(np.abs(residual - med)))
