"""Simple, transparent degradation flags based on measured input indicators."""
from .quality import calculate_quality, estimate_noise


def assess_degradation(image):
    metrics = calculate_quality(image)
    noise = metrics["noise_estimate"]
    flags = []
    if metrics["mean_brightness"] < 75:
        flags.append("underexposed")
    elif metrics["mean_brightness"] > 185:
        flags.append("overexposed")
    if metrics["contrast_stddev"] < 42:
        flags.append("low_contrast")
    if noise >= 8:
        flags.append("high_frequency_noise")
    if metrics["sharpness_laplacian_variance"] < 180:
        flags.append("low_edge_detail")
    if metrics["clipped_shadow_fraction"] > 0.05:
        flags.append("shadow_clipping")
    if metrics["clipped_highlight_fraction"] > 0.05:
        flags.append("highlight_clipping")
    if metrics["local_contrast_stddev"] < 30:
        flags.append("low_local_contrast")
    if metrics["haze_like_indicator"] >= 0.58:
        flags.append("haze_like_statistics")
    evidence = {
        "haze_like_indicator": metrics["haze_like_indicator"],
        "interpretation": "Weighted dark-channel mean, low global and local contrast, and low saturation proxy; not a probability or weather label.",
        "noise_proxy": noise,
        "noise_interpretation": "Median-filter residual MAD also responds to edges and texture.",
    }
    return {"metrics": metrics, "noise_estimate": noise, "flags": flags, "evidence": evidence}
