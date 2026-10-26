"""Classical dark-channel-prior dehazing with guarded atmospheric estimation."""
import numpy as np
from PIL import Image, ImageFilter
from .spatial import as_rgb


def dark_channel_prior(image, window=15, omega=0.90, transmission_floor=0.20, strength=1.0):
    """Estimate atmospheric light/transmission and recover an RGB image.

    This implements the dark-channel prior model. It can over-correct skies,
    bright vehicles, night scenes, or scenes that do not satisfy the prior.
    Returns (actual enhanced PIL image, measurable estimation diagnostics).
    """
    if window < 3 or window % 2 == 0:
        raise ValueError("window must be an odd integer >= 3")
    if not 0 < omega <= 1:
        raise ValueError("omega must be in (0, 1]")
    if not 0 < transmission_floor <= 1:
        raise ValueError("transmission_floor must be in (0, 1]")
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    rgb = as_rgb(image)
    source = np.asarray(rgb, dtype=np.float32) / 255.0
    h, w, _ = source.shape
    window = min(window, min(h, w) if min(h, w) % 2 else max(1, min(h, w) - 1))
    if window < 3:
        raise ValueError("Image must be at least 3 pixels wide and high for dark-channel estimation")

    def channel_dark(arr):
        return np.asarray(Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), mode="L")
                          .filter(ImageFilter.MinFilter(window)), dtype=np.float32) / 255.0

    dark = np.minimum.reduce([channel_dark(source[:, :, c]) for c in range(3)])
    count = max(1, int(np.ceil(h * w * 0.001)))
    candidates = np.argpartition(dark.ravel(), -count)[-count:]
    intensity = source.mean(axis=2).ravel()
    atmospheric = source.reshape(-1, 3)[candidates[np.argmax(intensity[candidates])]].copy()
    atmospheric = np.maximum(atmospheric, 0.05)
    normalized = np.clip(source / atmospheric[None, None, :], 0, 1)
    normalized_dark = np.minimum.reduce([channel_dark(normalized[:, :, c]) for c in range(3)])
    transmission = np.clip(1.0 - omega * normalized_dark, transmission_floor, 1.0)
    recovered = (source - atmospheric[None, None, :]) / transmission[:, :, None] + atmospheric[None, None, :]
    recovered = source * (1.0 - strength) + recovered * strength
    out = Image.fromarray(np.clip(np.rint(recovered * 255), 0, 255).astype(np.uint8), mode="RGB")
    diagnostics = {
        "window": int(window), "omega": float(omega), "transmission_floor": float(transmission_floor),
        "strength": float(strength),
        "atmospheric_light_rgb": [float(v) for v in atmospheric],
        "transmission_mean": float(transmission.mean()),
        "transmission_min": float(transmission.min()),
        "transmission_max": float(transmission.max()),
        "limitation": "Dark-channel assumptions may fail in skies, bright objects, night scenes, or non-hazy images.",
    }
    return out, diagnostics
