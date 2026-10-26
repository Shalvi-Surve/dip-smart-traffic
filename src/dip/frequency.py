"""FFT visualization and Gaussian low-pass denoising on luminance."""
import numpy as np
from PIL import Image
from .spatial import as_rgb


def spectrum_image(image):
    gray = np.asarray(as_rgb(image).convert("L"), dtype=np.float32)
    spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(gray))))
    lo, hi = float(spectrum.min()), float(spectrum.max())
    scaled = np.zeros_like(spectrum, dtype=np.uint8) if hi == lo else np.rint((spectrum-lo)*255/(hi-lo)).astype(np.uint8)
    return Image.fromarray(scaled, mode="L")


def gaussian_lowpass_luminance(image, sigma=0.8):
    """Gaussian low-pass in the Fourier domain, followed by inverse FFT."""
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    rgb = as_rgb(image)
    y, cb, cr = rgb.convert("YCbCr").split()
    a = np.asarray(y, dtype=np.float32)
    h, w = a.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    transfer = np.exp(-2 * np.pi**2 * sigma**2 * (fx*fx + fy*fy))
    restored = np.fft.ifft2(np.fft.fft2(a) * transfer).real
    restored = Image.fromarray(np.clip(np.rint(restored), 0, 255).astype(np.uint8))
    return Image.merge("YCbCr", (restored, cb, cr)).convert("RGB")
