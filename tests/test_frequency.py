import numpy as np
from PIL import Image
from dip.frequency import gaussian_lowpass_luminance, spectrum_image


def test_fft_preview_and_inverse_reconstruction_keep_geometry():
    a = np.zeros((32, 40), dtype=np.uint8)
    a[:, ::2] = 255
    image = Image.fromarray(a, "L")
    spectrum = spectrum_image(image)
    filtered = gaussian_lowpass_luminance(image, sigma=.8)
    assert spectrum.mode == "L" and spectrum.size == image.size
    assert filtered.mode == "RGB" and filtered.size == image.size
    assert np.std(np.asarray(filtered.convert("L"))) < np.std(a)
