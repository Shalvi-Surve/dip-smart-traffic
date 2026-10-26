"""Spatial-domain luminance operations that preserve chroma where possible."""
import numpy as np
from PIL import Image, ImageFilter, ImageOps


def as_rgb(image):
    if not isinstance(image, Image.Image):
        raise TypeError("Expected a PIL image")
    if image.width < 1 or image.height < 1:
        raise ValueError("Image has no pixels")
    return image.convert("RGB")


def normalize_luminance(image, low_percentile=1.0, high_percentile=99.0):
    """Robust percentile stretch of Y; returns RGB and preserves image geometry.

    This expands the selected luminance interval to [0,255]. It can clip useful
    shadow/highlight detail and is intentionally not used when the input already
    spans most of the available range.
    """
    if not 0 <= low_percentile < high_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= low < high <= 100")
    y, cb, cr = as_rgb(image).convert("YCbCr").split()
    values = np.asarray(y, dtype=np.uint8)
    low, high = np.percentile(values, [low_percentile, high_percentile])
    if high <= low:
        return Image.merge("YCbCr", (y, cb, cr)).convert("RGB")
    stretched = np.clip(np.rint((values.astype(np.float32) - low) * 255.0 / (high - low)), 0, 255).astype(np.uint8)
    return Image.merge("YCbCr", (Image.fromarray(stretched), cb, cr)).convert("RGB")


def gamma_correct(image, gamma):
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    y, cb, cr = as_rgb(image).convert("YCbCr").split()
    lut = [min(255, max(0, round(((v / 255) ** gamma) * 255))) for v in range(256)]
    return Image.merge("YCbCr", (y.point(lut), cb, cr)).convert("RGB")


def equalize_luminance(image):
    y, cb, cr = as_rgb(image).convert("YCbCr").split()
    return Image.merge("YCbCr", (ImageOps.equalize(y), cb, cr)).convert("RGB")


def clahe_luminance(image, tile_size=64, clip_limit=2.0):
    """CLAHE on Y of YCbCr using clipped tile histograms and interpolated LUTs."""
    if tile_size < 8 or clip_limit <= 0:
        raise ValueError("tile_size must be >= 8 and clip_limit positive")
    y, cb, cr = as_rgb(image).convert("YCbCr").split()
    a = np.asarray(y, dtype=np.uint8)
    h, w = a.shape
    rows = (h + tile_size - 1) // tile_size
    cols = (w + tile_size - 1) // tile_size
    maps = np.empty((rows, cols, 256), dtype=np.uint8)
    for ty in range(rows):
        for tx in range(cols):
            tile = a[ty*tile_size:min((ty+1)*tile_size,h), tx*tile_size:min((tx+1)*tile_size,w)]
            hist = np.bincount(tile.ravel(), minlength=256).astype(np.float64)
            limit = max(1, int(clip_limit * tile.size / 256))
            excess = int(np.maximum(hist - limit, 0).sum())
            hist = np.minimum(hist, limit) + excess / 256.0
            cdf = hist.cumsum()
            nz = np.flatnonzero(cdf)
            cmin = cdf[nz[0]] if len(nz) else 0
            denom = cdf[-1] - cmin
            maps[ty, tx] = np.clip(np.rint((cdf - cmin) * 255 / denom), 0, 255).astype(np.uint8) if denom > 0 else np.arange(256, dtype=np.uint8)
    # Bilinear interpolation of adjacent tile lookup tables reduces tile seams.
    yy = (np.arange(h) + 0.5) / tile_size - 0.5
    xx = (np.arange(w) + 0.5) / tile_size - 0.5
    gy = np.clip(yy, 0, rows - 1); gx = np.clip(xx, 0, cols - 1)
    y0 = np.floor(gy).astype(int); y1 = np.minimum(y0 + 1, rows - 1); wy = gy - y0
    x0 = np.floor(gx).astype(int); x1 = np.minimum(x0 + 1, cols - 1); wx = gx - x0
    values = a.astype(int)
    out = np.empty_like(a)
    for r in range(h):
        top = maps[y0[r], x0, values[r]] * (1-wx) + maps[y0[r], x1, values[r]] * wx
        bot = maps[y1[r], x0, values[r]] * (1-wx) + maps[y1[r], x1, values[r]] * wx
        out[r] = np.clip(np.rint(top * (1-wy[r]) + bot * wy[r]), 0, 255).astype(np.uint8)
    return Image.merge("YCbCr", (Image.fromarray(out), cb, cr)).convert("RGB")


def median_denoise(image, size=3):
    """Remove small impulse noise from luminance while retaining chroma."""
    if size < 3 or size % 2 == 0:
        raise ValueError("median size must be an odd integer >= 3")
    y, cb, cr = as_rgb(image).convert("YCbCr").split()
    return Image.merge("YCbCr", (y.filter(ImageFilter.MedianFilter(size=size)), cb, cr)).convert("RGB")


def gaussian_denoise(image, radius=1.0):
    """Gaussian smoothing of luminance; useful for fine, approximately Gaussian noise."""
    if radius <= 0:
        raise ValueError("radius must be positive")
    y, cb, cr = as_rgb(image).convert("YCbCr").split()
    return Image.merge("YCbCr", (y.filter(ImageFilter.GaussianBlur(radius)), cb, cr)).convert("RGB")


def bilateral_denoise(image, diameter=5, sigma_color=25.0, sigma_space=2.0):
    """Edge-aware bilateral filtering on luminance, implemented with NumPy.

    A small fixed neighborhood keeps this suitable for interactive still images.
    It is slower than optimized OpenCV implementations and may smooth fine road
    markings; color channels are retained from YCbCr.
    """
    if diameter < 3 or diameter % 2 == 0 or sigma_color <= 0 or sigma_space <= 0:
        raise ValueError("diameter must be odd and >= 3; sigmas must be positive")
    y, cb, cr = as_rgb(image).convert("YCbCr").split()
    src = np.asarray(y, dtype=np.float32)
    radius = diameter // 2
    padded = np.pad(src, radius, mode="reflect")
    h, w = src.shape
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    spatial = np.exp(-(xx * xx + yy * yy) / (2 * sigma_space * sigma_space)).astype(np.float32)
    result = np.empty((h, w), dtype=np.uint8)
    # Chunk rows to bound temporary float arrays for the configured large-image limit.
    for y0 in range(0, h, 96):
        y1 = min(h, y0 + 96)
        center = src[y0:y1]
        weighted = np.zeros_like(center); weight_sum = np.zeros_like(center)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                sample = padded[radius + y0 + dy:radius + y1 + dy, radius + dx:radius + dx + w]
                range_weight = np.exp(-((sample - center) ** 2) / (2 * sigma_color * sigma_color))
                weight = spatial[dy + radius, dx + radius] * range_weight
                weighted += weight * sample
                weight_sum += weight
        result[y0:y1] = np.clip(np.rint(weighted / np.maximum(weight_sum, 1e-12)), 0, 255).astype(np.uint8)
    return Image.merge("YCbCr", (Image.fromarray(result), cb, cr)).convert("RGB")


def edge_map(image, operator="sobel"):
    """Return an L-mode diagnostic edge magnitude using Sobel or Prewitt kernels."""
    if operator not in {"sobel", "prewitt"}:
        raise ValueError("operator must be 'sobel' or 'prewitt'")
    gray = np.asarray(as_rgb(image).convert("L"), dtype=np.float32)
    p = np.pad(gray, 1, mode="edge")
    if operator == "sobel":
        kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8
        ky = kx.T
    else:
        kx = np.array([[-1, 0, 1]] * 3, dtype=np.float32) / 6
        ky = kx.T
    gx = sum(kx[i, j] * p[i:i + gray.shape[0], j:j + gray.shape[1]] for i in range(3) for j in range(3))
    gy = sum(ky[i, j] * p[i:i + gray.shape[0], j:j + gray.shape[1]] for i in range(3) for j in range(3))
    magnitude = np.hypot(gx, gy)
    scale = float(np.percentile(magnitude, 99.5))
    out = np.zeros_like(gray, dtype=np.uint8) if scale <= 1e-8 else np.clip(magnitude * (255.0 / scale), 0, 255).astype(np.uint8)
    return Image.fromarray(out, mode="L")


def edge_enhance(image, strength=0.25, operator="sobel"):
    """Add a bounded gradient-magnitude detail term to Y and return a photograph."""
    if not 0 <= strength <= 1:
        raise ValueError("strength must be in [0, 1]")
    rgb = as_rgb(image)
    y, cb, cr = rgb.convert("YCbCr").split()
    edges = np.asarray(edge_map(rgb, operator), dtype=np.float32) / 255.0
    src = np.asarray(y, dtype=np.float32)
    enhanced = np.clip(src + strength * 24.0 * edges, 0, 255).astype(np.uint8)
    return Image.merge("YCbCr", (Image.fromarray(enhanced), cb, cr)).convert("RGB")


def morphology(image, operation="opening", size=3):
    """Grayscale diagnostic morphology (erosion/dilation/opening/closing).

    The result is an auxiliary L-mode image, not a natural color photograph.
    """
    if operation not in {"erosion", "dilation", "opening", "closing"}:
        raise ValueError("unsupported morphology operation")
    if size < 3 or size % 2 == 0:
        raise ValueError("size must be an odd integer >= 3")
    gray = as_rgb(image).convert("L")
    erode = lambda im: im.filter(ImageFilter.MinFilter(size))
    dilate = lambda im: im.filter(ImageFilter.MaxFilter(size))
    result = {"erosion": erode, "dilation": dilate,
              "opening": lambda im: dilate(erode(im)),
              "closing": lambda im: erode(dilate(im))}[operation](gray)
    return result


def unsharp_luminance(image, radius=1.5, percent=100, threshold=4):
    y, cb, cr = as_rgb(image).convert("YCbCr").split()
    sharpened = y.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
    return Image.merge("YCbCr", (sharpened, cb, cr)).convert("RGB")
