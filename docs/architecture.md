# Processing and application architecture

## Request flow

1. Streamlit accepts an upload or a selected bundled DAWN image. `decode_image` validates bytes, file decode and pixel count, captures the source mode, then makes an RGB working copy. Single-image mode waits for the explicit **Start DIP Processing** action.
2. `analysis.quality.calculate_quality` measures luminance distribution, clipping, Laplacian and Sobel responses, tile-local contrast, channels, saturation, dark-channel mean, and median residual MAD.
3. `analysis.degradation.assess_degradation` converts measurements into transparent heuristic flags and definitions. It does not infer weather labels from pixels with a trained model.
4. `dip.adaptive.analyze_and_plan` creates an operation record for every candidate, including selected/skipped/unavailable status, rationale, parameters, and limitation. The complete plan exists before processing begins.
5. `dip.adaptive.build_pipeline` executes only planned operations. A successful result stores the actual stage input and output, output kind, metrics, parameters, rationale, and timing. Auxiliary edge/morphology diagnostics do not replace the natural-color final image. A failure is recorded and the last valid image remains available.
6. The timeline indexes the original plus each successful saved stage. `stage_view` returns original/current/previous metrics without reprocessing. The interface presents an ordered replay after execution; it does not claim the operations are streamed live.
7. Reports omit image arrays and contain actual plan, measurements, diagnostic values, and elapsed times. Batch ZIP contains original source bytes for successful decodes, successful stage PNGs, final PNG, per-image JSON, batch manifest, stage CSV, and invalid-file errors where present.
8. Optional vehicle inference is isolated in `traffic_dip.detection`; it runs only when requested, uses cached local weights, and compares the original with the final DIP output without changing the enhancement stages. Batch detection observations download as a separate CSV.
9. `traffic_dip.evaluation` parses Pascal VOC XML and computes class-aware precision/recall/F1 at a fixed IoU threshold. It does not compute mAP, and no detector evaluation has been run without the optional runtime and weights.

## Method summary

- Illumination and equalization operate on Y in YCbCr to limit direct RGB channel shifts.
- Robust 1st–99th percentile luminance normalization and power-law gamma correction are available.
- Global histogram equalization and hand-implemented tile CLAHE are distinct alternatives; the adaptive policy favors CLAHE for low contrast and avoids stacking it with dehazing.
- Median, Gaussian, and edge-aware bilateral spatial denoisers are available, in addition to the Fourier Gaussian filter.
- CLAHE uses clipped tile histograms and bilinear LUT interpolation.
- Sobel/Prewitt maps and grayscale erosion, dilation, opening, and closing are explicit auxiliary methods; they are not mistaken for natural-color enhancement.
- Dark-channel dehazing estimates atmospheric light and transmission under the scattering model with a minimum transmission clamp.
- FFT denoising uses a Gaussian low-pass transfer function on luminance and inverse FFT reconstruction. It is selected only when the noise proxy crosses its rule or the user selects it.
- One thresholded unsharp mask or conservatively gated Sobel edge emphasis is allowed after the analysis/denoise decision.

All rules and limitations are in `README.md`; change thresholds only through documented, reproducible experiments.

## Data boundaries

Weather category, source URL, filename and license are metadata only for DIP enhancement. DAWN’s adverse-weather categories do not provide aligned clear-weather references. Its Pascal VOC detection annotations are available to the separate evaluation utility, but they do not enter enhancement decisions. User upload names and original bytes are retained in reports/archives where applicable; intermediate paths use index-prefixed sanitized names.
