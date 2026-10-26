# Smart Traffic Image Enhancement

**An Adaptive Digital Image Processing Approach for Adverse Weather Traffic Surveillance**

Research workbench for explainable, classical enhancement of traffic imagery. The adaptive planner is deterministic and rule-based. It is not a trained weather classifier, vehicle detector, or incident detector.

## Run (Windows / Python 3.11)

```powershell
cd D:\dip-project\Smart-Traffic-DIP
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install -r requirements-dev.txt
python -m streamlit run app.py
```

The interface has three destinations: **Traffic Vision Lab** for the project overview and bundled DAWN imagery, **Processing Studio** for guided single-image or batch processing, and **Results & Evaluation** for session measurements and research methodology. In the Processing Studio choose **Guided single image** or **Batch processing**. **Adaptive** is the default rule-based plan. **Fixed order** exposes the existing enhancement methods and runs the selected methods in selection order; denoising resolves to the chosen spatial/frequency filter, and no-op or failed operations are reported rather than shown as completed stages. Single-image processing validates the upload and waits for **Start DIP Processing**. During execution, a progress panel updates with each actual computed output; afterward, a serpentine route previews the retained stages with their real saved images. Navigation reads those saved arrays and does not rerun or mutate the upload. Batch mode accepts uploads or bundled DAWN samples and runs the same engine independently per image with progress previews. It accepts at most 10 images per run; each image is limited to 16 megapixels. Reports and the output archive are assembled in memory.

Run automated checks with:

```powershell
python -m pytest -q
python -m compileall -q app.py src tests
```

## Application and processing architecture

- `app.py`: dark responsive Streamlit research UI, single/batch workflows, timeline navigation, synchronized original/current comparison, metrics, downloads, and DAWN browser.
- `src/analysis/quality.py`: measured luminance, contrast, clipping, histogram, Laplacian variance, Sobel edge distribution, local contrast, channel statistics, noise proxy, and haze-like indicator.
- `src/analysis/degradation.py`: candidate flags and evidence with documented interpretations, not probabilities.
- `src/dip/adaptive.py`: analysis-first decision engine that builds a structured plan, then executes selected stages. Every selected/skipped/failed/unavailable action is logged.
- `src/dip/spatial.py`: robust Y-channel normalization, gamma correction, global equalization, hand-implemented CLAHE with interpolated tile LUTs, median/Gaussian/bilateral filters, Sobel/Prewitt edge maps and bounded edge emphasis, grayscale morphology, and luminance unsharp masking.
- `src/dip/dehaze.py`: guarded dark-channel-prior atmospheric-light and transmission estimate with scattering-model recovery.
- `src/dip/frequency.py`: centered log-magnitude FFT visualization and a real Gaussian frequency-domain luminance filter with inverse FFT reconstruction.
- `src/traffic_dip/reporting.py`: JSON processing reports and CSV batch-stage reports.
- `src/traffic_dip/batch.py`: validated decode, duplicate-safe ZIP paths, original preservation, per-image logs, and batch manifest.
- `src/traffic_dip/detection.py`: optional local-only Ultralytics COCO vehicle detection, isolated from the DIP pipeline.
- `src/traffic_dip/evaluation.py`: Pascal VOC XML parsing and fixed-IoU precision/recall/F1 calculation; it does not compute mAP.

The comparison viewer shows the original versus actual selected-stage output and a separate previous-stage versus current-stage comparison. Its divider is keyboard-operable with arrow keys; side-by-side mode is available as fallback. The stage timeline has Previous/Next controls and direct selection. Auxiliary Sobel and morphology diagnostics are labeled as grayscale diagnostic outputs and never replace the final natural-color photograph. “Actual pixels” permits scrolling at native resolution.

## Measurements and decision rules

All image measurements are computed from pixels. Brightness is mean grayscale luminance; global and local contrast are standard deviations; clipping fractions count luminance values ≤5 and ≥250; histogram is 32 luminance bins; edge distribution uses Sobel gradient magnitude; sharpness is variance of a four-neighbour discrete Laplacian. A higher sharpness value can reflect noise or oversharpening and does not necessarily indicate better appearance. The median-filter residual MAD is a fine-grain noise proxy but also responds to texture and edges. Color means and standard deviations are computed per RGB channel. No PSNR/SSIM is reported without a paired, spatially aligned clear reference.

The planner uses these initial demonstrator rules. They are conservative and may yield fewer than five photo-enhancement stages when measurements do not justify more; skipped choices include their reasons.

- **Gamma:** if mean luminance is below 75, select a power-law gamma calculated to map the mean toward 105; if above 185, calculate gamma toward 165. Gamma is clamped to [0.55, 1.60] and applied to Y in YCbCr. It is skipped if at least 98% of pixels are clipped at the corresponding black/white boundary.
- **Normalization:** if the 1st–99th luminance percentile range is under 225, stretch that interval to [0,255]. It is skipped when the available range is already broad; robust stretching can clip tonal details.
- **Dehazing:** the heuristic haze-like index combines mean dark-channel value (0.35), low global contrast (0.20), low local contrast (0.25), and low saturation (0.20). Dehazing is considered only when the index is ≥0.58, mean brightness ≥75, dark-channel mean ≥0.22, and Sobel edge fraction at gradient magnitude ≥4 is ≥0.005. The index is not a probability. The user can disable automatic dehazing or force it experimentally.
- **Contrast:** if global contrast is below 42 or local tile contrast is below 20 and robust luminance range is nonzero, select one CLAHE stage unless dehazing was selected. Global histogram equalization is implemented and tested as a standalone method, but the adaptive policy currently skips it to avoid global over-expansion and redundancy.
- **Denoising:** automatic mode selects FFT Gaussian low-pass only when median-residual MAD is ≥8; sigma is clipped to [0.5,1.2] and derived from that measurement. Median, Gaussian, bilateral, and frequency filters are available for explicit selection; “off” skips denoising. These filters can erase road markings or texture.
- **Sharpening:** one luminance unsharp mask is selected for Laplacian variance below 180 only when edge activity is measurable and the noise proxy is low or denoising is selected. No stacked sharpening operations are applied.
- **Edges and morphology:** Sobel and Prewitt edge maps, bounded Sobel edge emphasis, and grayscale erosion/dilation/opening/closing are implemented. Edge and morphology outputs are optional diagnostics; default policy avoids substituting them for a photograph. Morphology is also offered as a user-selected diagnostic. Rain-streak suppression is unavailable: no rain-specific method is implemented or mislabeled as removal.
- **Frequency:** the FFT spectrum is a diagnostic visualization. Automatic filtering is selected only by the measured noise rule; it is not presented as a universal enhancement.

Each operation report records the rule, rationale, parameters, elapsed time, before/after metrics, average absolute RGB pixel change, diagnostics where available, and known side effects. Pixel change and metric changes are not perceptual-quality claims.

### Dark-channel dehazing limitations

The implementation uses the classical atmospheric-scattering model: estimate atmospheric light from the brightest pixels among high dark-channel candidates, estimate transmission with the dark-channel prior, clamp transmission, and recover scene radiance. The adaptive plan uses a conservative omega of 0.70 and a transmission floor of 0.55 to reduce severe darkening. The direct algorithm API retains configurable parameters. Its assumptions often fail in skies, bright vehicles/buildings, night scenes, saturated scenes, and non-hazy images. It can produce halos, darken an image, or shift color; inspect the actual intermediate. The evidence gate is a heuristic, not weather recognition.

## DAWN data and attribution

The project includes an existing local DAWN collection under `data/samples/<category>/` and source files under `data/raw/dawn/`. Its current CSV manifest has 1,027 sample records: fog 300, rain 200, sandstorm 323, snow 204. The published Mendeley landing page describes 1,000 images and four weather groups, so the local 1,027-row count is recorded as an acquisition/version discrepancy rather than represented as the canonical published count.

Source: Mourad A. Kenk and Mahmoud Hassaballah, **DAWN: Vehicle Detection in Adverse Weather Nature Dataset**, Mendeley Data v3, DOI [10.17632/766ygrbt8y.3](https://data.mendeley.com/datasets/766ygrbt8y/3); paper [arXiv:2008.05402](https://arxiv.org/abs/2008.05402). The data page lists **CC BY-NC 3.0**, states non-commercial use, and disallows commercial use/distribution. Follow the source terms, attribute the authors/dataset, and verify the terms before redistributing any images. `data/metadata/dawn_sample_manifest.csv` preserves filenames, weather category, portable local path, source URL, DOI, attribution, and license. Category values are provenance metadata; they are not treated as classifier outputs or clean-reference labels.

DAWN contains adverse-weather images and object annotations, not aligned clean-weather reference images. Those annotations are not consumed by the enhancement pipeline. Images may have been collected from web sources and represent a limited range of scenes; the local category balance differs from the published page’s stated total. The UI can browse the local samples without downloading data.

The bundled raw DAWN directory contains 1,027 paired Pascal VOC XML and YOLO text annotation records. XML records contain the image filename and dimensions, named objects, and pixel-space `xmin/ymin/xmax/ymax` boxes. YOLO rows use normalized `class_id x_center y_center width height`; a full pair audit found their boxes agree after coordinate conversion. The local collection has no `.names` class map; observed IDs map to person=1, bicycle=2, car=3, motorcycle=4, bus=6, train=7, truck=8. One YOLO class-6 label disagrees with its VOC label, so the evaluator uses XML names and class-mapping limitations must be considered.

## Optional AI vehicle analysis

The base requirements do not install AI dependencies or model weights. To enable optional inference in the existing Python 3.11 environment:

```powershell
python -m pip install -r requirements-ai.txt
```

Place compatible pretrained weights at `models/yolo26n.pt`, or set `TRAFFIC_YOLO_MODEL` to an existing local `.pt` file before launching Streamlit. The detector checks for an existing file before calling Ultralytics, preventing implicit weight downloads. Ultralytics documents Python 3.8+ support and the pretrained YOLO26 nano model; check its code/model license for your intended use. If the optional package or weights are unavailable, classical DIP processing remains usable.

The optional analysis compares COCO car, motorcycle, bus, and truck detections on original and final images, displays separate box visualizations, confidence scores, per-class counts, count deltas, and inference times. These observations are not ground-truth accuracy and do not prove enhancement improved detection. Batch mode can export the same per-image observations to a separate CSV.

`parse_voc_annotation` reads annotation XML and ignores stale machine-specific `<path>` values. `evaluate_detection_records` performs class-aware, confidence-descending greedy matching per image at a fixed IoU threshold (default 0.5), excludes difficult objects from the denominator, and reports precision, recall, and F1. It is not an average-precision/mAP evaluation. No detector benchmark is claimed until optional inference has been run under a documented protocol.

## Evaluation workflow

Use the same fixed input filenames and source metadata for each algorithm/configuration. Download single-image JSON reports or the batch ZIP/CSV; keep the input hashes or immutable copies alongside the run. Compare measured brightness, clipping, contrast, Laplacian/Sobel response, noise proxy, and processing time while separately conducting blinded visual review. Do not rank algorithms by contrast or sharpness alone. A valid PSNR/SSIM study requires an aligned clear-weather reference or a documented synthetic corruption/clean-reference protocol. DAWN alone does not provide paired clear-weather targets. A detector impact study additionally needs a real detector, unchanged evaluation images/annotations, and a documented detection metric/protocol; this project makes no detection-improvement claim.

### Visibility evaluation exports

The optional visibility report reuses `analysis.quality.calculate_quality` for mean luminance, luminance standard deviation, 64-pixel-tile local contrast, four-neighbour Laplacian variance, median-residual noise proxy, clipping fractions, and Sobel edge evidence. It records the original and each actual saved stage, adjacent-stage pixel-change counts, absolute/relative metric deltas (relative change is blank when the baseline is zero), stage duration, and the selected sequence with parameters. It also produces per-image, per-stage, vehicle-region, and category-summary CSV files in batch mode and includes them in the batch ZIP. Category summaries are descriptive arithmetic means, not scores. Per-image vehicle counts distinguish eligible annotations, valid regions, and invalid regions; invalid boxes remain logged with a status but are excluded from region measurements. Category CSV headers are stable even when there are no category rows.

For explicitly identified bundled DAWN images, the report can use matching VOC XML vehicle boxes. Coordinates are scaled from annotation dimensions to the source dimensions, converted to a floor/ceil half-open crop, clipped to image bounds, and the scale/clipping are recorded. The same crop coordinates are applied to the original and each stage; dimension-changing stage outputs are marked as mismatches and not remapped. The paired YOLO files use normalized center-based coordinates; region evaluation uses VOC pixel `xyxy` names to avoid the known class-id mapping discrepancy. No DAWN plate boxes or readable-character labels are present in the inspected annotations, so plate evaluation is unavailable. User uploads are not assumed to be a specific DAWN image based on filename alone.

The `data/reference` folder contains an unrelated ACDC segmentation overlay, not a paired clear reference. Images in `data/samples/clear` are also unpaired. The application therefore omits PSNR/SSIM. Noise is reported as a texture-sensitive proxy, and edge/local-contrast/clipping conditions are indicators for inspection; none is a universal visibility index or a claim of true noise removal or improved vehicle/plate readability.

For a later plate-specific study, the [UFPR-ALPR dataset](https://web.inf.ufpr.br/vri/databases/ufpr-alpr/) is a candidate because its official project page describes license-plate and character position annotations. Access is restricted to academic research by educational/research institutions for non-commercial purposes; it is not an unrestricted open-license dataset. It is not bundled or downloaded by this project. Its moving-camera Brazilian scenes are not paired clean references for DAWN, so it would support a separate plate-region experiment only after access and terms are confirmed, not a DAWN adverse-weather plate-readability claim.

## Known limits

- Rule thresholds are explainable starting values, not trained or benchmark-optimized.
- Weather indicators can confuse illumination, scene content, and camera characteristics with degradation.
- Median-residual noise estimates count fine texture; FFT filtering may erase useful texture.
- CLAHE, gamma, dehazing, and sharpening can amplify artifacts or change appearance.
- Rain-streak removal and incident detection are not implemented. Optional vehicle detection uses a general COCO model, not a model trained on DAWN or an accident detector.
- Browser slider comparison is served as a small local HTML component; side-by-side comparison remains available if a browser blocks that component.
