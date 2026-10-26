"""In-memory batch execution and reproducible ZIP export."""
from io import BytesIO
import json
import re
import warnings
import zipfile
from PIL import Image, UnidentifiedImageError
from dip.adaptive import build_pipeline, normalize_pipeline_result
from .reporting import batch_csv_bytes, processing_report
from .visibility import (visibility_category_csv_bytes, visibility_image_csv_bytes,
                         visibility_region_csv_bytes, visibility_reports,
                         visibility_stage_csv_bytes)

MAX_IMAGE_PIXELS = 16_000_000
MAX_BATCH_IMAGES = 10


class BatchResults(list):
    """List-compatible successful items with per-file failures attached."""
    def __init__(self, *args, errors=None):
        super().__init__(*args)
        self.errors = errors or []


def decode_image(data):
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ValueError("The uploaded file is empty")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(BytesIO(data))
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise ValueError(f"Image exceeds the {MAX_IMAGE_PIXELS:,}-pixel processing limit")
            image.load()
            original_mode = image.mode
            rgb = image.convert("RGB")
            rgb.info["source_mode"] = original_mode
            return rgb
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ValueError("Unsupported or invalid image. Use a valid PNG, JPEG, BMP, TIFF, or WebP image.") from exc


def png_bytes(image):
    out = BytesIO()
    image.save(out, format="PNG")
    return out.getvalue()


def process_batch(items, denoise_choice="automatic", dehaze_choice="automatic",
                  edge_diagnostic=False, morphology_choice="off", selected_operations=None,
                  progress_callback=None):
    """items is an iterable of (filename, original file bytes)."""
    results = BatchResults()
    items = list(items)
    if len(items) > MAX_BATCH_IMAGES:
        raise ValueError(f"Please process at most {MAX_BATCH_IMAGES} images per batch")
    for index, record in enumerate(items, start=1):
        filename, data = record[0], record[1]
        source_metadata = record[2] if len(record) > 2 else {}
        safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", filename.rsplit(".", 1)[0]).strip("._") or "image"
        try:
            image = decode_image(data)
            def report_progress(event):
                if progress_callback:
                    progress_callback({**event, "filename": filename, "image_index": index,
                                       "image_count": len(items)})
            result = build_pipeline(image, denoise_choice=denoise_choice, dehaze_choice=dehaze_choice,
                                    source_metadata=source_metadata, edge_diagnostic=edge_diagnostic,
                                    morphology_choice=morphology_choice,
                                    selected_operations=selected_operations,
                                    progress_callback=report_progress if progress_callback else None)
        except (ValueError, OSError, MemoryError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            results.errors.append({"filename": filename, "error": error})
            if progress_callback:
                progress_callback({"event": "image_failed", "filename": filename,
                                   "image_index": index, "image_count": len(items), "error": error})
            continue
        result.update({"index": index, "filename": filename, "archive_stem": f"{index:03d}_{safe_stem}",
                       "original_bytes": bytes(data), "source_metadata": source_metadata})
        results.append(result)
    if progress_callback:
        progress_callback({"event": "batch_complete", "image_count": len(items),
                           "completed": len(results), "failed": len(results.errors)})
    return results


def batch_zip(results, failures=None, annotation_index=None, visibility_evaluation=None):
    out = BytesIO()
    summary = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as archive:
        for item in results:
            item = normalize_pipeline_result(item)
            root = item["archive_stem"]
            archive.writestr(f"{root}/original{_extension(item['filename'])}", item["original_bytes"])
            for number, stage in enumerate(item["stages"], start=1):
                archive.writestr(f"{root}/intermediate_{number:02d}_{_slug(stage['name'])}.png", png_bytes(stage["image"]))
            archive.writestr(f"{root}/final.png", png_bytes(item["final"]))
            record = processing_report(item, item["filename"], item.get("source_metadata", {}))
            archive.writestr(f"{root}/processing_log.json", json.dumps(record, indent=2))
            summary.append(record)
        failures = failures or getattr(results, "errors", [])
        archive.writestr("batch_summary.json", json.dumps({"image_count": len(results), "images": summary,
                                                             "failed_files": failures}, indent=2))
        archive.writestr("batch_stages.csv", batch_csv_bytes(results))
        if failures:
            archive.writestr("batch_errors.csv", _failure_csv(failures))
        evaluation_reports, category_rows = (visibility_evaluation if visibility_evaluation is not None
                                             else visibility_reports(results, annotation_index))
        archive.writestr("visibility_per_image.csv", visibility_image_csv_bytes(evaluation_reports))
        archive.writestr("visibility_per_stage.csv", visibility_stage_csv_bytes(evaluation_reports))
        archive.writestr("visibility_vehicle_regions.csv", visibility_region_csv_bytes(evaluation_reports))
        archive.writestr("visibility_by_category.csv", visibility_category_csv_bytes(category_rows))
    return out.getvalue()


def _slug(value):
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _extension(filename):
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else "bin"
    return f".{ext}" if ext in {"png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"} else ".bin"


def _failure_csv(failures):
    import csv
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["filename", "error"])
    writer.writeheader()
    writer.writerows(failures)
    return stream.getvalue().encode("utf-8-sig")
