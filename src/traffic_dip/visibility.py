"""Reproducible image-statistics and vehicle-region visibility indicators.

This module reports measurements and explicit heuristic warnings separately.
It does not compute a universal visibility score or infer detector accuracy.
"""
import csv
from io import StringIO
import json
import math
from pathlib import Path
import statistics

import numpy as np
from PIL import Image

from analysis.quality import calculate_quality
from dip.adaptive import normalize_pipeline_result
from .evaluation import parse_voc_annotation


METRIC_FIELDS = (
    "mean_brightness", "contrast_stddev", "local_contrast_stddev",
    "sharpness_laplacian_variance", "noise_estimate", "clipped_shadow_fraction",
    "clipped_highlight_fraction", "edge_gradient_mean",
    "edge_fraction_gradient_ge_4", "edge_fraction_gradient_ge_12",
    "robust_dynamic_range",
)
METRIC_LABELS = {
    "mean_brightness": "mean_luminance",
    "contrast_stddev": "luminance_stddev",
    "local_contrast_stddev": "local_contrast_stddev",
    "sharpness_laplacian_variance": "sharpness_laplacian_variance",
    "noise_estimate": "noise_proxy_estimate",
    "clipped_shadow_fraction": "shadow_clipping_fraction",
    "clipped_highlight_fraction": "highlight_clipping_fraction",
    "edge_gradient_mean": "edge_gradient_mean",
    "edge_fraction_gradient_ge_4": "edge_fraction_gradient_ge_4",
    "edge_fraction_gradient_ge_12": "edge_fraction_gradient_ge_12",
    "robust_dynamic_range": "robust_luminance_range",
}
VEHICLE_REGION_CLASSES = {"car", "motorcycle", "bus", "truck", "train", "bicycle"}


def metric_differences(before, after):
    """Return absolute and relative changes for the shared numeric metrics.

    Relative change is ``None`` for a zero baseline or non-finite values.
    Fractions are expressed as fractions (not percentage points).
    """
    differences = {}
    for key in METRIC_FIELDS:
        old, new = before.get(key), after.get(key)
        if not isinstance(old, (int, float)) or not isinstance(new, (int, float)):
            differences[METRIC_LABELS[key]] = {"absolute": None, "relative": None}
            continue
        if not math.isfinite(float(old)) or not math.isfinite(float(new)):
            differences[METRIC_LABELS[key]] = {"absolute": None, "relative": None}
            continue
        delta = float(new) - float(old)
        relative = None if float(old) == 0 else delta / abs(float(old))
        differences[METRIC_LABELS[key]] = {"absolute": delta, "relative": relative}
    return differences


def changed_pixel_statistics(before, after):
    """Count pixels where any RGB channel differs; input dimensions must match."""
    if not isinstance(before, Image.Image) or not isinstance(after, Image.Image):
        raise TypeError("Changed-pixel comparison requires PIL images")
    if before.size != after.size:
        raise ValueError(f"Image-size mismatch: {before.size} versus {after.size}")
    a = np.asarray(before.convert("RGB"), dtype=np.uint8)
    b = np.asarray(after.convert("RGB"), dtype=np.uint8)
    changed = np.any(a != b, axis=2)
    count = int(changed.sum())
    pixels = int(changed.size)
    return {"changed_pixel_count": count,
            "changed_pixel_fraction": count / pixels if pixels else None,
            "total_pixels": pixels}


def _detail_indicators(before, after):
    """Describe measurable detail changes without combining them into a score."""
    notes = []
    clip_before = before["clipped_shadow_fraction"] + before["clipped_highlight_fraction"]
    clip_after = after["clipped_shadow_fraction"] + after["clipped_highlight_fraction"]
    if clip_after > clip_before:
        notes.append("Measured clipped-pixel fraction increased; inspect tonal detail visually.")
    edge_before = before["edge_fraction_gradient_ge_4"]
    edge_after = after["edge_fraction_gradient_ge_4"]
    if edge_after < edge_before:
        notes.append("The fraction of pixels with gradient magnitude >=4 decreased; this is scene-content-sensitive edge evidence.")
    sharp_before = before["sharpness_laplacian_variance"]
    sharp_after = after["sharpness_laplacian_variance"]
    noise_before = before["noise_estimate"]
    noise_after = after["noise_estimate"]
    possible_amplification = (sharp_after > sharp_before and
                              noise_after - noise_before >= max(3.0, 0.5 * noise_before))
    if possible_amplification:
        notes.append("Sharpness proxy and texture-sensitive noise proxy both increased; possible detail or noise amplification needs visual inspection.")
    if not notes:
        notes.append("No configured adverse statistical indicator crossed its reporting condition; this does not establish improved visibility.")
    return {
        "edge_evidence_before": edge_before,
        "edge_evidence_after": edge_after,
        "edge_evidence_absolute_change": edge_after - edge_before,
        "local_contrast_before": before["local_contrast_stddev"],
        "local_contrast_after": after["local_contrast_stddev"],
        "combined_clipping_before": clip_before,
        "combined_clipping_after": clip_after,
        "possible_sharpness_or_noise_amplification": bool(possible_amplification),
        "noise_note": "Median-filter residual MAD is a texture-sensitive noise proxy, not a measure of true noise removed.",
        "interpretation_notes": notes,
    }


def compare_images(before_image, after_image, processing_ms=None,
                   before_metrics=None, after_metrics=None):
    """Measure one aligned pair with the same quality implementation."""
    if before_image.size != after_image.size:
        raise ValueError(f"Image-size mismatch: {before_image.size} versus {after_image.size}")
    before = before_metrics if before_metrics is not None else calculate_quality(before_image)
    after = after_metrics if after_metrics is not None else calculate_quality(after_image)
    return {
        "before": before,
        "after": after,
        "metric_differences": metric_differences(before, after),
        **changed_pixel_statistics(before_image, after_image),
        "processing_ms": processing_ms,
        "detail_indicators": _detail_indicators(before, after),
    }


def _category(metadata):
    return str((metadata or {}).get("category") or "unspecified").strip().lower() or "unspecified"


def _annotation_region(image, bbox_xyxy, annotation_size):
    """Crop a VOC xyxy region, scaling to image size and clipping to bounds.

    VOC coordinates are treated as continuous pixel coordinates. The crop uses
    floor for its top/left and ceil for its exclusive bottom/right edge.
    """
    if not isinstance(image, Image.Image):
        raise TypeError("Region extraction requires a PIL image")
    if len(bbox_xyxy) != 4:
        raise ValueError("Vehicle box must contain four xyxy coordinates")
    coords = tuple(float(value) for value in bbox_xyxy)
    if not all(math.isfinite(value) for value in coords):
        raise ValueError("Vehicle box coordinates must be finite")
    x1, y1, x2, y2 = coords
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Vehicle box must have positive width and height")
    ann_width, ann_height = map(int, annotation_size)
    if ann_width <= 0 or ann_height <= 0:
        raise ValueError("Annotation dimensions must be positive")
    annotation_ratio = ann_width / ann_height
    image_ratio = image.width / image.height
    if not math.isclose(annotation_ratio, image_ratio, rel_tol=1e-3, abs_tol=1e-6):
        raise ValueError("Image and annotation aspect ratios differ; vehicle coordinates cannot be aligned safely")
    scale_x, scale_y = image.width / ann_width, image.height / ann_height
    scaled = (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)
    ix1, iy1 = math.floor(scaled[0]), math.floor(scaled[1])
    ix2, iy2 = math.ceil(scaled[2]), math.ceil(scaled[3])
    cx1, cy1 = max(0, ix1), max(0, iy1)
    cx2, cy2 = min(image.width, ix2), min(image.height, iy2)
    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError("Vehicle box lies outside the image after clipping")
    crop = image.crop((cx1, cy1, cx2, cy2))
    original_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    retained_area = (cx2 - cx1) * (cy2 - cy1)
    return crop, {
        "source_bbox_xyxy": list(coords),
        "scaled_bbox_xyxy": list(scaled),
        "crop_bbox_xyxy_int": [cx1, cy1, cx2, cy2],
        "region_width": cx2 - cx1,
        "region_height": cy2 - cy1,
        "annotation_width": ann_width,
        "annotation_height": ann_height,
        "image_width": image.width,
        "image_height": image.height,
        "resized_relative_to_annotation": (image.width, image.height) != (ann_width, ann_height),
        "coordinate_scale_x": scale_x,
        "coordinate_scale_y": scale_y,
        "clipped_to_image": retained_area != original_area,
        "clipped_area_fraction": (1 - retained_area / original_area) if original_area else None,
    }


def vehicle_region_comparisons(original, stages, annotation):
    """Compare same annotated vehicle ROIs in original and each stage output."""
    if not annotation:
        return {"status": "annotations_unavailable", "annotation_count": 0,
                "valid_region_count": 0, "invalid_region_count": 0, "rows": []}
    if (annotation.get("width", 0) <= 0 or annotation.get("height", 0) <= 0):
        raise ValueError("Vehicle annotation has invalid image dimensions")
    rows = []
    eligible = [(number, obj) for number, obj in enumerate(annotation.get("annotations", []), start=1)
                if str(obj.get("class_name", "")).lower() in VEHICLE_REGION_CLASSES]
    ignored = len(annotation.get("annotations", [])) - len(eligible)
    valid_region_ids = set()
    invalid_region_ids = set()
    for object_index, obj in eligible:
        box = obj.get("bbox_xyxy")
        try:
            original_crop, geometry = _annotation_region(
                original, box, (annotation["width"], annotation["height"]))
        except (TypeError, ValueError) as exc:
            invalid_region_ids.add(object_index)
            rows.append({"object_index": object_index,
                         "class_name": obj.get("class_name", ""),
                         "status": "invalid_region", "reason": str(exc)})
            continue
        valid_region_ids.add(object_index)
        original_metrics = calculate_quality(original_crop)
        for stage in stages:
            image = stage["image"]
            row = {"object_index": object_index, "class_name": obj.get("class_name", ""),
                   "stage_index": stage["index"], "stage_name": stage["name"],
                   "algorithm_id": stage["algorithm_id"],
                   "output_kind": stage.get("output_kind", "photograph"), "status": "measured",
                   **geometry}
            if image.size != original.size:
                row.update({"status": "size_mismatch",
                            "reason": f"Stage image {image.size} does not match original {original.size}; ROI coordinates were not remapped."})
                rows.append(row)
                continue
            try:
                enhanced_crop, _ = _annotation_region(
                    image, box, (annotation["width"], annotation["height"]))
            except (TypeError, ValueError) as exc:
                row.update({"status": "invalid_region", "reason": str(exc)})
                rows.append(row)
                continue
            pair = compare_images(original_crop, enhanced_crop, stage.get("processing_ms"))
            row.update({"before_metrics": pair["before"], "after_metrics": pair["after"],
                        "metric_differences": pair["metric_differences"],
                        "changed_pixel_count": pair["changed_pixel_count"],
                        "changed_pixel_fraction": pair["changed_pixel_fraction"],
                        "detail_indicators": pair["detail_indicators"]})
            rows.append(row)
    measured = any(row.get("status") == "measured" for row in rows)
    if measured:
        status = "measured"
    elif not eligible:
        status = "no_vehicle_annotations" if ignored else "empty_annotations"
    elif not stages and valid_region_ids:
        status = "no_processed_stages"
    elif not valid_region_ids:
        status = "invalid_regions"
    else:
        status = "no_valid_stage_measurements"
    return {"status": status,
            "annotation_count": len(eligible), "ignored_non_vehicle_annotation_count": ignored,
            "valid_region_count": len(valid_region_ids),
            "invalid_region_count": len(invalid_region_ids),
            "rows": rows}


def evaluate_visibility(result, filename="image", source_metadata=None, annotation=None):
    """Build per-image and per-stage records from retained pipeline arrays."""
    result = normalize_pipeline_result(result)
    metadata = source_metadata or result.get("source_metadata") or result["plan"].get("source_metadata", {})
    original = result["original"]
    original_metrics = result["input_metrics"]
    stage_rows = [{
        "filename": filename, "category": _category(metadata), "source_dataset": metadata.get("source_dataset", ""),
        "stage_index": 0, "stage_name": "Original", "algorithm_id": "original", "status": "source",
        "processing_ms": None, "parameters": {}, "before_metrics": None,
        "after_metrics": original_metrics, "metric_differences": None,
        "changed_pixel_count": None, "changed_pixel_fraction": None,
        "detail_indicators": None,
    }]
    for stage in result["stages"]:
        pair = compare_images(stage["input_image"], stage["image"], stage.get("processing_ms"),
                              before_metrics=stage["before"], after_metrics=stage["after"])
        stage_rows.append({
            "filename": filename, "category": _category(metadata), "source_dataset": metadata.get("source_dataset", ""),
            "stage_index": stage["index"], "stage_name": stage["name"],
            "algorithm_id": stage["algorithm_id"], "status": stage.get("status", "executed"),
            "processing_ms": stage.get("processing_ms"), "parameters": stage.get("parameters", {}),
            "before_metrics": pair["before"], "after_metrics": pair["after"],
            "metric_differences": pair["metric_differences"],
            "changed_pixel_count": pair["changed_pixel_count"],
            "changed_pixel_fraction": pair["changed_pixel_fraction"],
            "detail_indicators": pair["detail_indicators"],
        })
    final_image = result["final"]
    final_photo_stage = next((stage for stage in reversed(result["stages"])
                              if stage.get("output_kind") != "diagnostic"), None)
    final_metrics_from_stage = final_photo_stage["after"] if final_photo_stage else original_metrics
    final_pair = compare_images(original, final_image,
                                before_metrics=original_metrics,
                                after_metrics=final_metrics_from_stage)
    final_metrics = final_pair["after"]
    selected = [stage for stage in result["stages"] if stage.get("output_kind") != "diagnostic"]
    final_stage_index = selected[-1]["index"] if selected else 0
    summary = {
        "filename": filename, "category": _category(metadata),
        "source_dataset": metadata.get("source_dataset", ""),
        "width": original.width, "height": original.height,
        "stage_count": len(result["stages"]),
        "final_photo_stage_index": final_stage_index,
        "operation_sequence": [stage["name"] for stage in selected],
        "operation_parameters": [{"operation": stage["name"], "algorithm_id": stage["algorithm_id"],
                                  "parameters": stage.get("parameters", {})} for stage in selected],
        "total_processing_ms": result.get("total_processing_ms"),
        "original_metrics": original_metrics, "final_metrics": final_metrics,
        "original_to_final_metric_differences": final_pair["metric_differences"],
        "changed_pixel_count": final_pair["changed_pixel_count"],
        "changed_pixel_fraction": final_pair["changed_pixel_fraction"],
        "detail_indicators": final_pair["detail_indicators"],
        "vehicle_annotation_status": "not_supplied",
        "vehicle_region_count": 0,
        "vehicle_annotation_count": 0,
        "invalid_vehicle_region_count": 0,
        "plate_annotation_status": "unavailable_no_plate_ground_truth",
        "conclusion_note": "Measurements and heuristic indicators do not by themselves demonstrate improved visibility; visual inspection and task-specific evaluation are required.",
        "reference_metrics_status": "unavailable_no_aligned_clear_reference",
    }
    regions = vehicle_region_comparisons(original, result["stages"], annotation)
    summary["vehicle_annotation_status"] = regions["status"]
    summary["vehicle_region_count"] = regions["valid_region_count"]
    summary["vehicle_annotation_count"] = regions["annotation_count"]
    summary["invalid_vehicle_region_count"] = regions["invalid_region_count"]
    return {"per_image": summary, "per_stage": stage_rows,
            "vehicle_regions": regions["rows"],
            "vehicle_annotation_status": regions["status"],
            "interpretation": {
                "measured_facts": "Pixel-derived statistics and elapsed processing times.",
                "heuristic_indicators": "Noise proxy, edge evidence, clipping warnings, and detail-risk conditions.",
                "not_established": "No universal visibility score, true noise-removal rate, plate readability, or detector accuracy is inferred.",
            }}


def build_dawn_annotation_index(annotation_root):
    """Read DAWN VOC XML files into exact filename/weather-category lookup."""
    root = Path(annotation_root)
    index = {}
    if not root.exists():
        return index
    for path in sorted(root.rglob("*.xml")):
        annotation = parse_voc_annotation(path)
        parts = [part.lower() for part in path.parts]
        category = next((name for key, name in (("fog", "fog"), ("rain", "rain"),
                                                 ("snow", "snow"), ("sand", "sandstorm"))
                         if any(key in part for part in parts)), "")
        index[(category, annotation["filename"])] = annotation
    return index


def find_dawn_annotation(index, filename, category=""):
    """Return an unambiguous annotation by filename and optional category."""
    category = str(category or "").lower()
    exact = index.get((category, filename)) if category else None
    if exact is not None:
        return exact
    if category:
        return None
    matches = [value for key, value in index.items() if isinstance(key, tuple) and key[1] == filename]
    return matches[0] if len(matches) == 1 else None


def visibility_reports(results, annotation_index=None):
    """Evaluate a batch and aggregate per-image final deltas by weather category."""
    reports = []
    for item in results:
        metadata = item.get("source_metadata", {})
        annotation = None
        if (annotation_index is not None and str(metadata.get("source_dataset", "")).lower() == "dawn"):
            annotation = find_dawn_annotation(annotation_index, item.get("filename", ""),
                                              metadata.get("category", ""))
        reports.append(evaluate_visibility(item, item.get("filename", "image"), metadata, annotation))
    categories = {}
    for report in reports:
        image = report["per_image"]
        category = image["category"]
        categories.setdefault(category, []).append(report)
    aggregate_rows = []
    for category, grouped in sorted(categories.items()):
        images = [report["per_image"] for report in grouped]
        final_deltas = [image["original_to_final_metric_differences"] for image in images]
        row = {"category": category, "image_count": len(images),
               "vehicle_region_comparison_count": sum(
                   1 for report in grouped for region in report["vehicle_regions"]
                   if region.get("status") == "measured" and region.get("stage_index") == report["per_image"]["final_photo_stage_index"])}
        for key in METRIC_LABELS.values():
            values = [d[key]["absolute"] for d in final_deltas if d.get(key, {}).get("absolute") is not None]
            row[f"mean_final_delta_{key}"] = statistics.fmean(values) if values else None
        for key in METRIC_LABELS.values():
            values = [region.get("metric_differences", {}).get(key, {}).get("absolute")
                      for report in grouped for region in report["vehicle_regions"]
                      if region.get("status") == "measured" and region.get("stage_index") == report["per_image"]["final_photo_stage_index"]]
            values = [value for value in values if value is not None]
            row[f"mean_final_vehicle_region_delta_{key}"] = statistics.fmean(values) if values else None
        aggregate_rows.append(row)
    return reports, aggregate_rows


def _csv_bytes(rows, fields):
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value
                         for key, value in row.items()})
    return stream.getvalue().encode("utf-8-sig")


def visibility_image_csv_bytes(reports):
    """Serialize one summary row per processed image."""
    rows = [report["per_image"] for report in reports]
    fields = ["filename", "category", "source_dataset", "width", "height", "stage_count",
              "operation_sequence", "operation_parameters", "total_processing_ms",
              "original_metrics", "final_metrics", "original_to_final_metric_differences",
              "changed_pixel_count", "changed_pixel_fraction", "detail_indicators",
              "vehicle_annotation_status", "vehicle_annotation_count", "vehicle_region_count",
              "invalid_vehicle_region_count", "plate_annotation_status",
              "reference_metrics_status", "conclusion_note"]
    return _csv_bytes(rows, fields)


def visibility_stage_csv_bytes(reports):
    """Serialize original plus each actual stage's paired measurements."""
    rows = [row for report in reports for row in report["per_stage"]]
    fields = ["filename", "category", "source_dataset", "stage_index", "stage_name", "algorithm_id",
              "status", "processing_ms", "parameters", "before_metrics", "after_metrics",
              "metric_differences", "changed_pixel_count", "changed_pixel_fraction", "detail_indicators"]
    return _csv_bytes(rows, fields)


def visibility_region_csv_bytes(reports):
    """Serialize annotated vehicle crop metrics; never represents plates."""
    rows = []
    for report in reports:
        image_meta = report["per_image"]
        for region in report["vehicle_regions"]:
            rows.append({"filename": image_meta["filename"], "category": image_meta["category"],
                         "source_dataset": image_meta["source_dataset"], **region})
    fields = ["filename", "category", "source_dataset", "object_index", "class_name", "stage_index",
              "stage_name", "algorithm_id", "output_kind", "status", "reason", "source_bbox_xyxy",
              "scaled_bbox_xyxy", "crop_bbox_xyxy_int", "region_width", "region_height",
              "annotation_width", "annotation_height", "image_width", "image_height",
              "resized_relative_to_annotation", "coordinate_scale_x", "coordinate_scale_y",
              "clipped_to_image", "clipped_area_fraction", "before_metrics", "after_metrics",
              "metric_differences", "changed_pixel_count", "changed_pixel_fraction", "detail_indicators"]
    return _csv_bytes(rows, fields)


def visibility_category_csv_bytes(rows):
    """Serialize descriptive means by supplied weather category."""
    fields = ["category", "image_count", "vehicle_region_comparison_count"]
    fields.extend(f"mean_final_delta_{key}" for key in METRIC_LABELS.values())
    fields.extend(f"mean_final_vehicle_region_delta_{key}" for key in METRIC_LABELS.values())
    return _csv_bytes(rows, fields)
