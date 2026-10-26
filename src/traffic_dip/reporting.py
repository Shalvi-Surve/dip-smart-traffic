"""Structured JSON and CSV reporting without image payloads or fabricated metrics."""
import csv
from io import StringIO
import json
from dip.adaptive import normalize_pipeline_result
from .detection import VEHICLE_CLASSES


def processing_report(result, filename, source_metadata=None):
    result = normalize_pipeline_result(result)
    plan = result["plan"]
    return {
        "schema_version": "1.0",
        "filename": filename,
        "source_metadata": source_metadata or plan.get("source_metadata", {}),
        "engine": plan["engine"],
        "processing_time_ms": result["total_processing_ms"],
        "input_metrics": result["input_metrics"],
        "degradation_assessment": {"flags": result["analysis"]["flags"], "evidence": result["analysis"]["evidence"]},
        "processing_plan": plan["actions"],
        "executed_stages": [{k: v for k, v in stage.items() if k not in {"image", "input_image", "output_image"}} for stage in result["stages"]],
        "limitations": [a["limitation"] for a in plan["actions"] if a.get("limitation")],
        "reference_metrics": "PSNR/SSIM omitted: no aligned clear-weather reference is supplied.",
    }


def report_json_bytes(result, filename, source_metadata=None):
    return json.dumps(processing_report(result, filename, source_metadata), indent=2, ensure_ascii=False).encode("utf-8")


def batch_csv_bytes(results):
    stream = StringIO(newline="")
    fields = ["filename", "category", "source_dataset", "stage", "status", "reason", "parameters",
              "processing_ms", "brightness_before", "brightness_after", "contrast_before", "contrast_after",
              "sharpness_before", "sharpness_after"]
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for item in results:
        item = normalize_pipeline_result(item)
        metadata = item.get("source_metadata", {})
        stages_by_name = {stage["name"]: stage for stage in item["stages"]}
        for action in item["plan"]["actions"]:
            stage = stages_by_name.get(action["operation"], {})
            before, after = stage.get("before", {}), stage.get("after", {})
            writer.writerow({"filename": item["filename"], "category": metadata.get("category", ""),
                             "source_dataset": metadata.get("source_dataset", ""), "stage": action["operation"],
                             "status": action["status"], "reason": action["reason"],
                             "parameters": json.dumps(action.get("parameters", {}), ensure_ascii=False),
                             "processing_ms": stage.get("processing_ms", action.get("processing_ms", "")),
                             "brightness_before": before.get("mean_brightness", ""),
                             "brightness_after": after.get("mean_brightness", ""),
                             "contrast_before": before.get("contrast_stddev", ""),
                             "contrast_after": after.get("contrast_stddev", ""),
                             "sharpness_before": before.get("sharpness_laplacian_variance", ""),
                             "sharpness_after": after.get("sharpness_laplacian_variance", "")})
    return stream.getvalue().encode("utf-8-sig")


def detection_comparison_csv_bytes(rows):
    """Serialize real original/final vehicle counts and inference times."""
    stream = StringIO(newline="")
    fields = ["filename", "category", "source_dataset", "model_path",
              "original_inference_ms", "enhanced_inference_ms"]
    for prefix in ("original", "enhanced", "delta"):
        fields.extend(f"{prefix}_{name}" for name in VEHICLE_CLASSES)
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    for item in rows:
        source = item["original_detection"]
        enhanced = item["enhanced_detection"]
        summary = item["summary"]
        metadata = item.get("source_metadata", {})
        row = {"filename": item["filename"], "category": metadata.get("category", ""),
               "source_dataset": metadata.get("source_dataset", ""), "model_path": source.get("model_path", ""),
               "original_inference_ms": source["inference_ms"],
               "enhanced_inference_ms": enhanced["inference_ms"]}
        for name in VEHICLE_CLASSES:
            row[f"original_{name}"] = summary["original_counts"][name]
            row[f"enhanced_{name}"] = summary["enhanced_counts"][name]
            row[f"delta_{name}"] = summary["count_delta"][name]
        writer.writerow(row)
    return stream.getvalue().encode("utf-8-sig")
