from io import BytesIO
import json
import zipfile
import pytest
from PIL import Image
from analysis.quality import calculate_quality
from traffic_dip.batch import batch_zip, decode_image, process_batch
from traffic_dip.visibility import visibility_reports


def image_bytes(mode="RGB", color=None):
    color = color or ((30, 50, 70, 120) if mode == "RGBA" else 25 if mode == "L" else (30, 50, 70))
    im = Image.new(mode, (20, 16), color)
    out = BytesIO(); im.save(out, format="PNG"); return out.getvalue()


@pytest.mark.parametrize("mode", ["L", "RGBA", "RGB"])
def test_decode_and_batch_support_image_modes(mode):
    raw = image_bytes(mode)
    decoded = decode_image(raw)
    assert decoded.mode == "RGB"
    assert calculate_quality(decoded)["mode"] == mode
    results = process_batch([("same.png", raw), ("same.png", raw)], denoise_choice="off")
    assert len(results) == 2
    assert results[0]["archive_stem"] != results[1]["archive_stem"]


def test_invalid_image_is_rejected_and_zip_has_original_logs_and_manifest():
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    results = process_batch([("traffic.png", image_bytes())], denoise_choice="off")
    with zipfile.ZipFile(BytesIO(batch_zip(results))) as archive:
        names = archive.namelist()
        assert any(name.endswith("/original.png") for name in names)
        assert any(name.endswith("/final.png") for name in names)
        assert any(name.endswith("/processing_log.json") for name in names)
        manifest = json.loads(archive.read("batch_summary.json"))
        assert manifest["image_count"] == 1


def test_batch_zip_accepts_ui_four_argument_call_and_keeps_reports_and_artifacts():
    raw = image_bytes()
    metadata = {"source_dataset": "DAWN", "category": "fog"}
    results = process_batch([("traffic.png", raw, metadata)], denoise_choice="off",
                            dehaze_choice="off", selected_operations=["Gamma correction"])
    annotation_index = {("fog", "traffic.png"): {
        "filename": "traffic.png", "width": 20, "height": 16,
        "annotations": [{"class_name": "car", "bbox_xyxy": [2, 2, 16, 13]}],
    }}
    failures = [{"filename": "skipped.png", "error": "invalid fixture"}]
    visibility_evaluation = visibility_reports(results, annotation_index)

    # This matches the batch UI's four-argument invocation exactly.
    archive_bytes = batch_zip(results, failures, annotation_index, visibility_evaluation)

    with zipfile.ZipFile(BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
        prefix = results[0]["archive_stem"]
        assert f"{prefix}/original.png" in names
        assert f"{prefix}/intermediate_01_gamma_correction.png" in names
        assert f"{prefix}/final.png" in names
        assert f"{prefix}/processing_log.json" in names
        assert "batch_summary.json" in names
        assert "batch_stages.csv" in names
        assert "batch_errors.csv" in names
        assert {
            "visibility_per_image.csv", "visibility_per_stage.csv",
            "visibility_vehicle_regions.csv", "visibility_by_category.csv",
        }.issubset(names)
        summary = json.loads(archive.read("batch_summary.json"))
        assert summary["failed_files"] == failures
        assert "car" in archive.read("visibility_vehicle_regions.csv").decode("utf-8-sig")


def test_batch_size_limit_is_reported():
    from traffic_dip.batch import MAX_BATCH_IMAGES
    with pytest.raises(ValueError, match="at most"):
        process_batch([(f"{i}.png", image_bytes()) for i in range(MAX_BATCH_IMAGES + 1)])
