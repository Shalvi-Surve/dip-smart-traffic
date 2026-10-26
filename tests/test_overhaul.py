import csv
from io import BytesIO, StringIO
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from analysis.quality import calculate_quality
from dip.adaptive import analyze_and_plan, build_pipeline, normalize_pipeline_result, stage_view
from dip.dehaze import dark_channel_prior
from traffic_dip.batch import batch_csv_bytes, batch_zip, process_batch
from traffic_dip.reporting import processing_report, report_json_bytes


def fog_scene(size=64):
    y, x = np.mgrid[0:size, 0:size]
    scene = np.zeros((size, size, 3), dtype=np.float32)
    scene[:, :, 0] = np.where((x // 8) % 2, 90, 35)
    scene[:, :, 1] = np.where((y // 8) % 2, 110, 45)
    scene[:, :, 2] = 60
    return Image.fromarray(np.clip(scene * .35 + .55 * 255, 0, 255).astype(np.uint8))


def png(image):
    out = BytesIO()
    image.save(out, "PNG")
    return out.getvalue()


def test_extended_metrics_are_measured_and_histogram_accounts_for_all_pixels():
    source = np.array([[0, 5, 128, 250], [255, 80, 90, 100]], dtype=np.uint8)
    metrics = calculate_quality(Image.fromarray(source))
    assert sum(metrics["luminance_histogram_32"]) == source.size
    assert metrics["clipped_shadow_fraction"] == pytest.approx(2 / 8)
    assert metrics["clipped_highlight_fraction"] == pytest.approx(2 / 8)
    assert 0 <= metrics["haze_like_indicator"] <= 1
    assert metrics["noise_estimate"] >= 0
    assert metrics["edge_fraction_gradient_ge_4"] >= metrics["edge_fraction_gradient_ge_12"]


def test_dark_channel_prior_returns_a_real_image_and_guards_parameters():
    image = fog_scene()
    enhanced, diagnostics = dark_channel_prior(image)
    assert enhanced.mode == "RGB" and enhanced.size == image.size
    assert not np.array_equal(np.asarray(enhanced), np.asarray(image))
    assert 0.2 <= diagnostics["transmission_min"] <= diagnostics["transmission_max"] <= 1
    with pytest.raises(ValueError):
        dark_channel_prior(image, window=4)
    with pytest.raises(ValueError):
        dark_channel_prior(Image.new("RGB", (2, 2)), window=3)


def test_haze_rule_selects_dehazing_and_clahe_is_skipped_as_redundant():
    plan = analyze_and_plan(fog_scene())
    actions = {a["operation"]: a for a in plan["actions"]}
    assert actions["Dark-channel-prior dehazing"]["selected"]
    assert actions["CLAHE on luminance"]["status"] == "skipped"
    assert "avoid stacking" in actions["CLAHE on luminance"]["reason"]
    assert actions["Rain-streak suppression"]["status"] == "unavailable"


def test_rule_engine_is_reproducible_and_records_skip_rationale():
    image = Image.new("RGB", (48, 48), (120, 120, 120))
    one = build_pipeline(image, denoise_choice="off")
    two = build_pipeline(image, denoise_choice="off")
    assert [a["parameters"] for a in one["plan"]["actions"]] == [a["parameters"] for a in two["plan"]["actions"]]
    assert np.array_equal(np.asarray(one["final"]), np.asarray(two["final"]))
    gamma = next(a for a in one["plan"]["actions"] if a["operation"] == "Gamma correction")
    assert gamma["status"] == "skipped" and "75–185" in gamma["reason"]
    assert one["plan"]["engine"] == "deterministic rule-based adaptive DIP"


def test_gamma_is_skipped_when_highlight_or_shadow_detail_is_clipped():
    for pixel, key in [(0, "clipped_shadow_fraction"), (255, "clipped_highlight_fraction")]:
        result = build_pipeline(Image.new("L", (24, 24), pixel), denoise_choice="off")
        action = next(a for a in result["plan"]["actions"] if a["operation"] == "Gamma correction")
        assert action["status"] == "skipped"
        assert "cannot recover" in action["reason"]
        assert result["input_metrics"][key] == 1.0


def test_stage_navigation_keeps_original_and_returns_matching_saved_stage():
    image = Image.new("L", (48, 48), 100)
    original_pixels = np.asarray(image).copy()
    result = build_pipeline(image, denoise_choice="off")
    assert np.array_equal(np.asarray(image), original_pixels)
    first = stage_view(result, 0)
    assert np.array_equal(np.asarray(first["image"]), np.asarray(image.convert("RGB")))
    for i, stage in enumerate(result["stages"], start=1):
        view = stage_view(result, i)
        assert view["name"] == stage["name"]
        assert np.array_equal(np.asarray(view["image"]), np.asarray(stage["image"]))
        assert view["metrics"] == stage["after"]
    with pytest.raises(IndexError):
        stage_view(result, len(result["stages"]) + 1)


def test_saved_stage_inputs_outputs_match_walkthrough_and_diagnostics_stay_auxiliary():
    source = fog_scene()
    original_pixels = np.asarray(source).copy()
    result = build_pipeline(source, edge_diagnostic=True, morphology_choice="closing")
    previous = result["original"]
    for stage in result["stages"]:
        assert np.array_equal(np.asarray(stage["input_image"]), np.asarray(previous))
        assert not np.array_equal(np.asarray(stage["image"]), np.asarray(stage["input_image"]))
        assert stage["before"]["width"] == stage["image"].width
        assert stage["after"]["height"] == stage["image"].height
        previous = stage["image"]
    diagnostics = [s for s in result["stages"] if s["output_kind"] == "diagnostic"]
    assert [s["name"] for s in diagnostics] == ["Sobel edge diagnostic", "Morphological diagnostic"]
    assert all(s["image"].mode == "L" for s in diagnostics)
    assert result["final"].mode == "RGB"
    assert np.array_equal(np.asarray(source), original_pixels)


def test_representative_fog_scene_selects_only_evidence_supported_photo_stages():
    result = build_pipeline(fog_scene(), source_metadata={"source_dataset": "DAWN", "category": "fog"})
    names = [stage["name"] for stage in result["stages"]]
    assert "Dark-channel-prior dehazing" in names
    assert "Robust luminance normalization" in names
    assert "Luminance unsharp mask" in names
    assert len(names) < 5
    statuses = {a["operation"]: a for a in result["plan"]["actions"]}
    assert statuses["Gamma correction"]["status"] == "skipped"
    assert statuses["Denoising"]["status"] == "skipped"
    assert statuses["CLAHE on luminance"]["status"] == "skipped"


def test_failed_processing_stage_is_marked_and_last_valid_image_survives(monkeypatch):
    import dip.adaptive as adaptive

    def fail(*args, **kwargs):
        raise RuntimeError("controlled test failure")

    monkeypatch.setattr(adaptive, "clahe_luminance", fail)
    source = Image.fromarray(np.tile(np.arange(40, dtype=np.uint8), (40, 1)))
    result = build_pipeline(source, denoise_choice="off")
    action = next(a for a in result["plan"]["actions"] if a["operation"] == "CLAHE on luminance")
    assert action["status"] == "failed"
    assert "controlled test failure" in action["error"]
    assert result["final"].size == source.size


def test_json_csv_and_zip_reports_preserve_category_and_actual_operations():
    metadata = {"source_dataset": "DAWN", "category": "fog", "license": "CC BY-NC 3.0"}
    image = fog_scene()
    raw = png(image)
    results = process_batch([("fog-scene.png", raw, metadata)], denoise_choice="off")
    report = processing_report(results[0], "fog-scene.png", metadata)
    assert report["source_metadata"]["category"] == "fog"
    assert report["executed_stages"]
    assert all(stage["algorithm_id"] and stage["name"] for stage in report["executed_stages"])
    decoded = json.loads(report_json_bytes(results[0], "fog-scene.png", metadata))
    assert "image" not in decoded["executed_stages"][0]
    assert "input_image" not in decoded["executed_stages"][0]
    rows = list(csv.DictReader(StringIO(batch_csv_bytes(results).decode("utf-8-sig"))))
    assert any(row["status"] == "executed" and row["category"] == "fog" for row in rows)
    with zipfile.ZipFile(BytesIO(batch_zip(results))) as archive:
        names = archive.namelist()
        assert "batch_stages.csv" in names and "batch_summary.json" in names
        assert "visibility_per_image.csv" in names
        assert "visibility_per_stage.csv" in names
        assert "visibility_vehicle_regions.csv" in names
        assert "visibility_by_category.csv" in names
        assert any("intermediate_" in name for name in names)
        item_log = next(n for n in names if n.endswith("/processing_log.json"))
        assert json.loads(archive.read(item_log))["source_metadata"]["category"] == "fog"


def test_real_pipeline_stages_have_ui_schema_and_serialize_after_legacy_cache_recovery():
    result = build_pipeline(fog_scene(), dehaze_choice="off", denoise_choice="off")
    assert result["stages"], "fixture must produce actual enhancement stages"
    legacy_cached = dict(result)
    legacy_cached["stages"] = [dict(stage) for stage in result["stages"]]
    # Reproduce a previous Streamlit session's cached records, where the plan
    # has the canonical identifier but an older stage record does not.
    for stage in legacy_cached["stages"]:
        stage.pop("algorithm_id")
    normalized = normalize_pipeline_result(legacy_cached)
    required_ui_fields = {"name", "algorithm_id", "image", "input_image", "parameters", "reason",
                          "before", "after", "processing_ms", "changed_pixel_fraction", "output_kind"}
    for index, stage in enumerate(normalized["stages"], start=1):
        assert required_ui_fields <= stage.keys()
        view = stage_view(normalized, index)
        assert view["stage"]["algorithm_id"] == stage["algorithm_id"]
        assert view["image"] is stage["image"]
    decoded = json.loads(report_json_bytes(normalized, "foggy-003.jpg"))
    assert all(stage["algorithm_id"] for stage in decoded["executed_stages"])


def test_missing_required_algorithm_identifier_has_clear_error():
    result = build_pipeline(fog_scene(), dehaze_choice="off", denoise_choice="off")
    broken = dict(result)
    broken["plan"] = {"actions": []}
    broken["stages"] = [dict(result["stages"][0])]
    broken["stages"][0].pop("algorithm_id")
    with pytest.raises(ValueError, match="required algorithm_id is missing"):
        normalize_pipeline_result(broken)


def test_batch_skips_invalid_file_and_records_failure_in_summary():
    valid = png(Image.new("RGB", (16, 16), (40, 80, 120)))
    results = process_batch([("broken.jpg", b"not an image"), ("valid.png", valid)], denoise_choice="off")
    assert [item["filename"] for item in results] == ["valid.png"]
    assert results.errors and results.errors[0]["filename"] == "broken.jpg"
    with zipfile.ZipFile(BytesIO(batch_zip(results))) as archive:
        manifest = json.loads(archive.read("batch_summary.json"))
        assert manifest["failed_files"][0]["filename"] == "broken.jpg"
