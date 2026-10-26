import csv
from io import BytesIO, StringIO
import zipfile

import numpy as np
import pytest
from PIL import Image

from dip.adaptive import build_pipeline
from traffic_dip.batch import batch_zip
from traffic_dip.visibility import (
    _annotation_region, build_dawn_annotation_index, changed_pixel_statistics,
    compare_images, evaluate_visibility, find_dawn_annotation, metric_differences,
    vehicle_region_comparisons, visibility_category_csv_bytes,
    visibility_image_csv_bytes, visibility_region_csv_bytes,
    visibility_reports, visibility_stage_csv_bytes,
)


def test_metrics_use_consistent_pair_and_changed_rgb_pixel_definition():
    before = Image.new("RGB", (4, 3), (0, 0, 0))
    array = np.zeros((3, 4, 3), dtype=np.uint8)
    array[1, 2, 1] = 1
    after = Image.fromarray(array)
    pair = compare_images(before, after, processing_ms=2.5)
    assert pair["changed_pixel_count"] == 1
    assert pair["changed_pixel_fraction"] == pytest.approx(1 / 12)
    assert pair["total_pixels"] == 12
    assert pair["processing_ms"] == 2.5
    assert pair["metric_differences"]["mean_luminance"]["relative"] is None
    assert pair["detail_indicators"]["noise_note"].startswith("Median-filter residual MAD")


def test_metric_relative_change_handles_zero_and_missing_baselines():
    delta = metric_differences({"mean_brightness": 0}, {"mean_brightness": 10})
    assert delta["mean_luminance"] == {"absolute": 10.0, "relative": None}
    assert delta["luminance_stddev"] == {"absolute": None, "relative": None}


def test_image_size_mismatch_is_rejected_for_pair_and_changed_pixel_count():
    before = Image.new("L", (3, 3), 0)
    after = Image.new("RGB", (4, 3), 0)
    with pytest.raises(ValueError, match="Image-size mismatch"):
        compare_images(before, after)
    with pytest.raises(ValueError, match="Image-size mismatch"):
        changed_pixel_statistics(before, after)


def test_voc_roi_coordinates_scale_and_clip_with_geometry_recorded():
    image = Image.new("RGB", (10, 10), "white")
    crop, geometry = _annotation_region(image, [-1, -1, 3, 4], (5, 5))
    assert crop.size == (6, 8)
    assert geometry["resized_relative_to_annotation"] is True
    assert geometry["coordinate_scale_x"] == 2
    assert geometry["coordinate_scale_y"] == 2
    assert geometry["clipped_to_image"] is True
    assert geometry["region_width"] == 6 and geometry["region_height"] == 8


@pytest.mark.parametrize("box", [[1, 1, 1, 4], [20, 20, 22, 24], [0, float("nan"), 1, 2]])
def test_invalid_or_empty_vehicle_regions_are_rejected(box):
    with pytest.raises(ValueError):
        _annotation_region(Image.new("RGB", (10, 10)), box, (10, 10))


def test_roi_mapping_rejects_aspect_ratio_mismatch():
    with pytest.raises(ValueError, match="aspect ratios differ"):
        _annotation_region(Image.new("RGB", (12, 10)), [0, 0, 4, 4], (10, 10))


def test_empty_vehicle_annotations_are_reported_without_fabricated_regions():
    report = vehicle_region_comparisons(Image.new("RGB", (4, 4)), [],
                                        {"width": 4, "height": 4, "annotations": []})
    assert report == {"status": "empty_annotations", "annotation_count": 0,
                      "ignored_non_vehicle_annotation_count": 0, "valid_region_count": 0,
                      "invalid_region_count": 0, "rows": []}


def test_non_vehicle_voc_objects_are_not_counted_as_vehicle_regions():
    report = vehicle_region_comparisons(Image.new("RGB", (4, 4)), [],
                                        {"width": 4, "height": 4, "annotations": [
                                            {"class_name": "person", "bbox_xyxy": [0, 0, 2, 3]}]})
    assert report["status"] == "no_vehicle_annotations"
    assert report["annotation_count"] == 0
    assert report["ignored_non_vehicle_annotation_count"] == 1


def test_invalid_boxes_are_reported_but_not_counted_as_measured_regions():
    image = Image.new("RGB", (10, 10), "black")
    stage = {"image": image, "index": 1, "name": "stage", "algorithm_id": "test"}
    annotation = {"width": 10, "height": 10, "annotations": [
        {"class_name": "car", "bbox_xyxy": [20, 20, 21, 21]},
        {"class_name": "bus", "bbox_xyxy": [4, 4, 4, 8]},
    ]}
    report = vehicle_region_comparisons(image, [stage], annotation)
    assert report["status"] == "invalid_regions"
    assert report["annotation_count"] == 2
    assert report["valid_region_count"] == 0
    assert report["invalid_region_count"] == 2
    assert [row["status"] for row in report["rows"]] == ["invalid_region", "invalid_region"]

    pipeline = build_pipeline(image, dehaze_choice="off", selected_operations=["Gamma correction"])
    summary = evaluate_visibility(pipeline, "invalid.png", {}, annotation)["per_image"]
    assert summary["vehicle_annotation_status"] == "invalid_regions"
    assert summary["vehicle_annotation_count"] == 2
    assert summary["vehicle_region_count"] == 0
    assert summary["invalid_vehicle_region_count"] == 2


def test_vehicle_region_comparison_uses_same_original_coordinates_for_each_stage():
    original = Image.new("RGB", (8, 8), (20, 20, 20))
    enhanced = Image.new("RGB", (8, 8), (30, 30, 30))
    annotation = {"width": 8, "height": 8, "annotations": [
        {"class_name": "car", "bbox_xyxy": [2, 1, 7, 6]}]}
    stage = {"image": enhanced, "index": 1, "name": "Gamma", "algorithm_id": "gamma", "processing_ms": 1}
    report = vehicle_region_comparisons(original, [stage], annotation)
    row = report["rows"][0]
    assert row["status"] == "measured"
    assert row["crop_bbox_xyxy_int"] == [2, 1, 7, 6]
    assert row["region_width"] == 5 and row["region_height"] == 5
    assert row["changed_pixel_count"] == 25


def test_stage_size_mismatch_is_recorded_not_remapped_against_vehicle_ground_truth():
    original = Image.new("RGB", (8, 8), "black")
    stage = {"image": Image.new("RGB", (4, 4), "white"), "index": 1,
             "name": "bad resize", "algorithm_id": "resize", "processing_ms": 1}
    annotation = {"width": 8, "height": 8, "annotations": [
        {"class_name": "car", "bbox_xyxy": [1, 1, 5, 5]}]}
    rows = vehicle_region_comparisons(original, [stage], annotation)["rows"]
    assert rows[0]["status"] == "size_mismatch"
    assert "not remapped" in rows[0]["reason"]


def test_dawn_voc_annotation_index_matches_exact_weather_and_filename():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    index = build_dawn_annotation_index(root / "data/raw/dawn/images/Fog")
    annotation = find_dawn_annotation(index, "foggy-003.jpg", "fog")
    assert annotation["width"] == 1280 and annotation["height"] == 672
    assert len(annotation["annotations"]) == 4
    assert all(item["class_name"] == "car" for item in annotation["annotations"])
    assert find_dawn_annotation(index, "foggy-003.jpg", "rain") is None


def test_visibility_reports_and_csv_exports_have_per_image_stage_region_and_category_rows():
    pixels = np.tile(np.linspace(50, 150, 48, dtype=np.uint8), (40, 1))
    result = build_pipeline(Image.fromarray(pixels), dehaze_choice="off",
                            selected_operations=["Gamma correction"])
    result.update({"filename": "demo.png", "source_metadata": {
        "source_dataset": "DAWN", "category": "fog"}})
    annotation = {"width": 48, "height": 40, "annotations": [
        {"class_name": "car", "bbox_xyxy": [5, 5, 25, 25]}]}
    report = evaluate_visibility(result, "demo.png", result["source_metadata"], annotation)
    assert report["per_image"]["operation_sequence"] == ["Gamma correction"]
    assert report["per_image"]["reference_metrics_status"] == "unavailable_no_aligned_clear_reference"
    assert report["per_image"]["plate_annotation_status"] == "unavailable_no_plate_ground_truth"
    assert [row["stage_name"] for row in report["per_stage"]] == ["Original", "Gamma correction"]
    assert report["vehicle_regions"][0]["status"] == "measured"

    reports, categories = visibility_reports([result], {("fog", "demo.png"): annotation})
    assert categories[0]["category"] == "fog" and categories[0]["image_count"] == 1
    image_rows = list(csv.DictReader(StringIO(visibility_image_csv_bytes(reports).decode("utf-8-sig"))))
    stage_rows = list(csv.DictReader(StringIO(visibility_stage_csv_bytes(reports).decode("utf-8-sig"))))
    region_rows = list(csv.DictReader(StringIO(visibility_region_csv_bytes(reports).decode("utf-8-sig"))))
    category_rows = list(csv.DictReader(StringIO(visibility_category_csv_bytes(categories).decode("utf-8-sig"))))
    assert image_rows[0]["operation_sequence"] == '["Gamma correction"]'
    assert len(stage_rows) == 2 and stage_rows[1]["status"] == "executed"
    assert region_rows[0]["class_name"] == "car"
    assert category_rows[0]["image_count"] == "1"
    assert image_rows[0]["vehicle_annotation_count"] == "1"
    assert image_rows[0]["vehicle_region_count"] == "1"

    source = BytesIO()
    result["original"].save(source, format="PNG")
    result.update({"archive_stem": "001_demo", "original_bytes": source.getvalue()})
    with zipfile.ZipFile(BytesIO(batch_zip([result], annotation_index={
            ("fog", "demo.png"): annotation}))) as archive:
        names = archive.namelist()
        assert "visibility_per_image.csv" in names
        assert "visibility_per_stage.csv" in names
        assert "visibility_vehicle_regions.csv" in names
        assert "visibility_by_category.csv" in names
        region_data = archive.read("visibility_vehicle_regions.csv").decode("utf-8-sig")
        assert "car" in region_data and "measured" in region_data


def test_category_csv_header_is_stable_for_empty_and_populated_reports():
    empty = list(csv.reader(StringIO(visibility_category_csv_bytes([]).decode("utf-8-sig"))))[0]
    populated = list(csv.reader(StringIO(visibility_category_csv_bytes([{
        "category": "fog", "image_count": 1, "vehicle_region_comparison_count": 0,
    }]).decode("utf-8-sig"))))[0]
    assert empty == populated
