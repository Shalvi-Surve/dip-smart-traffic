from pathlib import Path
import csv
from io import StringIO
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from traffic_dip.detection import (DetectorInferenceError, DetectorUnavailableError,
                                   compare_detection_results, detect_vehicles)
from traffic_dip.evaluation import (evaluate_detection_records, intersection_over_union,
                                    parse_voc_annotation)
from traffic_dip.reporting import detection_comparison_csv_bytes
from traffic_dip.detection import clear_model_cache


class FakeBoxes:
    xyxy = np.array([[-4, 4, 20, 25], [5, 5, 25, 19], [1, 1, 10, 10]], dtype=float)
    conf = np.array([0.91, 0.82, 0.99], dtype=float)
    cls = np.array([2, 7, 0], dtype=float)


class FakeResult:
    names = {0: "person", 2: "car", 7: "truck"}
    boxes = FakeBoxes()


class FakeModel:
    names = FakeResult.names

    def __init__(self, error=None):
        self.error = error
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return [FakeResult()]


def test_detection_filters_coco_classes_clamps_coordinates_and_draws_separately():
    source = Image.new("RGB", (30, 20), (20, 30, 40))
    source_pixels = np.asarray(source).copy()
    model = FakeModel()
    result = detect_vehicles(source, model=model)
    assert [d["class_name"] for d in result["detections"]] == ["car", "truck"]
    assert result["counts"] == {"car": 1, "motorcycle": 0, "bus": 0, "truck": 1}
    assert result["detections"][0]["bbox_xyxy"] == [0.0, 4.0, 20.0, 20.0]
    assert 0 <= result["detections"][0]["confidence"] <= 1
    assert result["annotated_image"] is not source
    assert not np.array_equal(np.asarray(result["annotated_image"]), source_pixels)
    assert result["inference_ms"] >= 0
    assert model.calls[0]["device"] == "cpu"
    assert np.array_equal(np.asarray(source), source_pixels)


def test_detection_requires_existing_weights_before_optional_import(tmp_path):
    with pytest.raises(DetectorUnavailableError, match="weights are not installed"):
        detect_vehicles(Image.new("RGB", (10, 10)), model_path=tmp_path / "missing.pt")


def test_local_model_is_cached_and_never_initialized_from_a_missing_alias(tmp_path, monkeypatch):
    weights = tmp_path / "local-model.pt"
    weights.write_bytes(b"test model fixture")
    loads = []

    def factory(path):
        loads.append(path)
        return FakeModel()

    monkeypatch.setitem(sys.modules, "ultralytics", SimpleNamespace(YOLO=factory))
    clear_model_cache()
    try:
        for _ in range(2):
            detect_vehicles(Image.new("RGB", (30, 20)), model_path=weights)
        assert loads == [str(weights.resolve())]
    finally:
        clear_model_cache()


def test_detection_wraps_inference_errors_and_validates_input():
    with pytest.raises(DetectorInferenceError, match="inference failed"):
        detect_vehicles(Image.new("RGB", (30, 20)), model=FakeModel(RuntimeError("boom")))
    with pytest.raises(ValueError, match="PIL image"):
        detect_vehicles(np.zeros((10, 10, 3), dtype=np.uint8), model=FakeModel())
    with pytest.raises(ValueError, match="between 0 and 1"):
        detect_vehicles(Image.new("RGB", (30, 20)), model=FakeModel(), confidence=1.1)


def test_comparison_summary_reports_count_differences_not_accuracy():
    before = {"counts": {"car": 2, "bus": 1}}
    after = {"counts": {"car": 3, "truck": 1}}
    summary = compare_detection_results(before, after)
    assert summary["count_delta"] == {"car": 1, "motorcycle": 0, "bus": -1, "truck": 1}
    assert "not ground-truth accuracy" in summary["interpretation"]


def test_batch_detection_csv_contains_real_counts_and_timings():
    summary = {"original_counts": {"car": 1, "motorcycle": 0, "bus": 0, "truck": 0},
               "enhanced_counts": {"car": 2, "motorcycle": 0, "bus": 0, "truck": 1},
               "count_delta": {"car": 1, "motorcycle": 0, "bus": 0, "truck": 1}}
    detection = {"counts": summary["original_counts"], "inference_ms": 18.5, "model_path": "local/model.pt"}
    enhanced = {"counts": summary["enhanced_counts"], "inference_ms": 16.2, "model_path": "local/model.pt"}
    row = {"filename": "fog-1.jpg", "source_metadata": {"category": "fog", "source_dataset": "DAWN"},
           "original_detection": detection, "enhanced_detection": enhanced, "summary": summary}
    rows = list(csv.DictReader(StringIO(detection_comparison_csv_bytes([row]).decode("utf-8-sig"))))
    assert rows[0]["category"] == "fog"
    assert rows[0]["original_inference_ms"] == "18.5"
    assert rows[0]["enhanced_car"] == "2" and rows[0]["delta_truck"] == "1"


def test_dawn_voc_annotation_parses_classes_size_and_xyxy_coordinates():
    root = Path(__file__).resolve().parents[1]
    record = parse_voc_annotation(root / "data/raw/dawn/images/Fog/Fog/Fog_PASCAL_VOC/foggy-003.xml")
    assert record["filename"] == "foggy-003.jpg"
    assert (record["width"], record["height"]) == (1280, 672)
    assert [item["class_name"] for item in record["annotations"]] == ["car"] * 4
    assert record["annotations"][0]["bbox_xyxy"] == [60.0, 220.0, 291.0, 486.0]
    assert intersection_over_union([60, 220, 291, 486], [60, 220, 291, 486]) == 1.0


def test_fixed_iou_evaluation_counts_class_aware_matches_and_duplicates():
    gt = [{"class_name": "car", "bbox_xyxy": [0, 0, 10, 10]},
          {"class_name": "bus", "bbox_xyxy": [20, 20, 30, 30]}]
    pred = [{"class_name": "car", "bbox_xyxy": [0, 0, 10, 10], "confidence": .9},
            {"class_name": "car", "bbox_xyxy": [0, 0, 10, 10], "confidence": .8},
            {"class_name": "truck", "bbox_xyxy": [20, 20, 30, 30], "confidence": .7}]
    report = evaluate_detection_records([{"predictions": pred, "ground_truth": gt}], iou_threshold=.5)
    assert report["per_class"]["car"]["true_positive"] == 1
    assert report["per_class"]["car"]["false_positive"] == 1
    assert report["per_class"]["bus"]["false_negative"] == 1
    assert report["per_class"]["truck"]["false_positive"] == 1
    assert report["overall"]["true_positive"] == 1
    assert "not mAP" in report["protocol"]
