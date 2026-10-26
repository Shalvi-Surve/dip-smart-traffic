from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from analysis.quality import calculate_quality
from app import _conceptual_stage_cards, _photo_stages, _stage_impact_data, _stage_metric_rows
from dip.adaptive import build_pipeline, stage_view
from traffic_dip.batch import png_bytes, process_batch


def _test_image(offset=0):
    y, x = np.mgrid[:96, :96]
    pixels = np.stack((45 + (x // 12) * 3 + (y // 12) * 2,
                       50 + (x // 12) * 3, 65 + (y // 12) * 2), axis=-1)
    return Image.fromarray(np.clip(pixels + offset, 0, 255).astype(np.uint8))


def _png(image):
    stream = BytesIO()
    image.save(stream, format="PNG")
    return stream.getvalue()


def test_fixed_pipeline_runs_in_requested_order_and_callbacks_show_real_saved_outputs():
    source = _test_image()
    original_pixels = np.asarray(source).copy()
    selected = ["Gamma correction", "CLAHE on luminance", "Luminance unsharp mask"]
    events = []
    result = build_pipeline(source, denoise_choice="off", selected_operations=selected,
                            progress_callback=events.append)

    assert [stage["name"] for stage in result["stages"]] == selected
    assert result["plan"]["selection_mode"] == "fixed-order"
    assert result["plan"]["selected_operation_order"] == selected
    previous = result["original"]
    for stage in result["stages"]:
        assert np.array_equal(np.asarray(stage["input_image"]), np.asarray(previous))
        assert stage["changed_pixel_count"] > 0
        assert stage["changed_pixel_fraction"] == stage["changed_pixel_count"] / (source.width * source.height)
        previous = stage["image"]
    assert np.array_equal(np.asarray(result["final"]), np.asarray(result["stages"][-1]["image"]))
    assert np.array_equal(np.asarray(source), original_pixels)

    completed = [event for event in events if event["event"] == "stage_completed"]
    assert len(completed) == len(result["stages"])
    for event, stage in zip(completed, result["stages"]):
        assert np.array_equal(np.asarray(event["stage"]["image"]), np.asarray(stage["image"]))
    assert events[-1]["event"] == "complete"


def test_navigation_reads_saved_stages_without_reprocessing(monkeypatch):
    import dip.adaptive as adaptive
    calls = []
    real_gamma = adaptive.gamma_correct
    def counted_gamma(*args, **kwargs):
        calls.append(1)
        return real_gamma(*args, **kwargs)
    monkeypatch.setattr(adaptive, "gamma_correct", counted_gamma)
    result = build_pipeline(_test_image(), denoise_choice="off",
                            selected_operations=["Gamma correction", "CLAHE on luminance"])
    assert len(calls) == 1
    stage_pixels = [np.asarray(stage["image"]).copy() for stage in result["stages"]]
    for index in range(len(result["stages"]), 0, -1):
        assert np.array_equal(np.asarray(stage_view(result, index)["image"]), stage_pixels[index - 1])
    assert len(calls) == 1
    assert all(np.array_equal(np.asarray(stage["image"]), pixels)
               for stage, pixels in zip(result["stages"], stage_pixels))


def test_final_png_bytes_are_the_actual_last_stage_output():
    result = build_pipeline(_test_image(), denoise_choice="off",
                            selected_operations=["Gamma correction", "CLAHE on luminance"])
    encoded = Image.open(BytesIO(png_bytes(result["final"]))).convert("RGB")
    assert np.array_equal(np.asarray(encoded), np.asarray(result["stages"][-1]["image"]))


def test_conceptual_card_count_and_each_filter_panel_use_its_own_stage_data():
    result = build_pipeline(_test_image(), denoise_choice="off",
                            selected_operations=["Gamma correction", "CLAHE on luminance", "Denoising",
                                                 "Luminance unsharp mask", "Global histogram equalization"])
    stages = _photo_stages(result)
    cards = _conceptual_stage_cards(result["original"], stages)
    assert len(cards) == len(stages) + 1
    assert len(stages) == 5 and len(cards) == 6
    assert cards[0]["step"] == 1 and cards[0]["image"].tobytes() == result["original"].tobytes()
    for card, stage in zip(cards[1:], stages):
        assert card["image"].tobytes() == stage["image"].tobytes()
        impact = _stage_impact_data(stage)
        assert impact["before_image"].tobytes() == stage["input_image"].tobytes()
        assert impact["after_image"].tobytes() == stage["image"].tobytes()
        assert impact["before_metrics"] == calculate_quality(stage["input_image"])
        assert impact["after_metrics"] == calculate_quality(stage["image"])
        before = np.asarray(stage["input_image"].convert("RGB"))
        after = np.asarray(stage["image"].convert("RGB"))
        changed = int(np.any(before != after, axis=2).sum())
        assert impact["changed_pixel_count"] == changed
        assert impact["changed_pixel_fraction"] == changed / (before.shape[0] * before.shape[1])
        metric_rows = _stage_metric_rows(stage)
        brightness = next(row for row in metric_rows if row["Measurement"] == "Mean luminance")
        assert brightness["Before"].startswith(f"{stage['before']['mean_brightness']:.3f}")
        assert brightness["After"].startswith(f"{stage['after']['mean_brightness']:.3f}")


def test_normalizer_rejects_stage_pixel_comparisons_with_different_dimensions():
    from dip.adaptive import normalize_pipeline_result
    result = build_pipeline(_test_image(), denoise_choice="off",
                            selected_operations=["Gamma correction"])
    invalid = dict(result)
    invalid["stages"] = [dict(stage) for stage in result["stages"]]
    invalid["stages"][0]["image"] = Image.new("RGB", (4, 4))
    invalid["stages"][0]["output_image"] = invalid["stages"][0]["image"]
    with pytest.raises(ValueError, match="input/output dimensions differ"):
        normalize_pipeline_result(invalid)


def test_batch_uses_same_fixed_sequence_independently_and_reports_progress():
    first, second = _test_image(), _test_image(20)
    selected = ["Gamma correction", "CLAHE on luminance"]
    events = []
    batch = process_batch([("a.png", _png(first)), ("b.png", _png(second))],
                          denoise_choice="off", selected_operations=selected,
                          progress_callback=events.append)
    assert len(batch) == 2
    assert all([stage["name"] for stage in item["stages"]] == selected for item in batch)
    assert np.array_equal(np.asarray(batch[0]["original"]), np.asarray(first))
    assert np.array_equal(np.asarray(batch[1]["original"]), np.asarray(second))
    assert {event["image_index"] for event in events if "image_index" in event} == {1, 2}
    assert events[-1] == {"event": "batch_complete", "image_count": 2, "completed": 2, "failed": 0}
    for item in batch:
        for stage, previous in zip(item["stages"], [item["original"], *[s["image"] for s in item["stages"][:-1]]]):
            assert np.array_equal(np.asarray(stage["input_image"]), np.asarray(previous))
