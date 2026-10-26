from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import app
import dip.adaptive
from dip.adaptive import build_pipeline, normalize_pipeline_result, stage_view
from traffic_dip.batch import process_batch


def sample_image():
    y, x = np.mgrid[0:64, 0:64]
    rgb = np.stack((35 + 2 * x, 45 + y, 70 + (x + y) // 2), axis=-1)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))


def encode_png(image):
    output = BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_app_import_uses_this_projects_adaptive_module():
    expected = (Path(app.ROOT) / "src" / "dip" / "adaptive.py").resolve()
    assert Path(dip.adaptive.__file__).resolve() == expected
    assert app.build_pipeline is build_pipeline
    assert callable(app.normalize_pipeline_result)


def test_each_real_stage_has_fields_used_by_single_and_batch_stage_views():
    image = sample_image()
    single = build_pipeline(image, dehaze_choice="off", denoise_choice="off")
    batch = process_batch([("sample.png", encode_png(image))], dehaze_choice="off", denoise_choice="off")
    assert single["stages"] and batch and batch[0]["stages"]
    for result in (single, batch[0]):
        result = normalize_pipeline_result(result)
        previous = result["original"]
        for index, stage in enumerate(result["stages"], start=1):
            assert stage["input_image"].tobytes() == previous.tobytes()
            assert stage["output_image"].tobytes() == stage["image"].tobytes()
            assert stage_view(result, index)["image"].tobytes() == stage["image"].tobytes()
            caption = app._stage_output_caption(stage)
            assert stage["algorithm_id"] in caption
            assert "pixels changed" in caption
            assert stage["parameters"] is not None
            assert stage["reason"]
            assert stage["before"] and stage["after"]
            assert stage["processing_ms"] is not None
            assert 0 <= stage["changed_pixel_fraction"] <= 1
            previous = stage["image"]


def test_normalizer_rejects_missing_required_output_image():
    result = build_pipeline(sample_image(), dehaze_choice="off", denoise_choice="off")
    broken = dict(result)
    broken["stages"] = [dict(result["stages"][0])]
    broken["stages"][0].pop("image")
    broken["stages"][0].pop("output_image")
    with pytest.raises(ValueError, match="required output image is missing"):
        normalize_pipeline_result(broken)
