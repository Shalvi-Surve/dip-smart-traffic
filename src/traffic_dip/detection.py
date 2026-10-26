"""Optional, local-only vehicle detection kept separate from classical DIP."""
from collections import Counter
from functools import lru_cache
import math
import os
from pathlib import Path
import time

from PIL import Image, ImageDraw

VEHICLE_CLASSES = ("car", "motorcycle", "bus", "truck")
COCO_VEHICLE_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
DEFAULT_MODEL_RELATIVE_PATH = Path("models") / "yolo26n.pt"


class DetectorUnavailableError(RuntimeError):
    """Raised when the optional runtime or local weights are unavailable."""


class DetectorInferenceError(RuntimeError):
    """Raised when model loading or inference fails after configuration checks."""


def configured_model_path(model_path=None):
    """Resolve an explicit path, TRAFFIC_YOLO_MODEL, or the local default."""
    value = model_path or os.environ.get("TRAFFIC_YOLO_MODEL")
    path = Path(value).expanduser() if value else DEFAULT_MODEL_RELATIVE_PATH
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[2] / path
    return path.resolve()


@lru_cache(maxsize=2)
def _load_model_cached(model_path):
    path = Path(model_path)
    # Never pass a model alias such as "yolo26n.pt" to Ultralytics: that form
    # may download weights implicitly. Only pre-existing local files are used.
    if not path.is_file():
        raise DetectorUnavailableError(
            f"YOLO weights are not installed at {path}. Install the optional AI requirements, "
            "place the pretrained weights there, or set TRAFFIC_YOLO_MODEL to a local .pt file."
        )
    if path.stat().st_size == 0:
        raise DetectorUnavailableError(f"YOLO weights file is empty: {path}")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise DetectorUnavailableError(
            "Ultralytics is not installed. Install the optional requirements-ai.txt; "
            "classical DIP processing remains available without it."
        ) from exc
    try:
        return YOLO(str(path))
    except Exception as exc:
        raise DetectorInferenceError(f"Could not load local YOLO weights {path}: {exc}") from exc


def clear_model_cache():
    """Clear cached model instances (primarily useful after changing weights)."""
    _load_model_cached.cache_clear()


def _as_list(value):
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return value


def _class_name(names, class_id):
    try:
        return names[class_id]
    except (KeyError, IndexError, TypeError):
        return None


def _vehicle_color(name):
    return {"car": (48, 205, 190), "motorcycle": (240, 191, 85),
            "bus": (163, 139, 255), "truck": (255, 133, 133)}[name]


def detect_vehicles(image, model_path=None, confidence=0.25, image_size=640, model=None):
    """Detect COCO cars, motorcycles, buses and trucks in one RGB PIL image.

    Returns ``detections`` with ``class_name``, ``class_id``, ``confidence``
    and pixel-space ``bbox_xyxy``; an independent ``annotated_image`` copy;
    per-class counts; inference milliseconds; and the local model path. Pass a
    model object only for controlled tests or advanced callers. Production
    calls cache the loaded model and always force CPU inference.
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Vehicle detection expects a decoded PIL image")
    if not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("confidence must be between 0 and 1")
    if int(image_size) < 32:
        raise ValueError("image_size must be at least 32 pixels")
    rgb = image.convert("RGB")
    path = configured_model_path(model_path)
    detector = model if model is not None else _load_model_cached(str(path))
    started = time.perf_counter()
    try:
        predictions = detector.predict(source=rgb, conf=float(confidence), imgsz=int(image_size),
                                       device="cpu", verbose=False)
    except Exception as exc:
        raise DetectorInferenceError(f"YOLO inference failed: {exc}") from exc
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    if not isinstance(predictions, (list, tuple)) or not predictions:
        raise DetectorInferenceError("YOLO returned no result record for the input image")
    prediction = predictions[0]
    names = getattr(prediction, "names", None) or getattr(detector, "names", None)
    boxes = getattr(prediction, "boxes", None)
    detections = []
    if boxes is not None:
        try:
            coords = _as_list(boxes.xyxy)
            scores = _as_list(boxes.conf)
            class_ids = _as_list(boxes.cls)
        except Exception as exc:
            raise DetectorInferenceError(f"YOLO returned malformed box data: {exc}") from exc
        for coordinate, score, class_value in zip(coords, scores, class_ids):
            class_id = int(class_value)
            class_name = str(_class_name(names, class_id) or COCO_VEHICLE_IDS.get(class_id, "")).lower()
            if class_name not in VEHICLE_CLASSES:
                continue
            if len(coordinate) != 4:
                continue
            x1, y1, x2, y2 = map(float, coordinate)
            if not all(math.isfinite(v) for v in (x1, y1, x2, y2, float(score))):
                continue
            x1, x2 = sorted((min(max(x1, 0.0), float(rgb.width)), min(max(x2, 0.0), float(rgb.width))))
            y1, y2 = sorted((min(max(y1, 0.0), float(rgb.height)), min(max(y2, 0.0), float(rgb.height))))
            if x2 <= x1 or y2 <= y1:
                continue
            detections.append({"class_name": class_name, "class_id": class_id,
                               "confidence": float(score), "bbox_xyxy": [x1, y1, x2, y2]})

    annotated = rgb.copy()
    draw = ImageDraw.Draw(annotated)
    for item in detections:
        color = _vehicle_color(item["class_name"])
        box = item["bbox_xyxy"]
        draw.rectangle(box, outline=color, width=max(2, round(min(rgb.size) / 320)))
        label = f"{item['class_name']} {item['confidence']:.2f}"
        bounds = draw.textbbox((box[0], box[1]), label)
        draw.rectangle(bounds, fill=color)
        draw.text((box[0], box[1]), label, fill=(10, 14, 18))
    counts = dict(Counter(item["class_name"] for item in detections))
    for name in VEHICLE_CLASSES:
        counts.setdefault(name, 0)
    return {"detections": detections, "counts": counts, "inference_ms": elapsed_ms,
            "annotated_image": annotated, "model_path": str(path),
            "confidence_threshold": float(confidence), "image_size": int(image_size)}


def compare_detection_results(original, enhanced):
    """Summarize count deltas without treating counts as accuracy measures."""
    for label, result in (("original", original), ("enhanced", enhanced)):
        if not isinstance(result, dict) or not isinstance(result.get("counts"), dict):
            raise ValueError(f"{label} detection result must contain a counts mapping")
    original_counts = {name: int(original["counts"].get(name, 0)) for name in VEHICLE_CLASSES}
    enhanced_counts = {name: int(enhanced["counts"].get(name, 0)) for name in VEHICLE_CLASSES}
    return {"original_counts": original_counts, "enhanced_counts": enhanced_counts,
            "count_delta": {name: enhanced_counts[name] - original_counts[name] for name in VEHICLE_CLASSES},
            "interpretation": "Detection-count differences are not ground-truth accuracy measurements."}
