"""Pascal VOC parsing and fixed-threshold object-detection evaluation."""
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .detection import VEHICLE_CLASSES


def parse_voc_annotation(path):
    """Parse one DAWN Pascal VOC XML record into pixel-space xyxy boxes.

    The XML ``filename`` and dimensions are retained. Coordinates are returned
    as continuous ``(xmin, ymin, xmax, ymax)`` values exactly as stored in
    VOC; the paired DAWN YOLO annotations were audited and convert to the same
    coordinates. The source ``path`` field is deliberately ignored because it
    contains stale machine-local paths in the bundled dataset.
    """
    try:
        root = ET.parse(Path(path)).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Could not read Pascal VOC annotation {path}: {exc}") from exc
    filename = (root.findtext("filename") or "").strip()
    size = root.find("size")
    if not filename or size is None:
        raise ValueError(f"Pascal VOC annotation {path} is missing filename or size")
    try:
        width, height = int(size.findtext("width")), int(size.findtext("height"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Pascal VOC annotation {path} has invalid image dimensions") from exc
    if width <= 0 or height <= 0:
        raise ValueError(f"Pascal VOC annotation {path} has non-positive image dimensions")
    annotations = []
    for number, obj in enumerate(root.findall("object"), start=1):
        class_name = (obj.findtext("name") or "").strip().lower()
        box = obj.find("bndbox")
        if not class_name or box is None:
            raise ValueError(f"Pascal VOC annotation {path} object {number} is missing name or bndbox")
        try:
            coordinates = [float(box.findtext(key)) for key in ("xmin", "ymin", "xmax", "ymax")]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Pascal VOC annotation {path} object {number} has invalid coordinates") from exc
        x1, y1, x2, y2 = coordinates
        if not all(math.isfinite(value) for value in coordinates) or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Pascal VOC annotation {path} object {number} has an invalid xyxy box")
        annotations.append({"class_name": class_name, "bbox_xyxy": coordinates,
                            "difficult": (obj.findtext("difficult") or "0").strip() == "1",
                            "truncated": (obj.findtext("truncated") or "0").strip() == "1"})
    return {"filename": filename, "width": width, "height": height, "annotations": annotations}


def intersection_over_union(box_a, box_b):
    """Compute IoU for continuous pixel-space xyxy boxes."""
    if len(box_a) != 4 or len(box_b) != 4:
        raise ValueError("IoU boxes must each contain four xyxy coordinates")
    ax1, ay1, ax2, ay2 = map(float, box_a)
    bx1, by1, bx2, by2 = map(float, box_b)
    if not all(math.isfinite(v) for v in (*box_a, *box_b)):
        raise ValueError("IoU coordinates must be finite")
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _scores(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"true_positive": tp, "false_positive": fp, "false_negative": fn,
            "precision": precision, "recall": recall, "f1": f1}


def evaluate_detection_records(records, iou_threshold=0.5, class_names=VEHICLE_CLASSES):
    """Aggregate class-aware precision/recall/F1 at one fixed IoU threshold.

    Each record has ``predictions`` (detection dictionaries from
    :func:`traffic_dip.detection.detect_vehicles`) and ``ground_truth`` (the
    annotations returned by :func:`parse_voc_annotation`). Predictions are
    greedily matched in descending confidence within each image and class.
    Difficult VOC objects are excluded from the denominator and absorb an
    overlapping prediction at threshold IoU. This is not an AP/mAP protocol.
    """
    if not 0 < float(iou_threshold) <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    classes = tuple(str(name).lower() for name in class_names)
    if not classes or len(set(classes)) != len(classes):
        raise ValueError("class_names must be a non-empty sequence of unique names")
    totals = {name: [0, 0, 0] for name in classes}
    for record_index, record in enumerate(records, start=1):
        if not isinstance(record, dict) or "predictions" not in record or "ground_truth" not in record:
            raise ValueError(f"Evaluation record {record_index} requires predictions and ground_truth")
        gt_by_class = {name: [] for name in classes}
        difficult_by_class = {name: [] for name in classes}
        for gt in record["ground_truth"]:
            name = str(gt.get("class_name", "")).lower()
            if name not in gt_by_class:
                continue
            (difficult_by_class if gt.get("difficult", False) else gt_by_class)[name].append(gt["bbox_xyxy"])
        for name in classes:
            matched = set()
            predictions = sorted((p for p in record["predictions"]
                                  if str(p.get("class_name", "")).lower() == name),
                                 key=lambda p: float(p.get("confidence", 0)), reverse=True)
            for pred in predictions:
                box = pred.get("bbox_xyxy")
                if box is None or len(box) != 4:
                    raise ValueError(f"Evaluation prediction for {name} is missing a valid bbox_xyxy")
                candidates = [(intersection_over_union(box, gt), i)
                              for i, gt in enumerate(gt_by_class[name]) if i not in matched]
                best_iou, best_index = max(candidates, default=(0.0, None))
                if best_iou >= iou_threshold:
                    matched.add(best_index)
                    totals[name][0] += 1
                elif any(intersection_over_union(box, gt) >= iou_threshold for gt in difficult_by_class[name]):
                    continue
                else:
                    totals[name][1] += 1
            totals[name][2] += len(gt_by_class[name]) - len(matched)
    per_class = {name: _scores(*values) for name, values in totals.items()}
    aggregate = [sum(values[i] for values in totals.values()) for i in range(3)]
    return {"iou_threshold": float(iou_threshold), "class_names": list(classes),
            "per_class": per_class, "overall": _scores(*aggregate),
            "protocol": "Class-aware greedy one-to-one matching per image, confidence descending, fixed IoU threshold; not mAP."}
