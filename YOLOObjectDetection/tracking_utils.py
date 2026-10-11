"""Tracking configuration and JSON contracts shared with future pipeline stages."""

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
import re


@contextmanager
def tracking_config(profile, buffer=None):
    """Use installed defaults for baseline; materialize overrides for this run only."""
    from ultralytics.utils import ROOT, YAML

    root = Path(__file__).resolve().parent
    path = ROOT / "cfg/trackers/botsort.yaml" if profile == "default" else root / "botsort.yaml"
    config = YAML.load(path)
    if profile == "buffer":
        config["with_reid"] = False
    if buffer is not None:
        config["track_buffer"] = buffer
    # A separate effective file prevents changing source configuration between runs.
    with TemporaryDirectory(prefix=".tracking-", dir=root) as folder:
        effective = Path(folder) / "botsort.yaml"
        YAML.save(effective, config)
        yield str(effective), config


def frame_record(result, frame_index, timestamp, fps, latency_ms, label_cache):
    """Export pixel boxes and session IDs; no inferred distance or identity relabeling."""
    objects = []
    boxes = result.boxes
    if boxes is not None:
        boxes = boxes.cpu().numpy()
        ids = boxes.id.tolist() if boxes.is_track else [None] * len(boxes)
        for xyxy, class_id, confidence, track_id in zip(
            boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), ids
        ):
            class_name = result.names[int(class_id)]
            numeric_id = int(track_id) if track_id is not None else None
            if numeric_id is not None:
                # Keep the first label for this numeric ID if the detector's class fluctuates.
                label_cache.setdefault(numeric_id, re.sub(r"\W+", "_", class_name).strip("_") + f"_{numeric_id}")
            objects.append({
                "class": class_name,
                "class_id": int(class_id),
                "track_id": numeric_id,
                "object_id": label_cache.get(numeric_id),
                "confidence": float(confidence),
                "bbox_xyxy": xyxy,
            })
    return {
        "type": "frame", "frame_index": frame_index, "timestamp_seconds": timestamp,
        "fps": fps, "inference_tracking_ms": latency_ms,
        "tracked_count": sum(item["track_id"] is not None for item in objects),
        "objects": objects,
    }


def draw_debug(cv2, image, record, profile, config):
    """Add class-prefixed IDs and a compact debug overlay to normal YOLO annotations."""
    for item in record["objects"]:
        if item["object_id"]:
            x, y = (int(value) for value in item["bbox_xyxy"][:2])
            cv2.putText(image, f"{item['object_id']} {item['confidence']:.2f}",
                        (max(0, x), max(16, y + 18)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 255), 1)
    lines = [f"FPS: {record['fps']:.1f} | tracked: {record['tracked_count']}",
             f"{profile} | buffer={config.get('track_buffer', '-')} | ReID={config.get('with_reid', False)}"]
    for index, line in enumerate(lines):
        cv2.putText(image, line, (10, 25 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (0, 255, 0), 2)
    return image
