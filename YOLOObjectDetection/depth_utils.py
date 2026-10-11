"""Depth Anything V2 helpers: per-frame depth maps and per-object distance.

Modes:
    indoor / outdoor  metric models, depth in meters
    relative          relative model, value is "proximity" 0-1 (1 = nearest in this frame)

Each object in a frame record gets:
    "depth": {"value": float, "unit": "m" | "relative"}
    "depth_rank": 0 for the nearest object, 1 for the next, ...
"""

import cv2
import numpy as np
import torch
from transformers import AutoModelForDepthEstimation

RELATIVE_MODELS = {
    "small": "depth-anything/Depth-Anything-V2-Small-hf",
    "base": "depth-anything/Depth-Anything-V2-Base-hf",
    "large": "depth-anything/Depth-Anything-V2-Large-hf",
}
METRIC_MODELS = {
    (scene, size): f"depth-anything/Depth-Anything-V2-Metric-{scene.title()}-{size.title()}-hf"
    for scene in ("indoor", "outdoor")
    for size in ("small", "base", "large")
}

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
METRIC_DISTANCE_OFFSET_M = 1.0


def _torch_device(device):
    """Accept 0, "0", "cuda:0", or "cpu" (the same values used for YOLO)."""
    text = str(device)
    if text == "cpu":
        return torch.device("cpu")
    if text.isdigit():
        return torch.device(f"cuda:{text}")
    return torch.device(text)


class DepthEstimator:
    def __init__(self, size="small", mode="indoor", device=0, half=True, input_size=518):
        if mode == "relative":
            model_id = RELATIVE_MODELS[size]
        else:
            model_id = METRIC_MODELS[(mode, size)]
        self.model_id, self.mode, self.input_size = model_id, mode, input_size
        self.device = _torch_device(device)
        self.dtype = torch.float16 if (half and self.device.type == "cuda") else torch.float32
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(model_id)
            .to(self.device, self.dtype)
            .eval()
        )
        self._mean = torch.tensor(MEAN, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self._std = torch.tensor(STD, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)

    @torch.inference_mode()
    def predict(self, frame_bgr):
        """Return a float32 depth map at model resolution (not upscaled to the frame)."""
        h, w = frame_bgr.shape[:2]
        scale = self.input_size / max(h, w)
        new_w = max(14, round(w * scale / 14) * 14)  # DINOv2 patch size is 14
        new_h = max(14, round(h * scale / 14) * 14)
        rgb = cv2.cvtColor(cv2.resize(frame_bgr, (new_w, new_h)), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(rgb).to(self.device).permute(2, 0, 1).unsqueeze(0)
        x = (x.to(self.dtype) / 255.0 - self._mean) / self._std
        depth = self.model(pixel_values=x).predicted_depth  # (1, h, w)
        depth_map = depth[0].float().cpu().numpy()
        if self.mode != "relative":
            depth_map = np.maximum(depth_map - METRIC_DISTANCE_OFFSET_M, 0.0)
        return depth_map


def object_depth(depth_map, bbox_xyxy, frame_shape, shrink=0.5):
    """Median depth in the central part of a box (less background than the full box)."""
    dh, dw = depth_map.shape
    fh, fw = frame_shape[:2]
    sx, sy = dw / fw, dh / fh
    x1, y1, x2, y2 = bbox_xyxy
    cx, cy = (x1 + x2) / 2 * sx, (y1 + y2) / 2 * sy
    half_w, half_h = (x2 - x1) * sx * shrink / 2, (y2 - y1) * sy * shrink / 2
    xa = int(np.clip(np.floor(cx - half_w), 0, dw - 1))
    ya = int(np.clip(np.floor(cy - half_h), 0, dh - 1))
    xb = int(np.clip(np.ceil(cx + half_w), xa + 1, dw))
    yb = int(np.clip(np.ceil(cy + half_h), ya + 1, dh))
    return float(np.median(depth_map[ya:yb, xa:xb]))


def add_depth(record, depth_map, frame_shape, mode):
    """Attach depth and depth_rank to every object in a frame record (in place)."""
    metric = mode != "relative"
    lo, hi = np.percentile(depth_map, [2, 98])
    values = []
    for item in record["objects"]:
        value = object_depth(depth_map, item["bbox_xyxy"], frame_shape)
        if metric:
            item["depth"] = {"value": round(value, 3), "unit": "m"}
        else:
            proximity = float(np.clip((value - lo) / (hi - lo + 1e-8), 0, 1))
            item["depth"] = {"value": round(proximity, 3), "unit": "relative"}
        values.append(item["depth"]["value"])
    # Nearest first: smallest meters, or largest proximity.
    order = sorted(range(len(values)), key=lambda i: values[i] if metric else -values[i])
    for rank, index in enumerate(order):
        record["objects"][index]["depth_rank"] = rank
    record["depth_mode"] = mode
    return record


# Strip straight ahead of the user, as fractions of the depth map (columns, rows).
# At 2 m with a 70 degree camera the central 20% of the width is about 0.56 m wide.
CORRIDOR_COLS = (0.40, 0.60)
CORRIDOR_ROWS = (0.20, 0.65)  # upper-middle band: skips most of the floor in the lower part


def corridor_min_depth(depth_map, mode, cols=CORRIDOR_COLS, rows=CORRIDOR_ROWS, percentile=5):
    """Nearest depth (meters) in the strip straight ahead, with no class needed.

    Uses the 5th percentile instead of the absolute minimum so one noisy pixel cannot
    trigger it. Returns None for relative depth, which has no meters. It will miss
    obstacles below the band (low tables, curbs) because we do not model the floor.
    """
    if mode == "relative":
        return None
    dh, dw = depth_map.shape
    strip = depth_map[int(dh * rows[0]):max(int(dh * rows[1]), int(dh * rows[0]) + 1),
                      int(dw * cols[0]):max(int(dw * cols[1]), int(dw * cols[0]) + 1)]
    value = float(np.percentile(strip, percentile))
    return round(value, 3) if np.isfinite(value) and value >= 0 else None


def colorize(depth_map, frame_shape, mode):
    """Colorized depth image at frame size. Near = bright/warm."""
    values = depth_map if mode == "relative" else -depth_map
    lo, hi = np.percentile(values, [2, 98])
    norm = np.clip((values - lo) / (hi - lo + 1e-8), 0, 1)
    color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
    return cv2.resize(color, (frame_shape[1], frame_shape[0]))


def draw_depth(image, record):
    """Draw a rank + distance badge inside each box; the nearest object (#0) is red and boxed."""
    height, width = image.shape[:2]
    font, scale, thickness, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2, 6
    for item in record["objects"]:
        depth = item.get("depth")
        if not depth:
            continue
        rank = item.get("depth_rank")
        distance = f"{depth['value']:.1f} m" if depth["unit"] == "m" else f"near {depth['value']:.2f}"
        label = f"#{rank}  {distance}"
        x1, y1, x2, y2 = (int(v) for v in item["bbox_xyxy"])
        nearest = rank == 0
        if nearest:
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 3)

        (text_w, text_h), baseline = cv2.getTextSize(label, font, scale, thickness)
        badge_x2 = min(width - 1, x2)
        badge_x1 = max(0, badge_x2 - text_w - 2 * pad)
        badge_y1 = min(max(0, y1), height - 1)
        badge_y2 = min(height - 1, badge_y1 + text_h + baseline + 2 * pad)
        color = (0, 0, 255) if nearest else (70, 70, 70)  # BGR
        cv2.rectangle(image, (badge_x1, badge_y1), (badge_x2, badge_y2), color, -1)
        cv2.putText(image, label, (badge_x1 + pad, badge_y1 + pad + text_h),
                    font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image
