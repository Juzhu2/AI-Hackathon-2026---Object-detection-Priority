"""Geometry fuser: turn detections + metric depth + camera FOV into real-world geometry.

Every object in a frame record gets item["geometry"], a dict that is ALWAYS present:

    "ok":            True if metric depth was usable, False otherwise (all numbers below are then None)
    "x_m":           right of the camera axis (+) / left (-)
    "y_m":           above the camera axis (+) / below (-)
    "z_m":           forward distance along the optical axis (smoothed for tracked objects)
    "range_m":       straight-line distance sqrt(x^2 + y^2 + z^2)
    "bearing_deg":   horizontal angle, + = right of center
    "elevation_deg": vertical angle, + = above center
    "approach_mps":  closing speed, + = getting closer. None until the track has
                     `min_approach_frames` frames of history, and None if untracked.
    "age_frames":    frames this track has been seen (None if untracked)
    "clipped_edges": frame edges the box touches, e.g. ["right"] (depth less reliable)

The record also gets "n_detected" and "n_detected_no_geometry" so a detection whose depth
failed is never silently lost. "depth_rank" is recomputed from the smoothed range_m
(0 = nearest); objects without geometry get rank None.

Smoothing, per tracked object (all in frames, not seconds):
    1. median of the last `window` raw depths   -> rejects single-frame spikes
    2. exponential moving average, weight `alpha` on the new median -> z_m
    3. approach_mps = EMA (same alpha) of d(z_m)/dt, using the frame timestamps

Assumes a pinhole camera with square pixels. Depth Anything's metric output is the
distance along the optical axis (Z), not the radial range, which is how it is used here.

Offline use on a saved session:
    python geometry_fuser.py session.jsonl --table
    python geometry_fuser.py session.jsonl clean.json --clean --hfov 70 --width 1280 --height 720
"""

import argparse
import json
import math
import statistics
from collections import deque


class GeometryFuser:
    def __init__(self, hfov_deg=70.0, window=5, alpha=0.4, max_age=30, edge_margin=8,
                 min_approach_frames=4):
        """
        hfov_deg:            camera horizontal field of view. Typical webcams are 65-78 deg.
        window:              median window (frames) used to reject depth spikes per track.
        alpha:               EMA weight for new data (higher = faster, noisier).
        max_age:             forget a track after this many frames without seeing it.
        edge_margin:         pixels from the frame border that count as "touching" it.
        min_approach_frames: approach_mps stays None until a track has this many frames.
        """
        self.hfov_deg = hfov_deg
        self.window, self.alpha, self.max_age = window, alpha, max_age
        self.edge_margin, self.min_approach_frames = edge_margin, min_approach_frames
        self.tracks = {}

    def describe(self):
        """Settings as a dict, for the metadata header of a log file."""
        return {"median_window_frames": self.window, "ema_alpha": self.alpha,
                "min_approach_frames": self.min_approach_frames,
                "forget_track_after_frames": self.max_age, "edge_margin_px": self.edge_margin,
                "method": "median(window) -> EMA(alpha) on z; approach = EMA(alpha) of -dz/dt"}

    def update(self, record, frame_shape):
        """Add geometry to every object in `record` (in place). frame_shape = (h, w, ...)."""
        h, w = frame_shape[:2]
        fx = (w / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        fy = fx  # square pixels
        cx, cy = w / 2, h / 2
        t, index = record["timestamp_seconds"], record["frame_index"]
        no_geometry = 0

        for item in record["objects"]:
            x1, y1, x2, y2 = item["bbox_xyxy"]
            m = self.edge_margin
            clipped = [name for name, hit in (("left", x1 <= m), ("right", x2 >= w - m),
                                              ("top", y1 <= m), ("bottom", y2 >= h - m)) if hit]
            track_id = item.get("track_id")
            depth = item.get("depth")
            usable = (depth is not None and depth.get("unit") == "m"
                      and math.isfinite(depth["value"]) and depth["value"] > 0)

            if not usable:
                no_geometry += 1
                item["depth_rank"] = None
                item["geometry"] = {
                    "ok": False, "x_m": None, "y_m": None, "z_m": None, "range_m": None,
                    "bearing_deg": None, "elevation_deg": None, "approach_mps": None,
                    "age_frames": self.tracks[track_id]["n"] if track_id in self.tracks else None,
                    "clipped_edges": clipped,
                }
                continue

            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            z, approach, age = depth["value"], None, None
            if track_id is not None:
                z, approach, age = self._smooth(track_id, z, t, index)
                if age < self.min_approach_frames:
                    approach = None  # not enough history: "unknown", not "steady"

            x = (u - cx) * z / fx
            y = -(v - cy) * z / fy
            item["geometry"] = {
                "ok": True,
                "x_m": round(x, 3),
                "y_m": round(y, 3),
                "z_m": round(z, 3),
                "range_m": round(math.sqrt(x * x + y * y + z * z), 3),
                "bearing_deg": round(math.degrees(math.atan2(x, z)), 1),
                "elevation_deg": round(math.degrees(math.atan2(y, z)), 1),
                "approach_mps": None if approach is None else round(approach, 3),
                "age_frames": age,
                "clipped_edges": clipped,
            }

        # Re-rank on the smoothed straight-line range so depth_rank and geometry agree
        # (add_depth ranks on raw per-frame depth, which can flip on noise).
        fused = [item for item in record["objects"] if item["geometry"]["ok"]]
        for rank, item in enumerate(sorted(fused, key=lambda i: i["geometry"]["range_m"])):
            item["depth_rank"] = rank

        record["n_detected"] = len(record["objects"])
        record["n_detected_no_geometry"] = no_geometry

        # Drop tracks that have disappeared.
        stale = [k for k, s in self.tracks.items() if index - s["seen"] > self.max_age]
        for key in stale:
            del self.tracks[key]
        return record

    def _smooth(self, track_id, z_raw, t, index):
        state = self.tracks.get(track_id)
        if state is None:
            state = {"hist": deque(maxlen=self.window), "z": None, "t": t, "v": 0.0,
                     "seen": index, "n": 0}
            self.tracks[track_id] = state
        state["n"] += 1
        state["hist"].append(z_raw)
        median = statistics.median(state["hist"])  # kills single-frame spikes
        if state["z"] is None:
            state["z"] = median
        else:
            new_z = self.alpha * median + (1 - self.alpha) * state["z"]
            dt = t - state["t"]
            if dt > 1e-3:
                instant = (state["z"] - new_z) / dt  # positive = approaching
                state["v"] = self.alpha * instant + (1 - self.alpha) * state["v"]
            state["z"] = new_z
        state["t"], state["seen"] = t, index
        return state["z"], state["v"], state["n"]


def direction_word(bearing_deg):
    if bearing_deg is None:
        return None
    side = "right" if bearing_deg > 0 else "left"
    size = abs(bearing_deg)
    if size < 10:
        return "ahead"
    return f"slightly {side}" if size < 25 else side


def clean_record(record):
    """Compact, rounded view of one fused frame. Every detection is included.

    Objects whose depth failed appear with geometry_ok false and null numbers, after the
    ranked ones. Every frame yields a record, even with no objects.
    """
    ranked = sorted((o for o in record["objects"] if o["geometry"]["ok"]),
                    key=lambda o: o["depth_rank"])
    unranked = [o for o in record["objects"] if not o["geometry"]["ok"]]
    objects = []
    for item in ranked + unranked:
        geo = item["geometry"]
        objects.append({
            "id": item["track_id"], "class": item["class"], "source": item.get("source"),
            "conf": round(item["confidence"], 2), "rank": item.get("depth_rank"),
            "geometry_ok": geo["ok"], "age_frames": geo["age_frames"],
            "range_m": geo["range_m"], "x_m": geo["x_m"], "y_m": geo["y_m"], "z_m": geo["z_m"],
            "bearing_deg": geo["bearing_deg"], "direction": direction_word(geo["bearing_deg"]),
            "approach_mps": geo["approach_mps"], "clipped": geo["clipped_edges"],
        })
    fps = record.get("fps")
    return {
        "frame": record["frame_index"],
        "t": round(record["timestamp_seconds"], 2),
        "capture_unix_s": record.get("capture_unix_s"),
        "capture_monotonic_s": record.get("capture_monotonic_s"),
        "latency_ms": None if record.get("latency_ms") is None else round(record["latency_ms"], 1),
        "fps": None if fps is None else round(fps, 1),
        "n_detected": record.get("n_detected", len(objects)),
        "n_detected_no_geometry": record.get("n_detected_no_geometry", len(unranked)),
        "corridor_min_z_m": record.get("corridor_min_z_m"),
        "objects": objects,
    }


def format_table(record):
    """Human-readable table of one fused frame."""
    clean = clean_record(record)
    corridor = clean["corridor_min_z_m"]
    corridor_text = "n/a" if corridor is None else f"{corridor:.2f}m"
    lines = [f"Frame {clean['frame']} | t={clean['t']:.2f}s | {clean['fps']} FPS | "
             f"detected {clean['n_detected']} ({clean['n_detected_no_geometry']} without geometry) | "
             f"corridor min z {corridor_text}"]
    if not clean["objects"]:
        return lines[0]
    header = (f"{'rank':>4}  {'id':>6}  {'class':<14}{'src':<7}{'conf':>5}  {'age':>4}  {'range':>7}  "
              f"{'x':>6}  {'y':>6}  {'z':>6}  {'bearing':>8}  {'direction':<14}{'approach':>9}  notes")
    lines += [header, "-" * len(header)]

    def num(value, fmt):
        return "n/a".rjust(len(fmt.format(0))) if value is None else fmt.format(value)

    def bearing_text(value):
        return "     n/a" if value is None else f"{value:+7.1f}\u00b0"

    for o in clean["objects"]:
        approach = "n/a" if o["approach_mps"] is None else f"{o['approach_mps']:+.2f}"
        notes = []
        if not o["geometry_ok"]:
            notes.append("NO DEPTH")
        if o["clipped"]:
            notes.append(f"clipped: {','.join(o['clipped'])}")
        rank = "-" if o["rank"] is None else f"#{o['rank']}"
        lines.append(
            f"{rank:>4}  {str(o['id']):>6}  {o['class'][:13]:<14}{(o['source'] or '-'):<7}"
            f"{o['conf']:>5.2f}  {str(o['age_frames'] if o['age_frames'] is not None else '-'):>4}  "
            f"{num(o['range_m'], '{:5.2f}m')}  {num(o['x_m'], '{:+6.2f}')}  {num(o['y_m'], '{:+6.2f}')}  "
            f"{num(o['z_m'], '{:6.2f}')}  {bearing_text(o['bearing_deg'])}  "
            f"{(o['direction'] or 'n/a'):<14}{approach:>9}  {'; '.join(notes)}")
    lines.append("units: meters | x right+, y up+, z forward | bearing right+ | approach m/s, + = closing, "
                 "n/a = unknown")
    return "\n".join(lines)


class CleanJsonWriter:
    """Write clean frame records to one valid, pretty-printed JSON file.

    Layout: {"meta": {...}, "frames": [ ... ]}. Every frame is written, including empty
    ones, so a consumer can tell "nothing there" from "no message". Call close() at the end.
    """

    def __init__(self, path, meta):
        self.file = open(path, "w", encoding="utf-8")
        header = json.dumps(meta, indent=2).replace("\n", "\n  ")
        self.file.write(f'{{\n  "meta": {header},\n  "frames": [\n')
        self.first = True

    def write(self, record):
        self.file.write("" if self.first else ",\n")
        self.first = False
        body = json.dumps(clean_record(record), indent=2)
        self.file.write("\n".join("    " + line for line in body.splitlines()))
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.write("\n  ]\n}\n")
            self.file.close()


def main():
    parser = argparse.ArgumentParser(description="Add geometry to a saved JSONL session")
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", help="Write fused output here (optional)")
    parser.add_argument("--hfov", type=float, default=70.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--table", action="store_true", help="Print a readable table per frame")
    parser.add_argument("--clean", action="store_true",
                        help="Write one clean JSON file (meta + frames) instead of full JSONL records")
    args = parser.parse_args()

    fuser = GeometryFuser(args.hfov)
    dst = open(args.output, "w", encoding="utf-8") if args.output and not args.clean else None
    writer = None
    if args.output and args.clean:
        writer = CleanJsonWriter(args.output, {
            "source": args.input, "camera": {"width": args.width, "height": args.height,
                                             "hfov_deg": args.hfov}, "smoothing": fuser.describe()})
    with open(args.input, encoding="utf-8") as src:
        for line in src:
            record = json.loads(line)
            if record.get("type") == "frame":
                fuser.update(record, (args.height, args.width))
                if args.table:
                    print(format_table(record), end="\n\n")
                if writer:
                    writer.write(record)
                elif dst:
                    dst.write(json.dumps(record) + "\n")
            elif dst:
                dst.write(json.dumps(record) + "\n")
    if writer:
        writer.close()
    if dst:
        dst.close()


if __name__ == "__main__":
    main()