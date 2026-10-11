"""Geometry fuser: turn detections + metric depth + camera FOV into real-world geometry.

For every object in a frame record that has metric depth (unit "m"), adds:

    item["geometry"] = {
        "x_m":  right of the camera axis (+) / left (-)
        "y_m":  above the camera axis (+) / below (-)
        "z_m":  forward distance along the optical axis (smoothed for tracked objects)
        "range_m": straight-line distance sqrt(x^2 + y^2 + z^2)
        "bearing_deg": horizontal angle, + = right of center
        "elevation_deg": vertical angle, + = above center
        "approach_mps": closing speed, + = getting closer (tracked objects only, else None)
        "clipped_edges": frame edges the box touches, e.g. ["right"] (depth less reliable)
    }

With the fuser on, "depth_rank" is recomputed from the smoothed range_m (0 = nearest).

Assumes a pinhole camera with square pixels. Depth Anything's metric output is the
distance along the optical axis (Z), not the radial range, which is how it is used here.

Offline use on a saved session:
    python geometry_fuser.py session.jsonl fused.jsonl --hfov 70 --width 1280 --height 720
"""

import argparse
import json
import math
import statistics
from collections import deque


class GeometryFuser:
    def __init__(self, hfov_deg=70.0, window=5, alpha=0.4, max_age=30, edge_margin=8):
        """
        hfov_deg:    camera horizontal field of view. Typical webcams are 65-78 deg;
                     check your camera's spec sheet for a better value.
        window:      median window (frames) used to reject depth spikes per track.
        alpha:       EMA weight for new data (higher = faster, noisier).
        max_age:     forget a track after this many frames without seeing it.
        edge_margin: pixels from the frame border that count as "touching" it.
        """
        self.hfov_deg = hfov_deg
        self.window, self.alpha, self.max_age = window, alpha, max_age
        self.edge_margin = edge_margin
        self.tracks = {}

    def update(self, record, frame_shape):
        """Add geometry to every object in `record` (in place). frame_shape = (h, w, ...)."""
        h, w = frame_shape[:2]
        fx = (w / 2) / math.tan(math.radians(self.hfov_deg) / 2)
        fy = fx  # square pixels
        cx, cy = w / 2, h / 2
        t, index = record["timestamp_seconds"], record["frame_index"]

        for item in record["objects"]:
            depth = item.get("depth")
            if not depth or depth["unit"] != "m":
                continue  # geometry needs metric depth
            x1, y1, x2, y2 = item["bbox_xyxy"]
            u, v = (x1 + x2) / 2, (y1 + y2) / 2
            z = depth["value"]
            approach = None

            track_id = item.get("track_id")
            if track_id is not None:
                z, approach = self._smooth(track_id, z, t, index)

            x = (u - cx) * z / fx
            y = -(v - cy) * z / fy
            m = self.edge_margin
            clipped = [name for name, hit in (("left", x1 <= m), ("right", x2 >= w - m),
                                              ("top", y1 <= m), ("bottom", y2 >= h - m)) if hit]
            item["geometry"] = {
                "x_m": round(x, 3),
                "y_m": round(y, 3),
                "z_m": round(z, 3),
                "range_m": round(math.sqrt(x * x + y * y + z * z), 3),
                "bearing_deg": round(math.degrees(math.atan2(x, z)), 1),
                "elevation_deg": round(math.degrees(math.atan2(y, z)), 1),
                "approach_mps": None if approach is None else round(approach, 3),
                "clipped_edges": clipped,
            }

        # Re-rank on the smoothed straight-line range so depth_rank and geometry agree
        # (add_depth ranks on raw per-frame depth, which can flip on noise).
        fused = [item for item in record["objects"] if "geometry" in item]
        for rank, item in enumerate(sorted(fused, key=lambda i: i["geometry"]["range_m"])):
            item["depth_rank"] = rank

        # Drop tracks that have disappeared.
        stale = [k for k, s in self.tracks.items() if index - s["seen"] > self.max_age]
        for key in stale:
            del self.tracks[key]
        return record

    def _smooth(self, track_id, z_raw, t, index):
        state = self.tracks.get(track_id)
        if state is None:
            state = {"hist": deque(maxlen=self.window), "z": None, "t": t, "v": 0.0, "seen": index}
            self.tracks[track_id] = state
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
        return state["z"], state["v"]


def direction_word(bearing_deg):
    side = "right" if bearing_deg > 0 else "left"
    size = abs(bearing_deg)
    if size < 10:
        return "ahead"
    return f"slightly {side}" if size < 25 else side


def clean_record(record):
    """Compact, rounded view of a fused frame for logs and downstream stages."""
    objects = []
    for item in sorted(record["objects"], key=lambda o: o.get("depth_rank", 99)):
        geo = item.get("geometry")
        if not geo:
            continue
        objects.append({
            "id": item["track_id"], "class": item["class"], "source": item.get("source"),
            "conf": round(item["confidence"], 2), "rank": item.get("depth_rank"),
            "range_m": geo["range_m"], "x_m": geo["x_m"], "y_m": geo["y_m"], "z_m": geo["z_m"],
            "bearing_deg": geo["bearing_deg"], "direction": direction_word(geo["bearing_deg"]),
            "approach_mps": geo["approach_mps"], "clipped": geo["clipped_edges"],
        })
    return {"frame": record["frame_index"], "t": round(record["timestamp_seconds"], 2),
            "fps": round(record.get("fps", 0.0), 1), "objects": objects}


def format_table(record):
    """Human-readable table of one fused frame."""
    clean = clean_record(record)
    lines = [f"Frame {clean['frame']} | t={clean['t']:.2f}s | {clean['fps']} FPS | "
             f"{len(clean['objects'])} object(s) with geometry"]
    if not clean["objects"]:
        return lines[0]
    header = (f"{'rank':>4}  {'id':>6}  {'class':<14}{'src':<7}{'conf':>5}  {'range':>6}  "
              f"{'x':>6}  {'y':>6}  {'z':>6}  {'bearing':>8}  {'direction':<14}{'approach':>9}  notes")
    lines += [header, "-" * len(header)]
    for o in clean["objects"]:
        approach = "-" if o["approach_mps"] is None else f"{o['approach_mps']:+.2f}"
        notes = f"clipped: {','.join(o['clipped'])}" if o["clipped"] else ""
        lines.append(
            f"{'#' + str(o['rank']):>4}  {o['id']:>6}  {o['class'][:13]:<14}{(o['source'] or '-'):<7}"
            f"{o['conf']:>5.2f}  {o['range_m']:>5.2f}m  {o['x_m']:>+6.2f}  {o['y_m']:>+6.2f}  "
            f"{o['z_m']:>6.2f}  {o['bearing_deg']:>+7.1f}\u00b0  {o['direction']:<14}{approach:>9}  {notes}")
    lines.append("units: meters | x right+, y up+, z forward | bearing right+ | approach m/s, + = closing")
    return "\n".join(lines)


class CleanJsonWriter:
    """Write clean frame records to one valid, pretty-printed JSON array.

    Frames with no objects are skipped. Call close() at the end to finish the array.
    """

    def __init__(self, path):
        self.file = open(path, "w", encoding="utf-8")
        self.file.write("[\n")
        self.first = True

    def write(self, record):
        clean = clean_record(record)
        if not clean["objects"]:
            return
        self.file.write("" if self.first else ",\n")
        self.first = False
        body = json.dumps(clean, indent=2)
        self.file.write("\n".join("  " + line for line in body.splitlines()))
        self.file.flush()

    def close(self):
        if not self.file.closed:
            self.file.write("\n]\n")
            self.file.close()


def main():
    parser = argparse.ArgumentParser(description="Add geometry to a saved JSONL session")
    parser.add_argument("input")
    parser.add_argument("output", nargs="?", help="Write fused JSONL here (optional)")
    parser.add_argument("--hfov", type=float, default=70.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--table", action="store_true", help="Print a readable table per frame")
    parser.add_argument("--clean", action="store_true", help="Write compact clean records instead of full ones")
    args = parser.parse_args()

    fuser = GeometryFuser(args.hfov)
    dst = open(args.output, "w", encoding="utf-8") if args.output else None
    with open(args.input, encoding="utf-8") as src:
        for line in src:
            record = json.loads(line)
            if record.get("type") == "frame":
                fuser.update(record, (args.height, args.width))
                if args.table:
                    print(format_table(record), end="\n\n")
                if dst:
                    dst.write(json.dumps(clean_record(record) if args.clean else record) + "\n")
            elif dst and not args.clean:
                dst.write(json.dumps(record) + "\n")
    if dst:
        dst.close()


if __name__ == "__main__":
    main()