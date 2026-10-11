"""Fast-path runner for vlm_loop.py: follows the fuser's JSONL, runs the Prioritizer on
EVERY frame, and hands utterances to the speaker. No VLM code in here on purpose.

    fastpath.start("/dev/shm/fused.jsonl", speaker.submit)  -> FastPath (daemon thread)

Safety behaviour
  * Stale frames (captured > max_lag_s ago) still update the smoothing state but are never
    spoken, so a hiccup cannot make it announce where things WERE.
  * If the fuser stops writing for stall_s seconds, the user hears "Vision paused" (and
    "Vision resumed" later). Silence must never be mistaken for "nothing there".
  * Any exception while processing a frame is logged and the loop keeps going.

Accepts the fuser's full frame records (--json-output) or the clean format (one record per line).
"""
import json
import os
import threading
import time

from prioritizer import Level, Prioritizer, Utterance, direction_bucket

# vlm_loop.py priorities: higher number = more urgent.
PRIORITY = {Level.CRITICAL: 3, Level.WARNING: 2, Level.INFO: 1}


def normalize(record):
    """Return a full-format frame record, or None for lines that are not frames."""
    if record.get("type") == "frame":
        return record
    if "frame" in record and "objects" in record:           # clean format
        objects = []
        for o in record["objects"]:
            objects.append({
                "class": o["class"], "track_id": o["id"], "confidence": o["conf"],
                "source": o.get("source"),
                "geometry": {"ok": o["geometry_ok"], "x_m": o["x_m"], "z_m": o["z_m"],
                             "range_m": o["range_m"], "bearing_deg": o["bearing_deg"],
                             "approach_mps": o["approach_mps"], "age_frames": o["age_frames"],
                             "clipped_edges": o["clipped"], "width_m": o.get("width_m")}})
        return {"type": "frame", "frame_index": record["frame"], "timestamp_seconds": record["t"],
                "capture_unix_s": record.get("capture_unix_s"),
                "corridor_min_z_m": record.get("corridor_min_z_m"), "objects": objects}
    return None


def context_from_record(record, limit=6):
    """Sensor facts to put in a VLM prompt so its answer agrees with the fast path."""
    if not record:
        return ""
    objs = [o for o in record["objects"] if o.get("geometry", {}).get("ok")]
    objs.sort(key=lambda o: o["geometry"]["range_m"])
    parts = [f"{o['class']} {o['geometry']['range_m']:.1f} m {direction_bucket(o['geometry']['bearing_deg'])}"
             for o in objs[:limit]]
    corridor = record.get("corridor_min_z_m")
    if corridor:
        parts.append(f"nearest solid surface straight ahead {corridor:.1f} m")
    return ("Distances measured by sensors (trust these over your own estimates): "
            + "; ".join(parts) + ".") if parts else ""


class FastPath(threading.Thread):
    def __init__(self, path, submit, prioritizer=None, stall_s=2.0, max_lag_s=1.5):
        super().__init__(daemon=True)
        self.path, self.submit = path, submit
        self.prioritizer = prioritizer or Prioritizer()
        self.stall_s, self.max_lag_s = stall_s, max_lag_s
        self.latest = None                    # newest record, for VLM context
        self._halt = threading.Event()
        self._last_frame = None               # monotonic time of last frame; None until first
        self._stalled = False
        self.last_alert_mono = -1e9           # last WARNING/CRITICAL spoken (monotonic clock)

    def stop(self):
        self._halt.set()

    def context(self):
        return context_from_record(self.latest)

    def _say(self, text, level, ttl=None):
        if level >= Level.WARNING:
            self.last_alert_mono = time.monotonic()
        self.submit(text, priority=PRIORITY[level], ttl=ttl or {3: 1.5, 2: 3.0, 1: 5.0}[PRIORITY[level]],
                    dedupe=False)             # the prioritizer already dedupes (and counts down)

    def _handle(self, line):
        try:
            record = normalize(json.loads(line))
            if record is None:
                return
            self.latest = record
            utterances = self.prioritizer.update(record)
            captured = record.get("capture_unix_s")
            lag = time.time() - captured if captured else 0.0
            if lag > self.max_lag_s:
                return                        # state updated, but do not speak about the past
            for u in utterances:
                self._say(u.text, u.level, u.ttl)
        except Exception as error:            # never let one bad frame kill safety
            print(f"(fast path error: {error!r})", flush=True)

    def _watchdog(self):
        if self._last_frame is None:
            return
        late = time.monotonic() - self._last_frame > self.stall_s
        if late and not self._stalled:
            self._stalled = True
            self._say("Vision paused", Level.WARNING, 3.0)
        elif not late and self._stalled:
            self._stalled = False
            self._say("Vision resumed", Level.INFO, 3.0)

    def run(self):
        while not self._halt.is_set() and not os.path.exists(self.path):
            time.sleep(0.2)
        with open(self.path, encoding="utf-8") as f:
            f.seek(0, os.SEEK_END)            # ignore anything left from an older session
            buf = ""
            while not self._halt.is_set():
                chunk = f.readline()
                if not chunk:
                    try:
                        if os.stat(self.path).st_size < f.tell():
                            f.seek(0)         # file was truncated: new session
                    except FileNotFoundError:
                        pass
                    self._watchdog()
                    time.sleep(0.01)
                    continue
                buf += chunk
                if not buf.endswith("\n"):
                    continue                  # partial line: wait for the rest
                line, buf = buf, ""
                self._last_frame = time.monotonic()
                self._watchdog()              # may announce "resumed"
                self._handle(line)


def start(path, submit, **kwargs):
    fast = FastPath(path, submit, **kwargs)
    fast.start()
    return fast