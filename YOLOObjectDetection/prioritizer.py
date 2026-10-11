"""Fast path: deterministic hazard logic. Knows nothing about the VLM.

    fused record -> Prioritizer.update() -> [Utterance(level=INFO|WARNING|CRITICAL)]

Why this replaces the weighted-score version
  * Every label belongs to a category with an explicit policy (distance, time-to-contact,
    in-path rule). "Why did it speak?" is always answerable from a few readable numbers;
    a hazard can no longer fall under a blended min_score.
  * Path overlap uses the object's WIDTH (geometry["width_m"] if present), not just its centre.
  * Dedup is by track AND by (label, direction), so a BoT-SORT ID switch does not re-announce.
  * Levels: CRITICAL (stop now) > WARNING (be aware) > INFO. The speaker preempts on level.
  * A class-free corridor check still catches unlabeled solid things straight ahead.
  * Health: if detections exist but depth/geometry keeps failing, it says so.

Replay a saved session:  python prioritizer.py session.jsonl
"""

import json
import math
import queue
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from enum import IntEnum


class Level(IntEnum):          # higher = more urgent (same convention as vlm_loop.py)
    INFO = 1
    WARNING = 2
    CRITICAL = 3


# Seconds an utterance may wait before it is stale and must be dropped.
TTL = {Level.INFO: 5.0, Level.WARNING: 3.0, Level.CRITICAL: 1.5}


@dataclass
class Utterance:
    text: str
    level: Level
    kind: str = "object"       # "object" | "obstacle" | "system"
    track_id: int | None = None
    t: float = 0.0

    @property
    def ttl(self):
        return TTL[self.level]

    @property
    def urgent(self):
        return self.level == Level.CRITICAL


# ----------------------------------------------------------------------------- categories
CATEGORY_LABELS = {
    # drops and level changes: dangerous even when standing still and off to the side
    "drop": {"staircase", "stairs", "stair", "descending stairs", "escalator", "pothole",
             "open manhole", "trench", "exposed construction hole", "hole", "sidewalk step",
             "step", "curb", "sidewalk curb", "drainage grate", "ladder", "fire escape ladder"},
    # fast or heavy things: judged by time-to-contact
    "vehicle": {"car", "truck", "bus", "motorcycle", "bicycle", "train", "electric scooter",
                "scooter", "skateboard", "delivery robot", "roller skates", "mobility scooter"},
    # people, animals and things pushed by people: unpredictable
    "moving": {"person", "dog", "cat", "horse", "cow", "stroller", "shopping cart", "wheelchair",
               "walker mobility aid", "crutches", "white cane", "walking cane"},
    # fixed obstacles that are easy to walk into (includes head-height ones)
    "fixed": {"pole", "utility pole", "streetlight pole", "bollard", "safety bollard",
              "parking bollard", "traffic delineator post", "traffic cone", "construction barricade",
              "concrete road barrier", "temporary fence", "scaffolding", "road closure gate",
              "signpost", "hanging sign", "low hanging branch", "protruding shelf",
              "protruding pipe", "open cabinet door", "open door", "glass door", "glass wall",
              "revolving door", "metal chain barrier", "rope barrier", "fire hydrant",
              "bicycle rack", "truck loading ramp", "roadwork sign", "open car door",
              "wet floor sign", "traffic spike strip"},
    # ground-level trip hazards (small, so detectors are less reliable on them)
    "trip": {"tree root", "fallen branch", "uneven sidewalk", "raised pavement slab",
             "cracked pavement", "speed bump", "loose gravel", "puddle", "door threshold",
             "manhole cover", "construction debris", "caution tape"},
    "furniture": {"chair", "bench", "couch", "bed", "table", "dining table", "potted plant",
                  "suitcase", "toilet", "refrigerator", "trash can", "bus shelter"},
    # useful to know, never urgent; off unless announce_landmarks=True
    "landmark": {"door", "elevator door", "automatic sliding door", "handrail", "railing",
                 "tactile paving", "tactile guidance strip", "curb ramp", "wheelchair ramp",
                 "pedestrian crosswalk", "pedestrian crossing button", "audible pedestrian signal",
                 "pedestrian signal", "bus stop sign"},
}
CATEGORY = {label: cat for cat, labels in CATEGORY_LABELS.items() for label in labels}


@dataclass(frozen=True)
class Policy:
    reach: float                   # ignore beyond this range (m)
    warn: float                    # in the path and within this range -> WARNING
    critical: float                # in the path and within this range -> CRITICAL
    ttc_warn: float = 0.0          # in the path and contact within this many s -> WARNING (0 = off)
    ttc_critical: float = 0.0      # ... -> CRITICAL (0 = off)
    side_range: float = 0.0        # beside the path, announce within this range (0 = never)
    side_needs_closing: bool = False
    max_level: Level = Level.CRITICAL


POLICIES = {
    "drop":      Policy(reach=6, warn=4.0, critical=1.5, side_range=3.0),
    "vehicle":   Policy(reach=10, warn=3.0, critical=1.2, ttc_warn=6.0, ttc_critical=3.5,
                        side_range=8.0, side_needs_closing=True),
    "moving":    Policy(reach=5, warn=2.5, critical=0.8, ttc_warn=4.0, ttc_critical=1.8,
                        side_range=3.0, side_needs_closing=True),
    "fixed":     Policy(reach=4, warn=2.5, critical=1.0),
    "trip":      Policy(reach=3, warn=2.0, critical=0.8, max_level=Level.WARNING),
    "furniture": Policy(reach=3, warn=1.8, critical=0.7),
    "landmark":  Policy(reach=5, warn=3.0, critical=0.0, max_level=Level.INFO),
    "other":     Policy(reach=2, warn=1.2, critical=0.5, max_level=Level.WARNING),
}
STOP_CATEGORIES = {"drop", "fixed", "furniture", "trip"}   # "Stop." vs "Careful."


def format_distance(meters):
    rounded = round(meters * 2) / 2
    if rounded <= 0.5:
        return "very close"
    return "1 meter" if rounded == 1 else f"{rounded:g} meters"


def direction_bucket(bearing_deg):
    side = "right" if bearing_deg > 0 else "left"
    size = abs(bearing_deg)
    if size < 10:
        return "ahead"
    return f"slightly {side}" if size < 25 else f"on your {side}"


@dataclass
class Candidate:
    track_id: int
    label: str
    level: Level
    text: str
    range: float
    bucket: str
    approaching: bool
    ttc: float | None


# ----------------------------------------------------------------------------- memory
class Announcer:
    """Decides whether something that qualifies should actually be said again."""
    COOLDOWN = {Level.CRITICAL: 4.0, Level.WARNING: 10.0, Level.INFO: 25.0}
    MIN_GAP = {Level.CRITICAL: 0.4, Level.WARNING: 1.2, Level.INFO: 2.5}   # between any two utterances

    def __init__(self, approach_step=1.0, reminder_factor=3.0):
        self.approach_step, self.reminder_factor = approach_step, reminder_factor
        self.tracks = {}    # track_id -> what we last said about it
        self.sigs = {}      # (label, bucket) -> same, survives tracker ID switches
        self.last_any = -1e9

    def gap_ok(self, level, t):
        return t - self.last_any >= self.MIN_GAP[level]

    def should_announce(self, c, t):
        cooldown = self.COOLDOWN[c.level]
        prev = self.tracks.get(c.track_id)
        if prev is None:
            other = self.sigs.get((c.label, c.bucket))     # same thing under a new ID?
            if (other and t - other["t"] < cooldown and abs(other["range"] - c.range) < 1.5
                    and other["level"] >= c.level):
                return False
            return True
        elapsed, closed = t - prev["t"], prev["range"] - c.range
        if c.level > prev["level"] and elapsed >= 0.5:
            return True                                   # escalation is always worth saying
        if c.level == Level.CRITICAL:
            return elapsed >= 2.0 and (closed >= 0.4 or elapsed >= cooldown)
        if c.approaching and elapsed >= 2.0 and (not prev["approaching"] or closed >= self.approach_step):
            return True                                   # countdown: 6 m ... 5 m ... 4 m
        if elapsed < cooldown:
            return False
        return abs(closed) >= 1.0 or c.bucket != prev["bucket"] or elapsed >= cooldown * self.reminder_factor

    def remember(self, c, t):
        rec = {"t": t, "range": c.range, "bucket": c.bucket, "approaching": c.approaching, "level": c.level}
        self.tracks[c.track_id] = rec
        self.sigs[(c.label, c.bucket)] = rec
        self.last_any = t

    def forget(self, track_id):
        self.tracks.pop(track_id, None)


# ----------------------------------------------------------------------------- prioritizer
class Prioritizer:
    def __init__(self, path_half_width=0.7, min_age=3, approach_mps=0.4, min_conf=None,
                 ignore=None, announce_landmarks=False, corridor_range=3.0, announcer=None):
        """
        path_half_width: half the walking corridor in meters (shoulders + margin).
        min_age:         frames a track must exist before it can be spoken.
        approach_mps:    closing speed that counts as "approaching" (it moving OR you walking).
        min_conf:        per-source confidence floor; YOLO-World scores run lower.
        """
        self.path_half_width, self.min_age, self.approach_mps = path_half_width, min_age, approach_mps
        self.min_conf = min_conf or {"yolo26": 0.35, "world": 0.12}
        self.ignore = set(ignore or [])
        self.announce_landmarks = announce_landmarks
        self.corridor_range = corridor_range
        self.ann = announcer or Announcer()
        self.tracks = {}
        self.corridor = None
        self._corridor_said = None
        self._blind_frames, self._blind_said = 0, -1e9
        self.last_alert_t = -1e9      # last WARNING/CRITICAL; the slow path stays quiet after it

    # ---- main entry ----
    def update(self, record):
        t, index = record["timestamp_seconds"], record["frame_index"]
        best = None
        for c in sorted((c for c in (self._assess(i, index) for i in record["objects"]) if c),
                        key=lambda c: (-c.level, c.ttc if c.ttc is not None else 99.0, c.range)):
            if self.ann.gap_ok(c.level, t) and self.ann.should_announce(c, t):
                best = c
                break
        obstacle = self._corridor_utterance(record, t)

        chosen = None
        if best is not None:
            chosen = Utterance(best.text, best.level, "object", best.track_id, t)
        if obstacle and (chosen is None or obstacle.level > chosen.level) \
                and self.ann.gap_ok(obstacle.level, t):
            chosen, best = obstacle, None
            self._corridor_said = {"t": t, "z": self.corridor["z"]}
        out = []
        if chosen:
            if best is not None:
                self.ann.remember(best, t)
            else:
                self.ann.last_any = t
            if chosen.level >= Level.WARNING:
                self.last_alert_t = t
            out.append(chosen)
        health = self._health(record, t)
        if health:
            out.append(health)
        self._prune(index)
        return out

    # ---- objects ----
    def _assess(self, item, index):
        geo, tid = item.get("geometry"), item.get("track_id")
        if not geo or not geo.get("ok") or tid is None:
            return None                         # needs metric geometry and a stable ID
        st = self.tracks.setdefault(tid, {"first": index, "votes": Counter()})
        st["last"] = index
        st["votes"][item["class"]] += item["confidence"]
        label = st["votes"].most_common(1)[0][0]            # stable class, not the first seen
        if label in self.ignore or index - st["first"] + 1 < self.min_age:
            return None
        if item["confidence"] < self.min_conf.get(item.get("source"), 0.3):
            return None
        cat = CATEGORY.get(label.lower(), "other")
        if cat == "landmark" and not self.announce_landmarks:
            return None
        pol = POLICIES[cat]
        rng, z, x = geo["range_m"], geo["z_m"], geo["x_m"]
        if rng > pol.reach:
            return None

        # Lateral gap between the walking corridor and the object's nearest edge (<= 0: overlaps).
        half_w = min((geo.get("width_m") or 0.0) / 2, 1.5)
        in_path = abs(x) - half_w - self.path_half_width <= 0.1

        closing = geo.get("approach_mps") or 0.0
        approaching = closing >= self.approach_mps
        ttc = z / closing if approaching and z > 0 else None

        level = None
        if in_path:
            if rng <= pol.critical or (ttc is not None and pol.ttc_critical and ttc <= pol.ttc_critical):
                level = Level.CRITICAL
            elif rng <= pol.warn or (ttc is not None and pol.ttc_warn and ttc <= pol.ttc_warn):
                level = Level.WARNING
        elif pol.side_range and rng <= pol.side_range and (approaching or not pol.side_needs_closing):
            level = Level.WARNING
        if level is None:
            return None
        level = min(level, pol.max_level)

        bucket = direction_bucket(geo["bearing_deg"])
        name = label.replace("_", " ")
        head = f"{name} approaching" if approaching and cat in ("vehicle", "moving") else name
        text = f"{head}, {format_distance(rng)}, {bucket}"
        if level == Level.CRITICAL:
            text = ("Stop. " if cat in STOP_CATEGORIES else "Careful. ") + text
        return Candidate(tid, label, level, text, rng, bucket, approaching, ttc)

    # ---- unknown obstacles (no class needed) ----
    def _update_corridor(self, z_raw, t):
        c = self.corridor
        if z_raw is None:
            self.corridor = None
            return None
        if c is None:
            self.corridor = {"z": z_raw, "t": t, "v": 0.0, "n": 1}
            return self.corridor
        new_z = 0.5 * z_raw + 0.5 * c["z"]
        dt = t - c["t"]
        if dt > 1e-3:
            c["v"] = 0.4 * ((c["z"] - new_z) / dt) + 0.6 * c["v"]       # + = closing
        c["z"], c["t"] = new_z, t
        c["n"] += 1
        return c

    def _corridor_utterance(self, record, t):
        c = self._update_corridor(record.get("corridor_min_z_m"), t)
        if c is None or c["n"] < 3:
            return None
        z = c["z"]
        if z > self.corridor_range:
            self._corridor_said = None                                   # clear again
            return None
        approaching = c["v"] >= self.approach_mps
        ttc = z / c["v"] if approaching else None
        if not (z <= 1.5 or (ttc is not None and ttc <= 3.0)):
            return None
        for item in record["objects"]:                                   # a detection covers it
            geo = item.get("geometry")
            if geo and geo.get("ok") and abs(geo["x_m"]) <= 0.8 and abs(geo["z_m"] - z) < 0.6:
                return None
        said = self._corridor_said
        if said is not None:
            elapsed, closed = t - said["t"], said["z"] - z
            if not (elapsed >= 8.0 or (closed >= 0.5 and elapsed >= 1.0)):
                return None
        critical = z < 1.0 or (ttc is not None and ttc < 2.0)
        text = f"Obstacle ahead, {format_distance(z)}"
        if critical:
            return Utterance("Stop. " + text, Level.CRITICAL, "obstacle", None, t)
        return Utterance(text, Level.WARNING, "obstacle", None, t)

    # ---- health: never silently lose depth ----
    def _health(self, record, t):
        objs = record["objects"]
        if objs and not any(o.get("geometry", {}).get("ok") for o in objs):
            self._blind_frames += 1
        else:
            self._blind_frames = 0
        if self._blind_frames >= 15 and t - self._blind_said >= 60.0:
            self._blind_said = t
            return Utterance("Distance sensing unavailable", Level.WARNING, "system", None, t)
        return None

    def _prune(self, index, max_gap=300):
        for key in [k for k, s in self.tracks.items() if index - s["last"] > max_gap]:
            del self.tracks[key]
            self.ann.forget(key)


# ----------------------------------------------------------------------------- compat
# Only for webcam_detection.py's in-process --speak/--tts mode. With vlm_loop.py the speaker
# there (preemptible, narration-aware) is used instead.
class UtteranceQueue:
    """Speaks the most urgent utterance first and drops ones that went stale."""

    def __init__(self, speak_fn=None):
        self.speak = speak_fn or (lambda text: print(f"SPEAK: {text}", flush=True))
        self._q, self._count = queue.PriorityQueue(), 0
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()
        return self

    def put(self, utt):
        self._count += 1
        self._q.put((-int(utt.level), self._count, time.monotonic(), utt))

    def _run(self):
        while not self._stop_event.is_set():
            try:
                _, _, queued_at, utt = self._q.get(timeout=0.2)
            except queue.Empty:
                continue
            if time.monotonic() - queued_at <= utt.ttl:
                self.speak(utt.text)

    def stop(self):
        self._stop_event.set()


def make_pyttsx3_speaker(rate=185):
    """Offline TTS via pyttsx3 (created lazily so it lives in the worker thread)."""
    engine = None

    def speak(text):
        nonlocal engine
        if engine is None:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", rate)
        engine.say(text)
        engine.runAndWait()

    return speak


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: python prioritizer.py session.jsonl")
    p = Prioritizer()
    with open(sys.argv[1], encoding="utf-8") as src:
        for line in src:
            record = json.loads(line)
            if record.get("type") == "frame":
                for u in p.update(record):
                    print(f"t={record['timestamp_seconds']:6.1f}s  {u.level.name:<8} {u.text}")


if __name__ == "__main__":
    main()