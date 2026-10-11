"""Fast-path hazard rules driven by the geometry fuser's output stream.

The fuser appends ONE JSON object per frame, ONE PER LINE (JSONL), to a file such as
/dev/shm/fused.jsonl. This module tails that file, decides what is worth telling the
user, and calls submit(text, priority=..., ttl=..., dedupe=False).

It has no heavy imports, so it can be tested without a camera, GPU or Ollama:
    python fastpath.py /dev/shm/fused.jsonl     (prints what it would say)

Field names follow the fuser's data dictionary: objects[].class, conf, rank, range_m,
x_m (right +), y_m, z_m (forward), bearing_deg (right +), approach_mps (closing +),
clipped, age_frames, geometry_ok; per frame: corridor_min_z_m, n_detected,
n_detected_no_geometry. Anything missing or null is treated as "unknown".
"""
import json
import math
import os
import sys
import threading
import time

# ---- depth calibration: z_true ~= DEPTH_SCALE * z_measured + DEPTH_OFFSET_M ----
# Measured on the GX10 camera: centered objects read ~1 m too far on average.
# Reading too far is the dangerous direction, so we correct the number itself
# (thresholds AND spoken distances stay honest). Refit both with a tape measure at
# several distances (e.g. 1, 2, 3, 5 m): a constant offset and a scale error look the
# same at one distance.
DEPTH_SCALE = 1.0
DEPTH_OFFSET_M = -1.0
MIN_Z_M = 0.3              # never report closer than this after correction


def calibrate(z):
    """Measured forward depth -> corrected depth (None stays None)."""
    return None if z is None else max(MIN_Z_M, DEPTH_SCALE * z + DEPTH_OFFSET_M)


# ---- tunable rules (meters, m/s, seconds). Tune these on recorded real data. ----
PATH_HALF_WIDTH_M = 0.7    # |x_m| below this = in the user's walking path
STOP_ZONE_M = 1.5          # in path and closer than this = urgent
WARN_ZONE_M = 3.0          # in path and closer than this = warn
APPROACH_MPS = 0.3         # closing speed that counts as "approaching" (the reported
                           # value lags the true speed, so keep this low)
APPROACH_ZONE_M = 5.0      # only warn about approach inside this distance
APPROACH_SIDE_DEG = 25.0   # ...and inside this bearing, even if not strictly in path
CORRIDOR_WARN_M = 1.5      # unclassified obstacle straight ahead closer than this
MIN_TRACK_FRAMES = 3       # ignore very young tracks (jitter) unless already very close
STALE_S = 2.0              # no fuser record for this long = sensing lost

REPEAT_S = 4.0             # same object + same kind may be announced again after this long...
KIND_REPEAT_S = {"nogeo": 8.0, "corridor": 5.0, "landmark": 30.0}
REWARN_CLOSER_M = 1.0      # ...or sooner if it got this much closer since last announced
REWARN_MIN_S = 1.5         # ...but not more often than this (fast movers close 1 m per frame)
MIN_GAP_S = 1.0            # never start two fast-path announcements closer than this
HAZARD_TTL_S = 2.0         # a hazard that waited longer than this is dropped as stale


def _num(v):
    """A real number or None (rejects None, NaN, strings)."""
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v) else None


def direction_word(bearing):
    if bearing is None:
        return "ahead"
    a = abs(bearing)
    if a < 10:
        return "ahead"
    side = "right" if bearing > 0 else "left"
    return f"slightly {side}" if a < 25 else side


def dist_text(z, clipped):
    if clipped:
        return "close"       # box touches the frame edge: the number is less reliable
    n = max(1, round(z))
    return f"{n} meter" + ("" if n == 1 else "s")


# ---- object tags: which rules apply to which class ----
# Class names are matched lower-case. Anything not listed is "default".
TAGS = {
    # Fast movers. The user walking at ~1.3 m/s toward a PARKED car also reads as
    # "approaching", so vehicles need a closing speed clearly above walking pace.
    "vehicle": {"car", "truck", "bus", "motorcycle", "bicycle", "train", "electric scooter",
                "mobility scooter", "delivery robot", "open car door"},
    # Ground-level drops and trip hazards: the corridor strip skips the floor, so these
    # rely on detection alone and get a longer warning distance.
    "ground": {"descending stairs", "staircase", "sidewalk step", "sidewalk curb", "open manhole",
               "exposed construction hole", "trench", "pothole", "uneven sidewalk",
               "raised pavement slab", "door threshold", "speed bump", "tree root",
               "fallen branch", "traffic spike strip", "truck loading ramp", "puddle"},
    # Head/torso-height obstacles a white cane does not find.
    "overhead": {"low hanging branch", "hanging sign", "protruding shelf", "protruding pipe",
                 "open cabinet door", "scaffolding"},
    # Useful to know about, not dangerous: said once, quietly, then left alone.
    "landmark": {"traffic light", "stop sign", "pedestrian crosswalk", "pedestrian signal",
                 "pedestrian crossing button", "audible pedestrian signal", "bus stop sign",
                 "bus shelter", "escalator", "elevator door", "handrail", "railing",
                 "open door", "glass door", "automatic sliding door", "revolving door",
                 "wheelchair ramp", "curb ramp", "tactile paving", "tactile guidance strip",
                 "bench", "fire hydrant"},
}
TAG_OF = {name: tag for tag, names in TAGS.items() for name in names}

DEFAULT_RULES = {"half_width": PATH_HALF_WIDTH_M, "stop": STOP_ZONE_M, "warn": WARN_ZONE_M,
                 "approach_mps": APPROACH_MPS, "approach_zone": APPROACH_ZONE_M,
                 "side_deg": APPROACH_SIDE_DEG, "ttc_s": None, "nogeo": False, "announce": None}
RULES = {
    "default":  {},
    "vehicle":  {"half_width": 1.2, "warn": 6.0, "approach_mps": 2.0, "approach_zone": 20.0,
                 "side_deg": 40.0, "ttc_s": 4.0, "nogeo": True},
    "ground":   {"stop": 2.0, "warn": 4.0},
    "overhead": {"stop": 2.0, "warn": 3.5},
    "landmark": {"stop": 0.0, "warn": 0.0, "approach_mps": 99.0, "announce": 10.0},
}
for _tag, _r in RULES.items():
    RULES[_tag] = {**DEFAULT_RULES, **_r}
NO_GEOMETRY_CLASSES = {"person"}   # plus every class whose rules say nogeo


def tag_of(cls):
    return TAG_OF.get(cls.lower(), "default")


def assess(rec):
    """One fuser record -> list of candidates (urgency, key, text, z). Pure function."""
    out = []
    objs = rec.get("objects") or []
    all_z = []
    for o in objs:
        cls = str(o.get("class", "object"))
        tag = tag_of(cls)
        r = RULES[tag]
        label = cls.capitalize()
        oid = o.get("id")
        ident = oid if oid is not None else cls
        z_raw = _num(o.get("z_m"))
        z = calibrate(z_raw)
        if o.get("geometry_ok") is False or z is None:
            if cls.lower() in NO_GEOMETRY_CLASSES or r["nogeo"]:
                out.append((1, (ident, "nogeo"), f"{label} detected, distance unknown.", None))
            continue
        all_z.append(z)
        x = _num(o.get("x_m"))
        if x is not None and z_raw:
            x *= z / z_raw          # x = (u - cx) * z / fx is linear in z; bearing is unchanged
        b = _num(o.get("bearing_deg"))
        v = _num(o.get("approach_mps"))
        if v is not None:
            v *= DEPTH_SCALE        # a constant offset cancels in dz/dt; a scale does not
        age = _num(o.get("age_frames"))
        clipped = bool(o.get("clipped"))
        in_path = x is not None and abs(x) < r["half_width"]
        near_axis = b is not None and abs(b) < r["side_deg"]
        approaching = v is not None and v >= r["approach_mps"]
        if age is not None and age < MIN_TRACK_FRAMES and z > r["stop"]:
            continue
        where = direction_word(b)
        dist = dist_text(z, clipped)
        ttc = z / v if (v is not None and v > 0.1) else None   # seconds to contact
        # Where will it be sideways when it reaches us? A car driving past on the road
        # closes in z too, but its x stays ~3 m: that is "passing", not "coming at us".
        vx = _num(o.get("_vx_mps"))                  # added by FastPath from the x history
        if ttc is not None and x is not None and vx is not None:
            will_hit = abs(x + vx * ttc) < r["half_width"]
        else:
            will_hit = in_path                       # no lateral history yet: be literal
        if r["ttc_s"] and approaching and ttc is not None and ttc < r["ttc_s"] and will_hit:
            out.append((3, (ident, "stop"), f"Stop. {label} coming {where}, fast.", z))
        elif r["ttc_s"] and approaching and not will_hit:
            pass                                     # fast mover that will miss us: stay quiet
        elif in_path and z < r["stop"]:
            out.append((3, (ident, "stop"), f"Stop. {label} {where}, {dist}.", z))
        elif approaching and z < r["approach_zone"] and (in_path or near_axis):
            out.append((2, (ident, "approach"), f"{label} {where}, {dist}, approaching.", z))
        elif in_path and z < r["warn"]:
            out.append((2, (ident, "path"), f"{label} {where}, {dist}.", z))
        elif r["announce"] and z < r["announce"]:
            out.append((1, (ident, "landmark"), f"{label} {where}, {dist}.", z))

    # Something solid straight ahead that no detected object accounts for (pole, table edge...)
    c = calibrate(_num(rec.get("corridor_min_z_m")))   # same bias assumed; unverified
    if c is not None and c < CORRIDOR_WARN_M and not any(abs(c - z) < 0.5 for z in all_z):
        out.append((2, ("corridor", "corridor"), f"Obstacle ahead, {dist_text(c, False)}.", c))
    return out


class FastPath:
    def __init__(self, path, submit, priority):
        self.path = path
        self.submit = submit
        self.priority = priority
        self.last_warn = {}          # key -> (time, z)
        self.last_any = 0.0
        self.last_seen = None        # local time of the last fuser record
        self.lost = False
        self.lateral = {}            # track id -> (t, x, vx)
        self.latest = None           # (local time, newest record), for scene_summary()

    def start(self):
        threading.Thread(target=self._run, daemon=True).start()

    # ---- file tailing ----
    def _run(self):
        f = None
        existed_at_start = os.path.exists(self.path)
        while True:
            if f is None:
                if not os.path.exists(self.path):
                    self._check_lost()
                    time.sleep(0.1)
                    continue
                f = open(self.path, "rb")
                if existed_at_start:
                    f.seek(0, os.SEEK_END)       # file was already there: follow new lines only
                existed_at_start = False         # any later reopen starts from the top
            line = f.readline()
            if not line:
                try:
                    if os.stat(self.path).st_size < f.tell():     # file was truncated/restarted
                        f.close()
                        f = None
                        continue
                except FileNotFoundError:
                    f.close()
                    f = None
                    continue
                self._check_lost()
                time.sleep(0.02)
                continue
            if not line.endswith(b"\n"):
                f.seek(-len(line), os.SEEK_CUR)  # half-written line: wait and retry
                time.sleep(0.01)
                continue
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if isinstance(rec, dict) and "objects" in rec:    # skips a leading "meta" line
                self._on_record(rec)

    # ---- decisions ----
    def _check_lost(self):
        if self.last_seen is not None and not self.lost and time.time() - self.last_seen > STALE_S:
            self.lost = True
            self._say("Distance sensing lost.")

    def _say(self, text, urgency=2):
        # Graded priority (HAZARD + 0.1 * urgency): an urgent "Stop" can cut off a less
        # urgent hazard sentence that is still being spoken, not only a description.
        self.submit(text, priority=self.priority + 0.1 * urgency, ttl=HAZARD_TTL_S, dedupe=False)
        self.last_any = time.time()

    def _add_lateral_speed(self, rec, now):
        """Per track: EMA of dx/dt (m/s, right +), stored on the object as _vx_mps."""
        t = _num(rec.get("t"))
        t = now if t is None else t
        for o in rec.get("objects") or []:
            oid, z_raw, x = o.get("id"), _num(o.get("z_m")), _num(o.get("x_m"))
            if oid is None or z_raw is None or x is None:
                continue
            z = calibrate(z_raw)
            x = x * z / z_raw if z_raw else x
            prev = self.lateral.get(oid)
            vx = None
            if prev is not None and t - prev[0] > 1e-3:
                inst = (x - prev[1]) / (t - prev[0])
                vx = inst if prev[2] is None else 0.4 * inst + 0.6 * prev[2]
            self.lateral[oid] = (t, x, vx)
            if vx is not None:
                o["_vx_mps"] = vx
        if len(self.lateral) > 500:
            self.lateral.clear()

    def _on_record(self, rec):
        now = time.time()
        self._add_lateral_speed(rec, now)
        self.latest = (now, rec)
        self.last_seen = now
        if self.lost:
            self.lost = False
            self._say("Distance sensing restored.")
            return
        cands = sorted(assess(rec), key=lambda c: (-c[0], c[3] if c[3] is not None else 99))
        for urgency, key, text, z in cands:
            if now - self.last_any < MIN_GAP_S and urgency < 3:
                continue                              # urgent (3) announcements ignore the gap
            prev = self.last_warn.get(key)
            if prev is not None:
                t_prev, z_prev = prev
                closer = (key[1] != "landmark" and z is not None and z_prev is not None
                          and z_prev - z >= REWARN_CLOSER_M)
                closer = closer and now - t_prev >= REWARN_MIN_S
                if now - t_prev < KIND_REPEAT_S.get(key[1], REPEAT_S) and not closer:
                    continue                      # already told them; try the next candidate
            self.last_warn[key] = (now, z)
            self._say(text, urgency)
            return


_instance = None


def start(path, submit, priority):
    global _instance
    fp = FastPath(path, submit, priority)
    fp.start()
    _instance = fp
    return fp


def scene_summary(max_age_s=1.5, limit=8):
    """Newest fuser objects as short phrases ("chair, 2 meters, left"), nearest first,
    with the depth calibration applied. Empty if there is no recent record."""
    if _instance is None or _instance.latest is None:
        return []
    seen_at, rec = _instance.latest
    if time.time() - seen_at > max_age_s:
        return []
    items = []
    for o in rec.get("objects") or []:
        z = calibrate(_num(o.get("z_m")))
        if o.get("geometry_ok") is False or z is None:
            continue
        cls = str(o.get("class", "object"))
        items.append((z, f"{cls}, {dist_text(z, bool(o.get('clipped')))}, "
                         f"{direction_word(_num(o.get('bearing_deg')))}"))
    return [text for _, text in sorted(items)[:limit]]


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "/dev/shm/fused.jsonl"

    def show(text, priority=None, ttl=None, dedupe=None):
        print(f"{time.strftime('%H:%M:%S')} [would speak] {text}", flush=True)

    start(path, show, 2)
    while True:
        time.sleep(1)
