"""Decide which path answers a user request.

    route(text, record) -> Decision(kind, speak, question)

kind:
  "sensor"  answered from fast-path measurements only; no VLM call
  "hybrid"  speak the measured summary now, then run a VLM description
  "vlm"     needs the picture, or the thing was not detected (absence is never "not there")

Pure functions, no I/O: easy to test without a camera.
    python router.py        runs a few self-checks
"""
import re
import time
from dataclasses import dataclass

from prioritizer import direction_bucket, format_distance

MAX_AGE_S = 1.5   # sensor data older than this is not used to answer


@dataclass
class Decision:
    kind: str
    speak: str = ""
    question: str = ""


DESCRIBE = re.compile(r"\b(describe|around me|surroundings|look around|where am i|"
                      r"what (do|can) you see|what'?s here|what is here)\b")
SPATIAL = re.compile(r"\b(in front of me|in front|ahead|in my way|in the way|nearby|near me|"
                     r"close to me|how close|obstacles?)\b")
FIND = re.compile(r"\b(?:is there|are there|where(?:'?s| is| are)|how far(?: away)? is|find|locate)\s+"
                  r"(?:a |an |the |any |my |some )?(?P<what>[a-z ]+?)\s*(?:\?|$| on | to | in | near | ahead)")
GENERIC = {"anything", "something", "obstacle", "obstacles", "an obstacle"}
PEOPLE = {"person", "people", "anyone", "anybody", "someone", "somebody", "pedestrian", "human"}


def fresh(record):
    """True if the record is recent enough to be spoken as the current situation."""
    if not record:
        return False
    captured = record.get("capture_unix_s")
    return captured is None or time.time() - captured <= MAX_AGE_S


def _say(obj):
    geo = obj["geometry"]
    name = obj["class"].replace("_", " ").capitalize()
    return f"{name}, {format_distance(geo['range_m'])}, {direction_bucket(geo['bearing_deg'])}"


def _measured(record):
    objs = [o for o in record["objects"] if o.get("geometry", {}).get("ok")]
    return sorted(objs, key=lambda o: o["geometry"]["range_m"])


def summarize(record, limit=4):
    parts = [_say(o) for o in _measured(record)[:limit]]
    corridor = record.get("corridor_min_z_m")
    if corridor and corridor <= 4.0:
        parts.append(f"Solid surface straight ahead, {format_distance(corridor)}")
    return ". ".join(parts) + "." if parts else "Nothing detected."   # never "clear" or "safe"


def _matches(wanted, label):
    wanted, label = wanted.strip().rstrip("s"), label.lower().replace("_", " ").rstrip("s")
    if label == "person" and wanted in {w.rstrip("s") for w in PEOPLE}:
        return True
    return label == wanted or label.endswith(" " + wanted)


def route(text, record):
    q = text.lower().replace("\u2019", "'").strip()
    ok = fresh(record)

    if DESCRIBE.search(q):
        return Decision("hybrid", speak=summarize(record) if ok else "")

    m = FIND.search(q)
    if m and ok and m["what"].strip() not in GENERIC:
        hits = [o for o in _measured(record) if _matches(m["what"], o["class"])]
        if hits:
            return Decision("sensor", speak=". ".join(_say(o) for o in hits[:2]) + ".")
        return Decision("vlm", question=text)       # not detected is not "not there"

    if SPATIAL.search(q):
        return Decision("sensor", speak=summarize(record) if ok else "Distance sensing is not current.")

    return Decision("vlm", question=text)


if __name__ == "__main__":
    def obj(cls, rng, bearing):
        return {"class": cls, "geometry": {"ok": True, "range_m": rng, "bearing_deg": bearing}}

    rec = {"objects": [obj("person", 2.0, -30), obj("door", 4.0, 5)],
           "corridor_min_z_m": 3.0, "capture_unix_s": time.time()}
    tests = [("what's in front of me", "sensor"), ("is there a door?", "sensor"),
             ("is there a chair?", "vlm"), ("what's around me", "hybrid"),
             ("what does the sign say", "vlm"), ("is there anyone near me", "sensor")]
    for text, expected in tests:
        d = route(text, rec)
        print(f"{'OK ' if d.kind == expected else 'BAD'} {text!r:32} -> {d.kind:7} {d.speak}")
    print("stale:", route("what's in front of me", {**rec, "capture_unix_s": time.time() - 10}))