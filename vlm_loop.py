"""Camera (live, shared-frame file, or video) -> Ollama VLM -> prioritizer -> speech,
plus an optional fast path driven by the geometry fuser's output.

Two modes (switch while running: type a key, then Enter):
  n  navigation mode   short, continuous, path-focused sentences (walking / outdoors)
  d  description mode  one longer, ordered description of the view, on demand (user is safe)
  r  describe again    a fresh description (switches to description mode if needed)
Hazards from the fast path stay ON in both modes and cut off whatever is being said.

Pipeline features:
  - downscaled frames, capped output, keep_alive, warm-up call at startup
  - navigation: frames sent only when the scene changed (plus a periodic refresh)
  - a prioritizer: one waiting slot (newest wins) for navigation, an ordered backlog for
    description-mode narration, hazards outrank everything and INTERRUPT speech mid-sentence,
    stale sentences are dropped, near-duplicate navigation sentences are suppressed
  - fast path (fastpath.py): hazards from the fuser's JSONL stream

Install: pip install opencv-python-headless requests numpy
Run:     python vlm_loop.py                                  live webcam
         python vlm_loop.py clip.mp4                         replay a recording in real time
         python vlm_loop.py --frame-file /dev/shm/latest.jpg frames saved by the fuser process
Add:     --fused /dev/shm/fused.jsonl                        enable the fast path
         --mode description                                  start in description mode
"""
import argparse
import base64
import json
import os
import re
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"       # also tested: gemma4:26b (needs "think": False)
CAMERA_INDEX = 0

MAX_SIDE = 512            # downscale long side to this many pixels
MAX_TOKENS = 40           # navigation: hard cap on output length
CHANGE_THRESHOLD = 12.0   # mean pixel difference (0-255) that counts as "changed"
MIN_INTERVAL_S = 1.0      # never call more often than this
MAX_INTERVAL_S = 8.0      # force a refresh at least this often
DARK_THRESHOLD = 20.0     # mean brightness (0-255) below this = camera blocked/dark

# Description mode
DESCRIBE_MAX_TOKENS = 200   # roughly 4-5 sentences
NARRATION_TTL_S = 120.0     # narration sentences wait in order; they do not go stale quickly

# Prioritizer settings
COOLDOWN_S = 10.0         # the same thing may be said again after this many seconds
SIMILARITY = 0.6          # fraction of shared object words that counts as "the same thing"
INFO_TTL_S = 5.0          # a waiting description older than this is dropped as stale
HAZARD, INFO = 2, 1       # higher number = more urgent
USE_TOLD_CONTEXT = False  # tell the model what was already said. Can push it to invent
                          # "new" things, so test with it off first, then A/B it.

# Until real TTS is attached, pretend speaking takes time so backlog behavior is realistic.
# Set to False when speak() calls Piper/Kokoro (those calls take the time themselves).
SIMULATE_SPEECH_TIME = True
SPEAK_WORDS_PER_S = 2.5

PROMPT = (
    "You are the eyes for a blind person. Say the most important things you can "
    "clearly see right now, with direction (left, ahead, right). One sentence, "
    "maximum 12 words. Only mention things that are definitely visible; never "
    "guess or invent. You see a single still image, so never describe motion or "
    "what anyone is doing or intending. For people, say only that a person is "
    "present and where; never describe clothing or faces, and never read names "
    "or badges. If the image is dark, blocked, blurry, or unclear, "
    "reply exactly: Camera view unclear."
)

DESCRIBE_PROMPT = (
    "Describe this scene for a blind person who is standing still and is safe. Use 4 to 5 "
    "short sentences, about 60 words: first the overall setting, then the main objects and "
    "where they are (left, right, ahead, near, far), then any readable signs or text, then "
    "notable colors. Describe only what you can see. Do not mention faces, clothing or "
    "names, and do not say what anyone is doing. "
    "Only if the image is completely dark or blank, reply exactly: Camera view unclear."
)

# Used for one retry if the model answers "Camera view unclear" on a frame that looks normal.
SIMPLE_DESCRIBE_PROMPT = (
    "Describe this image in 4 short sentences for a blind person. "
    "Say where things are (left, right, ahead)."
)
FLAT_THRESHOLD = 8.0        # gray-level spread below this = blank wall / lens cap, not a normal scene


# ---------- camera sources: all offer .get() -> newest frame (BGR) or None ----------
class LatestFrame:
    """Live webcam. Reads in a thread so we always get the newest frame, not a stale buffered one."""

    def __init__(self, index):
        self.cap = cv2.VideoCapture(index)
        self.frame = None
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while True:
            ok, f = self.cap.read()
            if ok:
                with self.lock:
                    self.frame = f

    def get(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


class SharedFrame:
    """Newest frame that ANOTHER process (the fuser) saved to a file such as /dev/shm/latest.jpg.
    The writer must save atomically: write to a temp name ending in .jpg, then os.replace().
    Returns None if the file is missing or older than max_age_s (camera stalled)."""

    def __init__(self, path, max_age_s=2.0):
        self.path = path
        self.max_age_s = max_age_s
        self.last_mtime = 0.0
        self.last = None

    def get(self):
        try:
            mtime = os.stat(self.path).st_mtime
        except FileNotFoundError:
            return None
        if time.time() - mtime > self.max_age_s:
            return None
        if mtime != self.last_mtime:
            img = cv2.imread(self.path)
            if img is not None:
                self.last, self.last_mtime = img, mtime
        return None if self.last is None else self.last.copy()


class VideoFrame:
    """Plays a video file in real time and offers the same .get() as LatestFrame.
    Skips frames when the pipeline is slow, exactly like a live camera would."""

    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open video: {path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
        self.start = time.time()
        self.pos = 0
        self.last = None
        self.done = False

    def get(self):
        target = int((time.time() - self.start) * self.fps)
        while self.pos <= target:
            ok, f = self.cap.read()
            if not ok:
                self.done = True
                return None
            self.last, self.pos = f, self.pos + 1
        return None if self.last is None else self.last.copy()


# ---------- helpers ----------
def downscale(frame, max_side=MAX_SIDE):
    h, w = frame.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return frame
    return cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)


def to_b64_jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return base64.b64encode(buf.tobytes()).decode()


def thumb(frame):
    return cv2.cvtColor(cv2.resize(frame, (64, 48)), cv2.COLOR_BGR2GRAY).astype(np.float32)


def scene_changed(a, b):
    return float(np.mean(np.abs(a - b))) > CHANGE_THRESHOLD


# ---------- memory of what was actually SPOKEN (drives duplicate filter + "already told") ----------
DIRECTIONS = {"left", "right", "ahead"}
STOPWORDS = {"a", "an", "the", "is", "are", "there", "of", "to", "and", "with", "on",
             "in", "slightly", "directly", "it", "its", "this", "that", "i", "see"}
spoken_history = []       # (object_words, direction_words, time_spoken, text)
history_lock = threading.Lock()


def _key(sentence):
    words = re.findall(r"[a-z]+", sentence.lower())
    return (
        {w for w in words if w not in STOPWORDS and w not in DIRECTIONS},
        {w for w in words if w in DIRECTIONS},
    )


def _prune():
    now = time.time()
    spoken_history[:] = [h for h in spoken_history if now - h[2] < COOLDOWN_S]


def is_repeat(sentence):
    objs, dirs = _key(sentence)
    with history_lock:
        _prune()
        for o, d, _, _ in spoken_history:
            union = objs | o
            sim = len(objs & o) / len(union) if union else 1.0
            if sim >= SIMILARITY and dirs == d:
                return True
    return False


def mark_spoken(sentence):
    objs, dirs = _key(sentence)
    with history_lock:
        spoken_history.append((objs, dirs, time.time(), sentence))


def told_recently(n=2):
    with history_lock:
        _prune()
        return [h[3] for h in spoken_history[-n:]]


# ---------- speech + prioritizer ----------
def speak(text, cancel=None):
    """Speak one sentence. `cancel` is an Event: if it gets set, stop talking immediately.
    Real TTS: start Piper/Kokoro/espeak-ng as a subprocess (Popen), poll `cancel`
    while it runs, and call proc.terminate() if it is set."""
    print(f"[SPEAK] {text}")
    if SIMULATE_SPEECH_TIME:
        end = time.time() + len(text.split()) / SPEAK_WORDS_PER_S
        while time.time() < end:
            if cancel is not None and cancel.is_set():
                print("[interrupted]")
                return
            time.sleep(0.05)


class Speaker:
    """Two ways to get something spoken:
      submit()   one waiting slot. A newer sentence replaces an older one of the same or lower
                 priority; a more urgent one is never replaced by a less urgent one. Sentences
                 that waited longer than their TTL are dropped. Used for navigation and hazards.
      narrate()  an ordered backlog (spoken first-in first-out when the slot is empty) for
                 description mode, where sentences must not replace each other.
    A higher-priority sentence cuts off a lower-priority one being spoken right now, and any
    urgent submit() also discards the rest of a narration (the moment has passed)."""

    def __init__(self):
        self.cv = threading.Condition()
        self.pending = None       # (priority, text, created, ttl, kind)
        self.backlog = deque()    # same tuples, kind == "narration"
        self.speaking = False
        self.current_priority = 0
        self.current_kind = None
        self.cancel = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, text, priority=INFO, ttl=INFO_TTL_S, dedupe=True):
        if dedupe and is_repeat(text):
            return "repeat suppressed"
        with self.cv:
            if self.pending is not None and self.pending[0] > priority:
                return "outranked"
            replaced = self.pending is not None
            self.pending = (priority, text, time.time(), ttl, "alert" if priority > INFO else "info")
            if priority > INFO:
                self.backlog.clear()         # anything urgent ends a narration in progress
            if self.speaking and priority > self.current_priority:
                self.cancel.set()            # urgent: cut off the sentence being spoken
            self.cv.notify()
        return "replaced waiting sentence" if replaced else "queued"

    def narrate(self, text):
        with self.cv:
            self.backlog.append((INFO, text, time.time(), NARRATION_TTL_S, "narration"))
            self.cv.notify()

    def stop_narration(self):
        with self.cv:
            self.backlog.clear()
            if self.speaking and self.current_kind == "narration":
                self.cancel.set()

    def idle(self):
        with self.cv:
            return not self.speaking and self.pending is None and not self.backlog

    def _run(self):
        while True:
            with self.cv:
                while self.pending is None and not self.backlog:
                    self.cv.wait()
                if self.pending is not None:
                    item, self.pending = self.pending, None
                else:
                    item = self.backlog.popleft()
                priority, text, created, ttl, kind = item
                self.speaking = True
                self.current_priority = priority
                self.current_kind = kind
                self.cancel.clear()
            if time.time() - created > ttl:
                print(f"[dropped stale] {text}")
            else:
                mark_spoken(text)
                speak(text, self.cancel)
            with self.cv:
                self.speaking = False
                self.current_priority = 0
                self.current_kind = None


speaker = Speaker()


# ---------- modes ----------
NAV, DESCRIBE = "navigation", "description"
_mode = NAV
_mode_lock = threading.Lock()
describe_request = threading.Event()


def current_mode():
    with _mode_lock:
        return _mode


def request_describe():
    """Ask for a fresh, longer description of the current view (cuts off narration in progress)."""
    speaker.stop_narration()
    describe_request.set()


def set_mode(new):
    """The single hook for switching modes: keyboard now, a voice command or button later."""
    global _mode
    with _mode_lock:
        changed = new != _mode
        _mode = new
    if changed:
        speaker.stop_narration()
        speaker.submit(f"{new.capitalize()} mode.", priority=HAZARD, ttl=3.0, dedupe=False)
        print(f"(mode -> {new})")
    if new == DESCRIBE:
        request_describe()
    else:
        describe_request.clear()


def input_worker():
    print("(keys, then Enter:  n = navigation   d = description   r = describe again)")
    for line in sys.stdin:
        cmd = line.strip().lower()
        if cmd in ("n", "nav", "navigation"):
            set_mode(NAV)
        elif cmd in ("d", "desc", "description", "describe", "r", "again"):
            set_mode(DESCRIBE)       # entering (or re-entering) description mode starts a description


# ---------- VLM calls ----------
# Punctuation must be followed by whitespace, so "1.5 meters" is not split at the decimal.
# A final sentence with no trailing whitespace is handled when the stream reports done.
SENTENCE_END = re.compile(r"(.+?[.!?])\s", re.S)


def build_prompt():
    told = told_recently() if USE_TOLD_CONTEXT else []
    if not told:
        return PROMPT
    said = " ".join(f'"{t}"' for t in told)
    return (PROMPT + f" Already told the user: {said} If the scene is basically the same, "
            "reply exactly: No change. Otherwise describe what is different.")


def warm_up():
    """Load the model into memory before the first real frame (first call can take 5+ s)."""
    blank = np.zeros((64, 64, 3), np.uint8)
    payload = {
        "model": MODEL, "stream": False, "keep_alive": "30m", "think": False,
        "options": {"num_predict": 1},
        "messages": [{"role": "user", "content": "ok", "images": [to_b64_jpeg(blank)]}],
    }
    t0 = time.time()
    requests.post(OLLAMA_URL, json=payload, timeout=300).raise_for_status()
    print(f"(model warm after {time.time() - t0:.1f}s)")


def _payload(frame, prompt, max_tokens):
    return {
        "model": MODEL,
        "stream": True,
        "keep_alive": "30m",             # keep the model loaded in memory
        "think": False,                  # reasoning mode would eat the token budget
        "options": {"num_predict": max_tokens, "temperature": 0},
        "messages": [{"role": "user", "content": prompt,
                      "images": [to_b64_jpeg(downscale(frame))]}],
    }


def describe(frame):
    """Navigation mode: ONE short sentence, handed to the prioritizer as soon as it is complete."""
    buf = ""
    t0 = time.time()
    with requests.post(OLLAMA_URL, json=_payload(frame, build_prompt(), MAX_TOKENS),
                       stream=True, timeout=60) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            chunk = json.loads(line)
            buf += chunk.get("message", {}).get("content", "")
            m = SENTENCE_END.match(buf)
            sentence = None
            if m:
                sentence = m.group(1).strip()
            elif chunk.get("done") and buf.strip()[-1:] in (".", "!", "?"):
                sentence = buf.strip()
            if sentence:
                if sentence.lower().startswith("no change"):
                    status = "no change"
                else:
                    status = speaker.submit(sentence)
                print(f"(first sentence after {time.time() - t0:.2f}s, {status})")
                return   # leaving the with-block closes the stream and stops generation
            if chunk.get("done"):
                break
    print(f"(no complete sentence after {time.time() - t0:.2f}s; nothing spoken)")


def _stream_description(frame, prompt):
    """Stream one long description into the narration backlog, sentence by sentence.
    Returns "ok", "unclear" (the model gave its fallback phrase; nothing was queued) or
    "cancelled" (the user switched mode or asked again)."""
    buf, n, t0 = "", 0, time.time()

    def is_fallback(sentence):
        return n == 0 and sentence.lower().startswith("camera view unclear")

    with requests.post(OLLAMA_URL, json=_payload(frame, prompt, DESCRIBE_MAX_TOKENS),
                       stream=True, timeout=90) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            if current_mode() != DESCRIBE or describe_request.is_set():
                print("(description cancelled)")
                return "cancelled"
            chunk = json.loads(line)
            buf += chunk.get("message", {}).get("content", "")
            while (m := SENTENCE_END.match(buf)):
                sentence = m.group(1).strip()
                if is_fallback(sentence):
                    return "unclear"
                speaker.narrate(sentence)
                n += 1
                if n == 1:
                    print(f"(first description sentence after {time.time() - t0:.2f}s)")
                buf = buf[m.end():]
            if chunk.get("done"):
                tail = buf.strip()
                if tail[-1:] in (".", "!", "?"):
                    if is_fallback(tail):
                        return "unclear"
                    speaker.narrate(tail)
                    n += 1
                break
    print(f"(description: {n} sentences generated in {time.time() - t0:.1f}s)")
    return "ok"


def describe_long(frame):
    """Description mode: one longer description, spoken in order while it is still being
    generated. If the model answers "Camera view unclear" although the frame looks like a
    normal scene (not dark, not flat), the model is refusing rather than the camera failing,
    so retry once with a simpler prompt before telling the user."""
    cv2.imwrite("last_described.jpg", downscale(frame))     # debug: exactly what the model sees
    t = thumb(frame)
    if float(t.mean()) < DARK_THRESHOLD:
        speaker.submit("Camera view blocked.", priority=HAZARD, ttl=3.0, dedupe=False)
        return
    result = _stream_description(frame, DESCRIBE_PROMPT)
    if result == "unclear" and float(t.std()) >= FLAT_THRESHOLD:
        print(f"(model said unclear on a normal-looking frame "
              f"[brightness {t.mean():.0f}, contrast {t.std():.0f}]; retrying with the simple prompt)")
        result = _stream_description(frame, SIMPLE_DESCRIBE_PROMPT)
    if result == "unclear":
        print(f"(still unclear [brightness {t.mean():.0f}, contrast {t.std():.0f}]; "
              f"check last_described.jpg)")
        speaker.narrate("Camera view unclear.")


# ---------- main loop ----------
def run(cam):
    last_sent_thumb, last_sent_time = None, 0.0

    while True:
        frame = cam.get()
        if getattr(cam, "done", False):
            while not speaker.idle():
                time.sleep(0.2)      # let the last sentence finish speaking
            print("Video finished.")
            break
        if frame is None:
            time.sleep(0.05)
            continue

        if current_mode() == DESCRIBE:
            last_sent_thumb = None   # so navigation describes the scene right away when re-entered
            if describe_request.is_set():
                describe_request.clear()
                describe_long(frame)
            else:
                time.sleep(0.05)
            continue

        now = time.time()
        t = thumb(frame)
        due = now - last_sent_time >= MIN_INTERVAL_S
        changed = last_sent_thumb is None or scene_changed(t, last_sent_thumb)
        stale = now - last_sent_time >= MAX_INTERVAL_S

        if due and (changed or stale):
            last_sent_thumb, last_sent_time = t, now
            cv2.imwrite("last_sent.jpg", downscale(frame))   # debug: exactly what the model sees
            if float(t.mean()) < DARK_THRESHOLD:
                status = speaker.submit("Camera view blocked.", HAZARD)   # code decides, not the model
                print(f"(dark frame, {status})")
            else:
                describe(frame)      # blocks: only one VLM call in flight at a time
        else:
            time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="replay this video file instead of a live camera")
    ap.add_argument("--frame-file", help="read frames another process saves here (e.g. /dev/shm/latest.jpg)")
    ap.add_argument("--fused", help="follow the fuser's JSONL output (e.g. /dev/shm/fused.jsonl)")
    ap.add_argument("--mode", choices=[NAV, DESCRIBE], default=NAV, help="starting mode")
    args = ap.parse_args()

    warm_up()
    if args.fused:
        import fastpath
        fastpath.start(args.fused, speaker.submit, HAZARD)
        print(f"(fast path following {args.fused})")
    # Camera objects are created AFTER warm-up so a replayed clip doesn't start playing while
    # the model loads.
    if args.video:
        cam = VideoFrame(args.video)
    elif args.frame_file:
        cam = SharedFrame(args.frame_file)
    else:
        cam = LatestFrame(CAMERA_INDEX)
    threading.Thread(target=input_worker, daemon=True).start()
    if args.mode == DESCRIBE:
        set_mode(DESCRIBE)
    run(cam)


if __name__ == "__main__":
    main()
