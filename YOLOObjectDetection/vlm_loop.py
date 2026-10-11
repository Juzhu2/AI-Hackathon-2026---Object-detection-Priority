"""Camera (live, shared-frame file, or video) -> Ollama VLM -> speaker,
plus the FAST PATH (fastpath.py + prioritizer.py) driven by the fuser's JSONL output.

Architecture
  Fast path  (deterministic, every frame, the ONLY thing responsible for safety):
             detections + depth + geometry -> prioritizer -> speaker.  Works with no VLM at all.
  Slow path  (this file's VLM calls, seconds): only
             - on request:  d / r = describe the view, or type any question and press Enter
             - on a trigger (optional, --vlm-auto): the scene changed a lot AND no hazard was
               just announced
             If the VLM is slow, wrong or dead, hazard warnings are unaffected.
  Router     (router.py): decides for each typed request whether the sensors can answer it
             ("what's in front of me?"), the VLM must ("what does the sign say?"), or both
             ("what's around me?": measured summary first, then the VLM description).

Keys (type, then Enter):
  n            navigation mode: fast path only (VLM stays silent unless --vlm-auto)
  d / r        description mode: one longer, ordered description of the view (r = again)
  anything else  is routed: sensors, VLM, or both ("is there a door?", "what's around me?")
Hazards stay ON in every mode and cut off whatever is being said.

Priorities (higher = more urgent): CRITICAL 3 > WARNING 2 > INFO 1.
  CRITICAL cuts off speech and discards a narration. WARNING cuts off the current sentence,
  and an interrupted narration sentence is repeated afterwards.

Install: pip install opencv-python-headless requests numpy   (and: sudo apt install espeak-ng)
Run:     python vlm_loop.py                                  live webcam
         python vlm_loop.py clip.mp4                         replay a recording in real time
         python vlm_loop.py --frame-file /dev/shm/latest.jpg frames saved by the fuser process
Add:     --fused /dev/shm/fused.jsonl                        enable the fast path
         --mode description                                  start in description mode
         --vlm-auto                                          scene-change background descriptions
"""
import argparse
import base64
import json
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests

import router

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"       # also tested: gemma4:26b (needs "think": False)
CAMERA_INDEX = 0

MAX_SIDE = 512            # downscale long side to this many pixels
MAX_TOKENS = 40           # navigation (scene-change) call: hard cap on output length
CHANGE_THRESHOLD = 12.0   # mean pixel difference (0-255) that counts as "changed"
DARK_THRESHOLD = 20.0     # mean brightness (0-255) below this = camera blocked/dark

# Slow-path trigger (only used with --vlm-auto)
AUTO_SCENE = False          # set by --vlm-auto
AUTO_MIN_INTERVAL_S = 25.0  # never run a background description more often than this
QUIET_AFTER_ALERT_S = 6.0   # stay silent this long after the fast path warned about something

# Description / question calls
DESCRIBE_MAX_TOKENS = 200   # roughly 4-5 sentences
ASK_MAX_TOKENS = 80         # 1-2 sentences
NARRATION_TTL_S = 120.0     # narration sentences wait in order; they do not go stale quickly

# Speaker settings
COOLDOWN_S = 10.0         # the same thing may be said again after this many seconds
SIMILARITY = 0.6          # fraction of shared object words that counts as "the same thing"
INFO_TTL_S = 5.0          # a waiting description older than this is dropped as stale
CRITICAL, WARNING, INFO = 3, 2, 1   # higher number = more urgent
HAZARD = WARNING                    # mode announcements, "Camera view blocked"
USE_TOLD_CONTEXT = False  # tell the model what was already said. Can push it to invent
                          # "new" things, so test with it off first, then A/B it.

# Real speech through espeak-ng when it is installed; otherwise pretend speaking takes time
# so backlog behavior is still realistic. Replace speak() to use Piper/Kokoro instead.
ESPEAK = shutil.which("espeak-ng")
SIMULATE_SPEECH_TIME = ESPEAK is None
SPEAK_WORDS_PER_S = 2.5

NO_SAFETY_CLAIM = "Never say the path is clear or safe to walk; you cannot know that."

PROMPT = (
    "You are the eyes for a blind person. Say the most important things you can "
    "clearly see right now, with direction (left, ahead, right). One sentence, "
    "maximum 12 words. Only mention things that are definitely visible; never "
    "guess or invent. You see a single still image, so never describe motion or "
    "what anyone is doing or intending. For people, say only that a person is "
    "present and where; never describe clothing or faces, and never read names "
    "or badges. " + NO_SAFETY_CLAIM + " If the image is dark, blocked, blurry, or unclear, "
    "reply exactly: Camera view unclear."
)

DESCRIBE_PROMPT = (
    "Describe this scene for a blind person who is standing still and is safe. Use 4 to 5 "
    "short sentences, about 60 words: first the overall setting, then the main objects and "
    "where they are (left, right, ahead, near, far), then any readable signs or text, then "
    "notable colors. Describe only what you can see. Do not mention faces, clothing or "
    "names, and do not say what anyone is doing. " + NO_SAFETY_CLAIM + " "
    "Only if the image is completely dark or blank, reply exactly: Camera view unclear."
)

ASK_PROMPT = (
    "A blind person asks about what the camera sees: \"{q}\" "
    "Answer in at most 2 short sentences, using only what is clearly visible, with direction "
    "(left, ahead, right) where relevant. If you cannot tell from the image, say so. "
    "Never describe faces, clothing or names. " + NO_SAFETY_CLAIM + " {ctx}"
)

# Used for one retry if the model answers "Camera view unclear" on a frame that looks normal.
SIMPLE_DESCRIBE_PROMPT = (
    "Describe this image in 4 short sentences for a blind person. "
    "Say where things are (left, right, ahead)."
)
FLAT_THRESHOLD = 8.0        # gray-level spread below this = blank wall / lens cap, not a normal scene

fast = None                 # the FastPath thread, set in main() when --fused is given

# Marker queued in question_queue by the router's "hybrid" decision: describe the current view
# once, without switching to description mode.
DESCRIBE_ONCE = object()


def sensor_context():
    """Distances the fast path measured, to keep VLM answers consistent with the warnings."""
    return fast.context() if fast is not None else ""


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
    Uses espeak-ng when installed (killed on cancel); otherwise simulates the time it takes."""
    print(f"[SPEAK] {text}")
    if ESPEAK:
        proc = subprocess.Popen([ESPEAK, "-s", "185", "--stdin"], stdin=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        try:
            proc.stdin.write(text.encode("utf-8"))
            proc.stdin.close()
        except BrokenPipeError:
            pass
        while proc.poll() is None:
            if cancel is not None and cancel.is_set():
                proc.terminate()
                proc.wait()
                print("[interrupted]")
                return
            time.sleep(0.02)
        return
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
                 that waited longer than their TTL are dropped. Used for hazards and short info.
      narrate()  an ordered backlog (spoken first-in first-out when the slot is empty) for
                 descriptions and answers, where sentences must not replace each other.
    A higher-priority sentence cuts off a lower-priority one being spoken right now.
    A CRITICAL submit also discards the rest of a narration (the moment has passed); a WARNING
    only interrupts the sentence in progress, which is repeated once the warning is spoken."""

    def __init__(self):
        self.cv = threading.Condition()
        self.pending = None       # (priority, text, created, ttl, kind)
        self.backlog = deque()    # same tuples, kind == "narration"
        self.speaking = False
        self.current_priority = 0
        self.current_kind = None
        self.abandon = False      # the interrupted narration sentence must NOT be repeated
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
            if priority >= CRITICAL:
                self.backlog.clear()         # a critical alert ends a narration in progress
                self.abandon = True
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
                self.abandon = True
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
                self.abandon = False
                self.cancel.clear()
            if time.time() - created > ttl:
                print(f"[dropped stale] {text}")
            else:
                mark_spoken(text)
                speak(text, self.cancel)
            with self.cv:
                if self.cancel.is_set() and kind == "narration" and not self.abandon:
                    self.backlog.appendleft(item)    # interrupted by a warning: say it again after
                self.speaking = False
                self.current_priority = 0
                self.current_kind = None


speaker = Speaker()


# ---------- modes and requests ----------
NAV, DESCRIBE = "navigation", "description"
_mode = NAV
_mode_lock = threading.Lock()
describe_request = threading.Event()
question_queue = queue.Queue()


def current_mode():
    with _mode_lock:
        return _mode


def request_describe():
    """Ask for a fresh, longer description of the current view (cuts off narration in progress)."""
    speaker.stop_narration()
    describe_request.set()


def ask(question):
    """The single hook for user questions: keyboard now, speech-to-text or a button later.
    Normally a string for the VLM; DESCRIBE_ONCE asks for a one-off scene description."""
    question_queue.put(question)


def route_request(raw):
    """Typed (later: spoken) request -> fast path, slow path, or both.
    The router decides; this function only carries it out."""
    d = router.route(raw, fast.latest if fast is not None else None)
    print(f"(route: {d.kind})")
    if d.kind in ("sensor", "hybrid"):
        speaker.stop_narration()
        if d.speak:
            speaker.narrate(d.speak)           # measured facts first: instant, no VLM
    if d.kind == "hybrid":
        ask(DESCRIBE_ONCE)                     # the VLM description follows
    elif d.kind == "vlm":
        ask(d.question)


def set_mode(new):
    """The single hook for switching modes: keyboard now, a voice command or button later."""
    global _mode
    with _mode_lock:
        changed = new != _mode
        _mode = new
    if changed:
        speaker.stop_narration()
        speaker.submit(f"{new.capitalize()} mode.", priority=WARNING, ttl=3.0, dedupe=False)
        print(f"(mode -> {new})")
    if new == DESCRIBE:
        request_describe()
    else:
        describe_request.clear()


def input_worker():
    print("(type then Enter:  n = navigation   d = description   r = describe again   "
          "or any question, e.g. 'is there a door?' / 'what's in front of me?')")
    for line in sys.stdin:
        raw = line.strip()
        cmd = raw.lower()
        if cmd in ("n", "nav", "navigation"):
            set_mode(NAV)
        elif cmd in ("d", "desc", "description", "describe", "r", "again"):
            set_mode(DESCRIBE)       # entering (or re-entering) description mode starts a description
        elif raw:
            route_request(raw)


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
    """Scene-change (background) call: ONE short sentence, spoken at INFO priority."""
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


def _stream_description(frame, prompt, max_tokens=DESCRIBE_MAX_TOKENS, bound_to_mode=True):
    """Stream one answer into the narration backlog, sentence by sentence.
    Returns "ok", "unclear" (the model gave its fallback phrase; nothing was queued) or
    "cancelled" (the user switched mode, asked again, or asked a new question)."""
    buf, n, t0 = "", 0, time.time()

    def is_fallback(sentence):
        return n == 0 and sentence.lower().startswith("camera view unclear")

    def cancelled():
        return (describe_request.is_set() or not question_queue.empty()
                or (bound_to_mode and current_mode() != DESCRIBE))

    with requests.post(OLLAMA_URL, json=_payload(frame, prompt, max_tokens),
                       stream=True, timeout=90) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            if cancelled():
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
                    print(f"(first sentence after {time.time() - t0:.2f}s)")
                buf = buf[m.end():]
            if chunk.get("done"):
                tail = buf.strip()
                if tail[-1:] in (".", "!", "?"):
                    if is_fallback(tail):
                        return "unclear"
                    speaker.narrate(tail)
                    n += 1
                break
    print(f"(answer: {n} sentences generated in {time.time() - t0:.1f}s)")
    return "ok"


def describe_long(frame, bound_to_mode=True):
    """One longer description, spoken in order while it is still being generated.
    bound_to_mode=True: description mode (cancelled if the user leaves the mode).
    bound_to_mode=False: a one-off from the router's "hybrid" decision, in any mode.
    If the model answers "Camera view unclear" although the frame looks like a
    normal scene (not dark, not flat), the model is refusing rather than the camera failing,
    so retry once with a simpler prompt before telling the user."""
    cv2.imwrite("last_described.jpg", downscale(frame))     # debug: exactly what the model sees
    t = thumb(frame)
    if float(t.mean()) < DARK_THRESHOLD:
        speaker.submit("Camera view blocked.", priority=WARNING, ttl=3.0, dedupe=False)
        return
    ctx = sensor_context()
    result = _stream_description(frame, DESCRIBE_PROMPT + (" " + ctx if ctx else ""),
                                 bound_to_mode=bound_to_mode)
    if result == "unclear" and float(t.std()) >= FLAT_THRESHOLD:
        print(f"(model said unclear on a normal-looking frame "
              f"[brightness {t.mean():.0f}, contrast {t.std():.0f}]; retrying with the simple prompt)")
        result = _stream_description(frame, SIMPLE_DESCRIBE_PROMPT, bound_to_mode=bound_to_mode)
    if result == "unclear":
        print(f"(still unclear [brightness {t.mean():.0f}, contrast {t.std():.0f}]; "
              f"check last_described.jpg)")
        speaker.narrate("Camera view unclear.")


def answer_question(frame, question):
    """User asked something ("what does the sign say?"). Works in any mode; does not change the mode."""
    speaker.stop_narration()                                # drop an older answer or description
    cv2.imwrite("last_described.jpg", downscale(frame))
    if float(thumb(frame).mean()) < DARK_THRESHOLD:
        speaker.submit("Camera view blocked.", priority=WARNING, ttl=3.0, dedupe=False)
        return
    speaker.submit("Looking.", priority=INFO, ttl=3.0, dedupe=False)    # instant acknowledgement
    ctx = sensor_context()
    prompt = ASK_PROMPT.format(q=question.replace('"', "'"), ctx=ctx)
    print(f"(question: {question})")
    _stream_description(frame, prompt, max_tokens=ASK_MAX_TOKENS, bound_to_mode=False)


# ---------- main loop ----------
def safe_vlm(fn, *args):
    """Run a VLM call; if it fails, say so briefly and carry on. Never lets the VLM stop the loop."""
    try:
        fn(*args)
    except Exception as error:
        print(f"(VLM error: {error!r})")
        speaker.submit("Description unavailable.", priority=INFO, ttl=3.0)


def quiet_enough():
    """Background descriptions wait until the fast path has not warned for a few seconds."""
    return fast is None or time.monotonic() - fast.last_alert_mono > QUIET_AFTER_ALERT_S


def run(cam):
    last_sent_thumb, last_sent_time, last_check = None, 0.0, 0.0

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

        if not question_queue.empty():
            item = question_queue.get()
            if item is DESCRIBE_ONCE:
                safe_vlm(describe_long, frame, False)
            else:
                safe_vlm(answer_question, frame, item)
            continue

        if current_mode() == DESCRIBE:
            last_sent_thumb = None   # so navigation re-baselines when re-entered
            if describe_request.is_set():
                describe_request.clear()
                safe_vlm(describe_long, frame)
            else:
                time.sleep(0.05)
            continue

        # Navigation mode: the fast path speaks. Here we only do cheap checks once per second.
        now = time.time()
        if now - last_check < 1.0:
            time.sleep(0.05)
            continue
        last_check = now
        t = thumb(frame)
        if float(t.mean()) < DARK_THRESHOLD:                 # code decides, not the model
            print(f"(dark frame, {speaker.submit('Camera view blocked.', WARNING)})")
            continue
        if not AUTO_SCENE:
            continue
        if last_sent_thumb is None:                          # first frame: just set the baseline
            last_sent_thumb, last_sent_time = t, now
            continue
        if (now - last_sent_time >= AUTO_MIN_INTERVAL_S and scene_changed(t, last_sent_thumb)
                and quiet_enough()):
            last_sent_thumb, last_sent_time = t, now
            cv2.imwrite("last_sent.jpg", downscale(frame))   # debug: exactly what the model sees
            safe_vlm(describe, frame)      # blocks this loop only; the fast path runs in its own thread


def main():
    global fast, AUTO_SCENE
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?", help="replay this video file instead of a live camera")
    ap.add_argument("--frame-file", help="read frames another process saves here (e.g. /dev/shm/latest.jpg)")
    ap.add_argument("--fused", help="follow the fuser's JSONL output (e.g. /dev/shm/fused.jsonl)")
    ap.add_argument("--mode", choices=[NAV, DESCRIBE], default=NAV, help="starting mode")
    ap.add_argument("--vlm-auto", action="store_true",
                    help="also describe the scene in the background when it changes a lot")
    args = ap.parse_args()
    AUTO_SCENE = args.vlm_auto

    if args.fused:
        import fastpath
        # Start the fast path FIRST: hazard warnings must not wait for the VLM to load.
        fast = fastpath.start(args.fused, speaker.submit)
        print(f"(fast path following {args.fused})")
    try:
        warm_up()
    except Exception as error:      # the VLM being down must not stop the fast path
        print(f"(VLM unavailable: {error!r}; fast path keeps running)")
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