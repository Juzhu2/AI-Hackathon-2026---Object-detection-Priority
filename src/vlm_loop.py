"""Camera (live, shared-frame file, or video) -> Ollama VLM -> prioritizer -> speech,
plus an optional fast path driven by the geometry fuser's output.

Two modes (switch while running: type a key, then Enter):
  n  navigation mode   short, continuous, path-focused sentences (walking / outdoors)
  d  description mode  one longer, ordered description of the view, on demand (user is safe)
  r  describe again    a fresh description (switches to description mode if needed)
  m  tell me more      more detail about the SAME view, without repeating what was said
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
         --describe-model gemma4:26b                         richer descriptions (slower; on demand)
         --tts-cmd 'python say.py --text {text} --out {out}' real speech via a file-writing TTS
"""
import argparse
import base64
import queue
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

# Description mode (on demand, user standing still: latency matters less than detail)
DESCRIBE_MODEL = MODEL      # --describe-model gemma4:26b for richer descriptions
DESCRIBE_MAX_TOKENS = 450   # roughly 8-10 sentences
DESCRIBE_MAX_SIDE = 896     # Gemma 3's vision encoder works at 896 px: small text and objects survive
MORE_MAX_TOKENS = 300
DESCRIBE_SECTIONS = True    # after the overview, describe the left and right halves as zoomed
                            # crops: a small VLM sees ~2x the detail and says more
SECTION_MAX_TOKENS = 260
SECTION_OVERLAP = 0.10      # each half extends 10% past the middle so nothing on the seam is lost
USE_FUSED_HINTS = True      # give the model the fuser's measured distances for its objects
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
    "You are describing a scene to a blind person who is standing still, is safe, and wants a "
    "detailed picture of what is in front of them. Speak naturally in 8 to 10 sentences, about "
    "150 words, in this order. First, what kind of place this is, its overall layout and the "
    "lighting. Then sweep from left to right: for each main object say where it is (left, "
    "ahead, right; near or far), and its size, color, material and state, for example open or "
    "closed, on or off, full or empty. Then read any visible text, signs, labels or screens "
    "word for word. End with where the open space or clear path is. "
    "Describe only what you can actually see; if you are not sure what something is, say so. "
    "For people, say only that a person is there and where; never describe faces, clothing or "
    "names, and never say what anyone is doing or feeling. "
    "Only if the image is completely dark or blank, reply exactly: Camera view unclear."
)

OVERVIEW_PROMPT = (
    "You are describing a scene to a blind person who is standing still, is safe, and wants a "
    "detailed picture of what is in front of them. In 4 to 5 sentences: say what kind of place "
    "this is, its overall layout and the lighting, then name the main things in view with where "
    "they are (left, ahead, right; near or far), and end with where the open space or clear path "
    "is. Details of the left and right sides come later, so keep this an overview. "
    "Describe only what you can actually see. "
    "People: if anyone is in view, mention them once, briefly (for example: a person is "
    "standing ahead), and then describe the place and objects instead; never describe faces, "
    "clothing, names or what anyone is doing. "
    "Only if the image is completely dark or blank, reply exactly: Camera view unclear."
)

SECTION_PROMPT = (
    "This image is the {side} part of the view in front of a blind person. Describe what is in "
    "it in detail, from nearest to farthest: what each thing is, its size, color, material and "
    "state (open or closed, on or off, full or empty), and read any text, labels or screens "
    "word for word. Begin with: On your {side},  Use 3 to 5 sentences. "
    "Already told the user: \"{told}\" Do not repeat that; add new details only. "
    "Only describe what you can actually see. Do not mention people at all here: describe the "
    "furniture, objects, surfaces, text and space around them. If there is nothing new to add, reply exactly: "
    "Nothing more on the {side}."
)

MORE_PROMPT = (
    "Tell me more about the same image: smaller objects, textures, patterns, things in the "
    "background, and any text you did not read yet. 4 to 6 sentences. Do not talk about people. "
    "Do not repeat anything "
    "you already said. Only describe what you can actually see; if there is nothing more, "
    "say: That is everything I can see."
)


def fused_hint():
    """Measured distances from the fuser, as one sentence for the description prompt."""
    if not USE_FUSED_HINTS:
        return ""
    try:
        import fastpath
        items = fastpath.scene_summary()
    except Exception:
        return ""
    if not items:
        return ""
    return (" Distance sensors measured these objects (they can be wrong): " + "; ".join(items)
            + ". When you mention one of these, use its distance. Do not mention an object "
              "only because it is in this list, and do not guess distances for anything else.")

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


# ---------- UI events (only collected when the window is open) ----------
UI_EVENTS = None          # queue.Queue set by main() with --ui


def emit(kind, text="", priority=0.0):
    """kind: speak | interrupted | mode | status. The UI drains UI_EVENTS every 100 ms."""
    if UI_EVENTS is not None:
        UI_EVENTS.put((time.time(), kind, text, priority))


# ---------- speech + prioritizer ----------
TTS = None                # set in main() from --tts-cmd (tts_files.FileTTS)

URGENT_PRIORITY = 2.25    # fast-path urgency 3 arrives as HAZARD + 0.3 = 2.3: beep first


def speak(text, cancel=None, priority=0):
    """Speak one sentence. `cancel` is an Event: if it gets set, stop talking immediately.
    Real TTS: start Piper/Kokoro/espeak-ng as a subprocess (Popen), poll `cancel`
    while it runs, and call proc.terminate() if it is set."""
    print(f"[SPEAK] {text}")
    emit("speak", text, priority)
    if TTS is not None:
        status = TTS.speak(text, cancel, urgent=priority >= URGENT_PRIORITY)
        if status == "cancelled":
            print("[interrupted]")
            emit("interrupted", text)
        return
    if SIMULATE_SPEECH_TIME:
        end = time.time() + len(text.split()) / SPEAK_WORDS_PER_S
        while time.time() < end:
            if cancel is not None and cancel.is_set():
                print("[interrupted]")
                emit("interrupted", text)
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
        self.resume_narration = False   # description mode: an alert PAUSES the narration
        self._drop_current = False      # set by stop_narration(): do not resume this sentence
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, text, priority=INFO, ttl=INFO_TTL_S, dedupe=True):
        if dedupe and is_repeat(text):
            return "repeat suppressed"
        with self.cv:
            if self.pending is not None and self.pending[0] > priority:
                return "outranked"
            replaced = self.pending is not None
            self.pending = (priority, text, time.time(), ttl, "alert" if priority > INFO else "info")
            if priority > INFO and self.backlog and not self.resume_narration:
                self.backlog.clear()         # navigation: anything urgent ends a narration
                if TTS is not None:
                    TTS.discard()
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
            if self.current_kind == "narration":
                self._drop_current = True
            if TTS is not None:
                TTS.discard()
            if self.speaking and self.current_kind == "narration":
                self.cancel.set()

    def hush(self):
        """Stop talking now: drop what is waiting and cut off the current sentence."""
        with self.cv:
            self.backlog.clear()
            self.pending = None
            self._drop_current = True
            if TTS is not None:
                TTS.discard()
            if self.speaking:
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
                if TTS is not None and self.backlog:
                    TTS.prepare(self.backlog[0][1])   # synthesize the next sentence while this one plays
                self.speaking = True
                self.current_priority = priority
                self.current_kind = kind
                self._drop_current = False
                self.cancel.clear()
            if time.time() - created > ttl:
                print(f"[dropped stale] {text}")
            else:
                mark_spoken(text)
                speak(text, self.cancel, priority)
            with self.cv:
                if (kind == "narration" and self.cancel.is_set() and self.resume_narration
                        and not self._drop_current):
                    # cut off by an alert: say this sentence again once the alert is done
                    self.backlog.appendleft((priority, text, time.time(), ttl, kind))
                    print("(narration paused for an alert; will resume)")
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


more_request = threading.Event()
cancel_generation = threading.Event()   # "Stop talking": also stop generating the description


def hush():
    cancel_generation.set()
    speaker.hush()
    emit("status", "Stopped talking.")
last_description = {}        # {"image": b64, "prompt": str, "text": str} of the latest description


def request_more():
    """More detail about the view that was just described (needs a previous description)."""
    if current_mode() != DESCRIBE or not last_description:
        speaker.submit("Ask for a description first.", priority=HAZARD, ttl=3.0, dedupe=False)
        return
    speaker.stop_narration()
    more_request.set()


def set_mode(new):
    """The single hook for switching modes: keyboard now, a voice command or button later."""
    global _mode
    with _mode_lock:
        changed = new != _mode
        _mode = new
    speaker.resume_narration = new == DESCRIBE
    try:
        import fastpath
        fastpath.set_description_mode(new == DESCRIBE)
    except Exception:
        pass
    if changed:
        speaker.stop_narration()
        speaker.submit(f"{new.capitalize()} mode.", priority=HAZARD, ttl=3.0, dedupe=False)
        print(f"(mode -> {new})")
    emit("mode", new)
    if new == DESCRIBE:
        request_describe()
    else:
        describe_request.clear()


def input_worker():
    print("(keys, then Enter:  n = navigation   d = description   r = describe again   m = more)")
    for line in sys.stdin:
        cmd = line.strip().lower()
        if cmd in ("n", "nav", "navigation"):
            set_mode(NAV)
        elif cmd in ("d", "desc", "description", "describe", "r", "again"):
            set_mode(DESCRIBE)       # entering (or re-entering) description mode starts a description
        elif cmd in ("m", "more"):
            request_more()


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
    """Load the model(s) into memory before the first real frame (first call can take 5+ s)."""
    for model in dict.fromkeys([MODEL, DESCRIBE_MODEL]):
        _warm(model)


def _warm(model):
    blank = np.zeros((64, 64, 3), np.uint8)
    payload = {
        "model": model, "stream": False, "keep_alive": "30m", "think": False,
        "options": {"num_predict": 1},
        "messages": [{"role": "user", "content": "ok", "images": [to_b64_jpeg(blank)]}],
    }
    t0 = time.time()
    requests.post(OLLAMA_URL, json=payload, timeout=300).raise_for_status()
    print(f"({model} warm after {time.time() - t0:.1f}s)")


def _payload(frame, prompt, max_tokens, model=None, max_side=MAX_SIDE, messages=None):
    if messages is None:
        messages = [{"role": "user", "content": prompt,
                     "images": [to_b64_jpeg(downscale(frame, max_side))]}]
    return {
        "model": model or MODEL,
        "stream": True,
        "keep_alive": "30m",             # keep the model loaded in memory
        "think": False,                  # reasoning mode would eat the token budget
        "options": {"num_predict": max_tokens, "temperature": 0},
        "messages": messages,
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


def _stream_description(frame, prompt, messages=None, max_tokens=DESCRIBE_MAX_TOKENS):
    """Stream one long description into the narration backlog, sentence by sentence.
    Returns "ok", "unclear" (the model gave its fallback phrase; nothing was queued) or
    "cancelled" (the user switched mode or asked again)."""
    buf, n, t0 = "", 0, time.time()
    spoken = []

    def is_fallback(sentence):
        return n == 0 and sentence.lower().startswith("camera view unclear")

    def say(sentence):
        if sentence.lower().startswith(("nothing more on the", "that is everything")) and n > 0:
            return False
        speaker.narrate(sentence)
        spoken.append(sentence)
        return True

    payload = _payload(frame, prompt, max_tokens, DESCRIBE_MODEL, DESCRIBE_MAX_SIDE, messages)
    if messages is None:
        last_description.clear()
        last_description.update(image=payload["messages"][0]["images"][0], prompt=prompt, text="")
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=180) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            if (current_mode() != DESCRIBE or describe_request.is_set() or more_request.is_set()
                    or cancel_generation.is_set()):
                print("(description cancelled)")
                return "cancelled"
            chunk = json.loads(line)
            buf += chunk.get("message", {}).get("content", "")
            while (m := SENTENCE_END.match(buf)):
                sentence = m.group(1).strip()
                if is_fallback(sentence):
                    return "unclear"
                if not say(sentence):
                    buf = buf[m.end():]
                    continue
                n += 1
                if n == 1:
                    print(f"(first description sentence after {time.time() - t0:.2f}s)")
                buf = buf[m.end():]
            if chunk.get("done"):
                tail = buf.strip()
                if tail[-1:] in (".", "!", "?"):
                    if is_fallback(tail):
                        return "unclear"
                    if say(tail):
                        n += 1
                break
    if messages is None:
        last_description["text"] = " ".join(spoken)
    else:
        last_description["text"] += " " + " ".join(spoken)
    print(f"(description: {n} sentences generated in {time.time() - t0:.1f}s)")
    return "ok"


def describe_long(frame):
    """Description mode: one longer description, spoken in order while it is still being
    generated. If the model answers "Camera view unclear" although the frame looks like a
    normal scene (not dark, not flat), the model is refusing rather than the camera failing,
    so retry once with a simpler prompt before telling the user."""
    cancel_generation.clear()
    emit("status", "Describing the view...")
    cv2.imwrite("last_described.jpg", downscale(frame))     # debug: exactly what the model sees
    t = thumb(frame)
    if float(t.mean()) < DARK_THRESHOLD:
        speaker.submit("Camera view blocked.", priority=HAZARD, ttl=3.0, dedupe=False)
        return
    prompt = (OVERVIEW_PROMPT if DESCRIBE_SECTIONS else DESCRIBE_PROMPT) + fused_hint()
    result = _stream_description(frame, prompt)
    if result == "ok" and DESCRIBE_SECTIONS:
        _describe_sections(frame)
        return
    if result == "unclear" and float(t.std()) >= FLAT_THRESHOLD:
        print(f"(model said unclear on a normal-looking frame "
              f"[brightness {t.mean():.0f}, contrast {t.std():.0f}]; retrying with the simple prompt)")
        result = _stream_description(frame, SIMPLE_DESCRIBE_PROMPT)
    if result == "unclear":
        print(f"(still unclear [brightness {t.mean():.0f}, contrast {t.std():.0f}]; "
              f"check last_described.jpg)")
        speaker.narrate("Camera view unclear.")


def _describe_sections(frame):
    """Left and right halves as zoomed crops, each told what was already said.
    Generation of the next part overlaps with speaking the previous one."""
    h, w = frame.shape[:2]
    half = w // 2
    pad = int(w * SECTION_OVERLAP)
    crops = [("left", frame[:, :half + pad]), ("right", frame[:, half - pad:])]
    for side, crop in crops:
        if (current_mode() != DESCRIBE or describe_request.is_set() or more_request.is_set()
                or cancel_generation.is_set()):
            return
        told = last_description.get("text", "")[-600:].replace('"', "'")
        prompt = SECTION_PROMPT.format(side=side, told=told)
        messages = [{"role": "user", "content": prompt,
                     "images": [to_b64_jpeg(downscale(crop, DESCRIBE_MAX_SIDE))]}]
        if _stream_description(None, None, messages=messages,
                               max_tokens=SECTION_MAX_TOKENS) == "cancelled":
            return


def describe_more():
    """Follow-up on the SAME image: the model sees its own previous answer, so it can skip it."""
    d = dict(last_description)
    if not d.get("text"):
        return
    cancel_generation.clear()
    emit("status", "Looking for more detail...")
    messages = [{"role": "user", "content": d["prompt"], "images": [d["image"]]},
                {"role": "assistant", "content": d["text"]},
                {"role": "user", "content": MORE_PROMPT}]
    _stream_description(None, None, messages=messages, max_tokens=MORE_MAX_TOKENS)


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
                more_request.clear()
                describe_long(frame)
            elif more_request.is_set():
                more_request.clear()
                describe_more()
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
    ap.add_argument("--describe-model", help="model for description mode (e.g. gemma4:26b)")
    ap.add_argument("--ui", action="store_true", help="open the window (camera view, text, buttons)")
    ap.add_argument("--view", help="image the window shows (default: the --frame-file); "
                                   "e.g. /dev/shm/annotated.jpg for boxes and distances")
    ap.add_argument("--tts-cmd", help="file-writing TTS command with {text} and {out} placeholders")
    args = ap.parse_args()

    global DESCRIBE_MODEL, TTS, UI_EVENTS
    if args.describe_model:
        DESCRIBE_MODEL = args.describe_model
    if args.tts_cmd:
        from tts_files import FileTTS
        TTS = FileTTS(cmd=args.tts_cmd)
        print(f"(TTS: {args.tts_cmd} | player: {' '.join(TTS.player or ['none found'])})")
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
    if args.ui:
        UI_EVENTS = queue.Queue()
        import ui                                    # Tk must own the main thread
        threading.Thread(target=run, args=(cam,), daemon=True).start()
        if args.mode == DESCRIBE:
            set_mode(DESCRIBE)
        else:
            emit("mode", NAV)
        ui.App(sys.modules[__name__], view_path=args.view or args.frame_file,
               frame_source=cam.get).mainloop()
        if TTS is not None:
            TTS.sweep(0)
        return
    if args.mode == DESCRIBE:
        set_mode(DESCRIBE)
    run(cam)


if __name__ == "__main__":
    main()
