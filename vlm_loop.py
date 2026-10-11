"""Webcam (or video file) -> Ollama VLM -> prioritizer -> speech.

Pipeline features:
  - downscaled frames, capped output, keep_alive, warm-up call at startup
  - frames sent only when the scene changed (plus a periodic refresh)
  - the model is told what was already said, so it only reports what is new
  - a prioritizer: one waiting slot (newest wins), hazards outrank descriptions,
    stale sentences are dropped, near-duplicates are suppressed for COOLDOWN_S

Install: pip install opencv-python-headless requests numpy
Run:     python vlm_loop.py            (live webcam)
         python vlm_loop.py clip.mp4   (replay a recording in real time)
"""
import base64
import json
import re
import sys
import threading
import time

import cv2
import numpy as np
import requests

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "gemma3:4b"       # also tested: gemma4:26b (needs "think": False)
CAMERA_INDEX = 0

MAX_SIDE = 512            # downscale long side to this many pixels
MAX_TOKENS = 40           # hard cap on output length
CHANGE_THRESHOLD = 12.0   # mean pixel difference (0-255) that counts as "changed"
MIN_INTERVAL_S = 1.0      # never call more often than this
MAX_INTERVAL_S = 8.0      # force a refresh at least this often
DARK_THRESHOLD = 20.0     # mean brightness (0-255) below this = camera blocked/dark

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
    "guess or invent. If the image is dark, blocked, blurry, or unclear, "
    "reply exactly: Camera view unclear."
)


# ---------- camera: keep only the latest frame (avoids stale buffered frames) ----------
class LatestFrame:
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
def speak(text):
    # Replace with Piper / Kokoro. For a quick test on Linux: subprocess.run(["espeak-ng", text])
    print(f"[SPEAK] {text}")
    if SIMULATE_SPEECH_TIME:
        time.sleep(len(text.split()) / SPEAK_WORDS_PER_S)


class Speaker:
    """One waiting slot. A newer sentence replaces an older one of the same or lower
    priority; a more urgent one is never replaced by a less urgent one. Sentences
    that waited longer than their TTL are dropped instead of spoken late."""

    def __init__(self):
        self.cv = threading.Condition()
        self.pending = None       # (priority, text, created, ttl)
        self.speaking = False
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, text, priority=INFO, ttl=INFO_TTL_S):
        if is_repeat(text):
            return "repeat suppressed"
        with self.cv:
            if self.pending is not None and self.pending[0] > priority:
                return "outranked"
            replaced = self.pending is not None
            self.pending = (priority, text, time.time(), ttl)
            self.cv.notify()
        return "replaced waiting sentence" if replaced else "queued"

    def idle(self):
        with self.cv:
            return not self.speaking and self.pending is None

    def _run(self):
        while True:
            with self.cv:
                while self.pending is None:
                    self.cv.wait()
                priority, text, created, ttl = self.pending
                self.pending = None
                self.speaking = True
            if time.time() - created > ttl:
                print(f"[dropped stale] {text}")
            else:
                mark_spoken(text)
                speak(text)
            with self.cv:
                self.speaking = False


speaker = Speaker()


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


def describe(frame):
    payload = {
        "model": MODEL,
        "stream": True,
        "keep_alive": "30m",             # keep the model loaded in memory
        "think": False,                  # reasoning mode would eat the token budget
        "options": {"num_predict": MAX_TOKENS, "temperature": 0},
        "messages": [{
            "role": "user",
            "content": build_prompt(),
            "images": [to_b64_jpeg(downscale(frame))],
        }],
    }
    buf = ""
    t0 = time.time()
    with requests.post(OLLAMA_URL, json=payload, stream=True, timeout=60) as r:
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


# ---------- main loop ----------
def main():
    warm_up()
    # python vlm_loop.py clip.mp4  -> replay a recording; no argument -> live webcam
    # (created AFTER warm-up so a replayed clip doesn't start playing while the model loads)
    cam = VideoFrame(sys.argv[1]) if len(sys.argv) > 1 else LatestFrame(CAMERA_INDEX)
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


if __name__ == "__main__":
    main()
