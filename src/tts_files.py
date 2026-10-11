"""Play sentences through a TTS that writes audio FILES, and never leave files behind.

Your TTS is plugged in as a shell command template with {text} and {out}, e.g.

    --tts-cmd 'python ../tts/say.py --text {text} --out {out}'
    --tts-cmd 'piper --model en_US-lessac-medium.onnx --output_file {out}'   (text on stdin: use {stdin})

or from Python:  FileTTS(synth=lambda text, out_path: my_tts(text, out_path))

File lifecycle (all files live in one private folder in RAM, /dev/shm/va_tts):
  1. synthesize  -> <uuid>.wav
  2. play it with a local player (paplay / aplay / ffplay), polling `cancel`
  3. ALWAYS delete in a `finally:` block - after playing, after an interruption, or on error
  4. a sentence prepared in advance but never played (an alert cut the narration) is
     deleted by discard() or by the age sweep
  5. the whole folder is emptied at start and at exit, so a crash cannot pile files up

For a UI that plays audio in a browser instead of on the GX10: use read_and_delete(), which
returns the bytes and removes the file at once; the server then sends bytes, not paths.
"""
import atexit
import math
import os
import struct
import wave
import shlex
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor

TTS_DIR = "/dev/shm/va_tts"
MAX_FILE_AGE_S = 120          # anything older than this in TTS_DIR is junk
BEEP_PATH = "/dev/shm/va_beep.wav"   # kept outside TTS_DIR so the sweep never deletes it


def make_beep(path=BEEP_PATH, freq=1000, seconds=0.15, rate=22050, volume=0.5):
    """Short tone played the instant an urgent alert arrives, while its speech is synthesized."""
    n = int(rate * seconds)
    fade = int(rate * 0.01)                       # 10 ms fade in/out: no click
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        frames = bytearray()
        for i in range(n):
            env = min(1.0, i / fade, (n - i) / fade)
            frames += struct.pack("<h", int(32767 * volume * env * math.sin(2 * math.pi * freq * i / rate)))
        w.writeframes(bytes(frames))
    return path


def find_player():
    for cmd in (["paplay"], ["aplay", "-q"], ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]):
        if shutil.which(cmd[0]):
            return cmd
    return None


class FileTTS:
    def __init__(self, cmd=None, synth=None, player=None, ext=".wav"):
        if not cmd and not synth:
            raise ValueError("give a --tts-cmd template or a synth function")
        self.cmd, self.synth_fn, self.ext = cmd, synth, ext
        self.player = player or find_player()
        os.makedirs(TTS_DIR, exist_ok=True)
        self.sweep(max_age=0)                       # leftovers from a previous crash
        atexit.register(self.sweep, 0)
        self.pool = ThreadPoolExecutor(max_workers=1)   # one synthesis at a time, in order
        self.prepared = {}                          # text -> Future[path]
        self.lock = threading.Lock()
        self.beep = make_beep()

    # ---------- synthesis ----------
    def _synth(self, text):
        out = os.path.join(TTS_DIR, f"{uuid.uuid4().hex}{self.ext}")
        t0 = time.time()
        if self.synth_fn:
            self.synth_fn(text, out)
        elif "{stdin}" in self.cmd:
            subprocess.run(shlex.split(self.cmd.replace("{stdin}", "").format(out=out)),
                           input=text.encode(), check=True, capture_output=True, timeout=30)
        else:
            args = [a.format(text=text, out=out) for a in shlex.split(self.cmd)]
            subprocess.run(args, check=True, capture_output=True, timeout=30)
        if not os.path.exists(out):
            raise RuntimeError(f"TTS did not write {out}")
        print(f"(tts {time.time() - t0:.2f}s)")
        return out

    def prepare(self, text):
        """Start synthesizing `text` in the background (call it for the NEXT sentence)."""
        with self.lock:
            if text not in self.prepared:
                self.prepared[text] = self.pool.submit(self._synth, text)

    def discard(self):
        """Throw away everything prepared but not played (e.g. an alert ended a narration)."""
        with self.lock:
            futures, self.prepared = list(self.prepared.values()), {}
        for f in futures:
            f.add_done_callback(lambda fut: fut.exception() is None and _remove(fut.result()))

    def _take(self, text):
        with self.lock:
            fut = self.prepared.pop(text, None)
        return fut.result() if fut is not None else self._synth(text)

    # ---------- playback ----------
    def speak(self, text, cancel=None, urgent=False):
        """Synthesize (or reuse a prepared file), play it, delete it. Interruptible.
        urgent=True plays a beep immediately, overlapping the synthesis time."""
        path = None
        try:
            if urgent and self.player:
                subprocess.Popen(self.player + [self.beep], stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL)
            path = self._take(text)
            if cancel is not None and cancel.is_set():
                return "cancelled"
            if self.player is None:
                print("(no audio player found: install pulseaudio-utils or alsa-utils)")
                return "no player"
            proc = subprocess.Popen(self.player + [path], stdout=subprocess.DEVNULL,
                                    stderr=subprocess.DEVNULL)
            while proc.poll() is None:
                if cancel is not None and cancel.is_set():
                    proc.terminate()
                    proc.wait(timeout=2)
                    return "cancelled"
                time.sleep(0.02)
            return "done"
        except Exception as e:  # a TTS failure must never kill the speaker thread
            print(f"(tts error: {e})")
            return "error"
        finally:
            _remove(path)
            self.sweep()

    def read_and_delete(self, text):
        """For a browser UI: return the audio bytes; the file is gone before this returns."""
        path = self._take(text)
        try:
            with open(path, "rb") as f:
                return f.read()
        finally:
            _remove(path)

    def sweep(self, max_age=MAX_FILE_AGE_S):
        now = time.time()
        busy = set()
        with self.lock if hasattr(self, "lock") else _nolock():
            for fut in getattr(self, "prepared", {}).values():
                if fut.done() and fut.exception() is None:
                    busy.add(fut.result())
        for name in os.listdir(TTS_DIR) if os.path.isdir(TTS_DIR) else []:
            p = os.path.join(TTS_DIR, name)
            try:
                if p not in busy and now - os.path.getmtime(p) >= max_age:
                    os.remove(p)
            except FileNotFoundError:
                pass


class _nolock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _remove(path):
    if path:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
