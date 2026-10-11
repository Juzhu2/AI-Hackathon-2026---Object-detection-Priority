"""Start everything with one command: detector + fuser, VLM, fast path and live speech.

    ./start.sh                                          # same as: python run_all.py
    python run_all.py --camera 1                        # other webcam
    python run_all.py --describe-model gemma4:26b       # richer descriptions
    python run_all.py --outdoor                         # outdoor depth model
    python run_all.py --no-window                       # over SSH (no screen)
    python run_all.py --no-tts                          # print instead of speaking

Process 1: webcam_detection.py owns the camera, runs YOLO + depth + fuser, writes
           /dev/shm/fused.jsonl (every frame) and /dev/shm/latest.jpg (every 0.5 s).
           Its console output goes to detector.log (tail -f detector.log to watch it).
Process 2: vlm_loop.py reads both files: fast-path hazards + VLM + speaker.
           It keeps this terminal, so the n / d / r + Enter mode keys still work.

If either process stops, the other is stopped too. Ctrl+C stops both.
"""
import argparse
import json
import os
import shlex
import shutil
import signal
import subprocess
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))


def preflight(args):
    """Catch the common setup problems before loading big models."""
    ok = True
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            have = {m["name"] for m in json.load(r).get("models", [])}
        for model in dict.fromkeys(filter(None, ["gemma3:4b", args.describe_model])):
            if model not in have and f"{model}:latest" not in have:
                print(f"[check] Ollama has no model {model}: run  ollama pull {model}")
                ok = False
    except Exception as e:
        print(f"[check] Ollama is not reachable ({e}): run  ollama serve  (or sudo systemctl start ollama)")
        ok = False
    if not args.no_tts:
        if not shutil.which(args.tts_cmd.split()[0]):
            print(f"[check] TTS program not found: {args.tts_cmd.split()[0]}  (sudo apt install -y espeak-ng)")
            ok = False
        if not any(shutil.which(p) for p in ("paplay", "aplay", "ffplay")):
            print("[check] no audio player: sudo apt install -y pulseaudio-utils")
            ok = False
    if not os.path.exists(f"/dev/video{args.camera}"):
        print(f"[check] /dev/video{args.camera} does not exist (try --camera 1)")
        ok = False
    if args.window and not os.environ.get("DISPLAY"):
        print("[check] no screen in this terminal (SSH?): showing no window. "
              "Run it from the GX10's own desktop to see the camera view.")
        args.window = False
    return ok


def wait_for_fuser(fused, frame, detector, timeout):
    """Wait until the fuser has written a real frame record and a shared frame."""
    t0 = time.time()
    last_note = 0
    while time.time() - t0 < timeout:
        if detector.poll() is not None:
            return False
        try:
            with open(fused, "rb") as f:
                ready = any(b'"objects"' in line for line in f)
        except FileNotFoundError:
            ready = False
        if ready and os.path.exists(frame):
            print(f"[run_all] fuser ready after {time.time() - t0:.0f} s")
            return True
        if time.time() - last_note > 10:
            print(f"[run_all] waiting for models to load ({time.time() - t0:.0f} s)...")
            last_note = time.time()
        time.sleep(0.5)
    return False


def stop(proc, name):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGINT)          # lets webcam_detection release the camera cleanly
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        print(f"[run_all] {name} did not stop; killing it")
        proc.kill()
        proc.wait()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fused", default="/dev/shm/fused.jsonl")
    ap.add_argument("--frame", default="/dev/shm/latest.jpg")
    ap.add_argument("--mode", choices=["navigation", "description"], default="navigation")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--outdoor", action="store_true", help="outdoor metric depth model")
    ap.add_argument("--describe-model", default=None, help="e.g. gemma4:26b")
    ap.add_argument("--tts-cmd", default="espeak-ng -s 165 -w {out} {text}",
                    help="file-writing TTS with {text} and {out}")
    ap.add_argument("--no-tts", action="store_true", help="print sentences instead of speaking")
    ap.add_argument("--no-window", dest="window", action="store_false",
                    help="do not show the camera window (needed over SSH)")
    ap.add_argument("--skip-checks", action="store_true")
    ap.add_argument("--detector-args", default="", help='extra args, e.g. "--depth-mode outdoor"')
    ap.add_argument("--vlm-args", default="", help="extra args for vlm_loop.py")
    ap.add_argument("--detector-script", default=os.path.join(HERE, "webcam_detection.py"))
    ap.add_argument("--vlm-script", default=os.path.join(HERE, "vlm_loop.py"))
    ap.add_argument("--log", default=os.path.join(HERE, "detector.log"))
    ap.add_argument("--startup-timeout", type=float, default=180, help="seconds to wait for models")
    args = ap.parse_args()

    if not preflight(args) and not args.skip_checks:
        print("[run_all] fix the items above (or add --skip-checks)")
        return 1

    for path in (args.fused, args.frame):     # never start on the previous run's data
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

    det_cmd = [sys.executable, args.detector_script, "--depth", "--fuse",
               "--fused-jsonl", args.fused, "--share-frame", args.frame,
               "--camera", str(args.camera),
               "--depth-mode", "outdoor" if args.outdoor else "indoor",
               *([] if args.window else ["--no-display"]),
               *shlex.split(args.detector_args)]
    vlm_cmd = [sys.executable, args.vlm_script, "--frame-file", args.frame, "--fused", args.fused,
               "--mode", args.mode,
               *(["--describe-model", args.describe_model] if args.describe_model else []),
               *([] if args.no_tts else ["--tts-cmd", args.tts_cmd]),
               *shlex.split(args.vlm_args)]

    detector = vlm = None
    log = open(args.log, "w")
    try:
        print("[run_all] starting fuser:", " ".join(det_cmd))
        print(f"[run_all] fuser output -> {args.log}")
        detector = subprocess.Popen(det_cmd, cwd=os.path.dirname(args.detector_script),
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if not wait_for_fuser(args.fused, args.frame, detector, args.startup_timeout):
            print(f"[run_all] fuser did not start (exit code {detector.poll()}). Last log lines:")
            if "could not connect to display" in open(args.log).read():
                print("[run_all] -> no screen: add --no-window, or run from the GX10 desktop")
            log.flush()
            with open(args.log) as f:
                print("".join(f.readlines()[-15:]))
            return 1

        print("[run_all] starting VLM loop:", " ".join(vlm_cmd))
        print("[run_all] keys (type in THIS terminal, then Enter): n navigation | d describe | "
              "r describe again | m more | Ctrl+C quit. Q in the camera window also quits.")
        vlm = subprocess.Popen(vlm_cmd, cwd=os.path.dirname(args.vlm_script))  # keeps the terminal
        while True:
            if detector.poll() is not None:
                print(f"[run_all] fuser exited ({detector.returncode}); see {args.log}. Stopping VLM.")
                return 1
            if vlm.poll() is not None:
                print(f"[run_all] VLM loop exited ({vlm.returncode}). Stopping fuser.")
                return vlm.returncode
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[run_all] Ctrl+C: stopping both")
        return 0
    finally:
        stop(vlm, "VLM loop")
        stop(detector, "fuser")
        log.close()


if __name__ == "__main__":
    sys.exit(main())
