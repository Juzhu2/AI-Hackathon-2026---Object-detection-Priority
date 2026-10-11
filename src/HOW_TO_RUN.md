# How to run: offline vision assistant (GX10)

Live webcam → object detection + depth (fast path, hazards) + local VLM (scene sentences) → prioritizer → speech in headphones. Everything runs offline on the Asus Ascent GX10.

---

## TL;DR

On the GX10's own desktop (not over SSH), in a terminal:

```bash
cd ~/src
./start.sh
```

Wait until the camera window appears (first start takes about a minute), then type commands in that terminal, each followed by Enter:

| Key | Does |
|---|---|
| `n` | Navigation mode: short sentences while walking |
| `d` | Description mode: one long, detailed description of the view |
| `r` | Describe again (fresh image) |
| `m` | Tell me more about the same view, without repeating |
| `Ctrl+C` | Quit everything (or press `Q` in the camera window) |

Hazard warnings ("Stop. Person ahead, 1 meter.") interrupt whatever is being said.
- **Navigation mode:** all hazards; an alert ends any narration.
- **Description mode** (user standing still): only things **moving toward the user** interrupt. Static "Stop" lines (e.g. you standing in front of the camera) are held back, and the description **resumes** after an alert, repeating the sentence that was cut.

---

## 1. One-time setup

### Files (all in `~/src`)

| File | Role |
|---|---|
| `start.sh` | The one command: activates the venv and runs `run_all.py` |
| `run_all.py` | Runs startup checks, then starts the two processes below and stops both together |
| `webcam_detection.py` | Process 1: owns the camera; YOLO26 + YOLO-World + BoT-SORT + Depth Anything V2 + geometry fuser |
| `geometry_fuser.py`, `depth_utils.py`, `tracking_utils.py` | Fuser helpers |
| `labels.txt`, `botsort.yaml`, `weights/` | Extra YOLO-World labels, tracker config, model weights |
| `fused_stream.py` | Writes `/dev/shm/fused.jsonl` (one record per frame) and `/dev/shm/latest.jpg` |
| `vlm_loop.py` | Process 2: VLM via Ollama, modes, prioritizer / speaker |
| `fastpath.py` | Hazard rules on the fuser stream (stop / approaching / in path / object tags) |
| `tts_files.py` | Speech: file-writing TTS (espeak-ng), plays then deletes each file, beep before urgent alerts |
| `tts_check.py` | Stand-alone audio + TTS test |

Copy updated files from Windows PowerShell:

```powershell
scp run_all.py start.sh vlm_loop.py fastpath.py tts_files.py fused_stream.py webcam_detection.py asus@76RH:~/src/
```

Check on the GX10:

```bash
cd ~/src
ls labels.txt botsort.yaml weights/
grep -c "fused-jsonl" webcam_detection.py   # must be 2 or more (patched version)
chmod +x start.sh
```

### Packages

```bash
sudo apt install -y espeak-ng pulseaudio-utils alsa-utils
source ~/venv/bin/activate
pip install requests numpy    # plus the team's requirements.txt (ultralytics, opencv, lap, clip, torch, transformers)
```

### Ollama models

```bash
ollama list                  # gemma3:4b must be there
ollama pull gemma3:4b        # if missing
# optional, for richer descriptions:
ollama pull gemma4:26b
```

### Headphones

The GX10 shows only HDMI audio by default. Use one of:

- the monitor's 3.5 mm headphone jack (carries the HDMI audio, no setup)
- a USB headset or USB audio dongle
- Bluetooth headphones (`bluetoothctl` → `power on`, `scan on`, `pair`, `trust`, `connect`)

Wired is better for the demo: Bluetooth adds about 0.2–0.3 s to every warning.

```bash
pactl list short sinks                 # your headphones should appear here
pactl set-default-sink <name>          # make them the default
python tts_check.py                    # beep, speech, interruption, cleanup test
```

---

## 2. Running

```bash
./start.sh
```

What happens:

1. **Checks.** Ollama is reachable and has `gemma3:4b`, espeak-ng and an audio player are installed, the camera exists. If something is missing, it prints the fix and stops.
2. **Detector.** `webcam_detection.py` starts with the camera window. Its text output goes to `~/src/detector.log`.
3. **VLM and speech.** When the first fused frame arrives, `vlm_loop.py` starts in your terminal.

### Options

```bash
./start.sh --camera 1                    # the other webcam (/dev/video1)
./start.sh --describe-model gemma4:26b   # richer descriptions (slower, on demand only)
./start.sh --outdoor                     # outdoor metric depth model (streets, cars)
./start.sh --mode description            # start in description mode
./start.sh --no-tts                      # print sentences instead of speaking
./start.sh --no-window                   # no camera window (required over SSH)
./start.sh --skip-checks                 # start even if a check fails
```

Watch the detector while it runs (second terminal):

```bash
tail -f ~/src/detector.log
```

---

## 3. Demo checklist

| Do | Expect to hear |
|---|---|
| Stand still in view | Short navigation sentences from the VLM |
| Walk toward the camera | "Person ahead, … approaching", then beep + "Stop. Person ahead, 1 meter." |
| Type `d` + Enter | "Description mode.", an overview, then "On your left, …" and "On your right, …" in detail |
| Type `m` + Enter | Extra details, no repeats |
| Stand in front of the camera during a description | Nothing: static objects do not interrupt in description mode |
| Someone walks toward the camera during a description | "Person ahead, … approaching", then the description continues |
| Cover the lens | "Camera view blocked." |
| A traffic light / bus stop in view | Said once, quietly ("Traffic light slightly right, 8 meters.") |

---

## 4. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `could not connect to display` / Qt `xcb` crash | Running over SSH. Run from the GX10 desktop, or add `--no-window` |
| `fuser did not start` | Read the log lines `run_all` prints, or `detector.log` |
| `Ollama is not reachable` | `ollama serve` or `sudo systemctl start ollama` |
| No sound at all | `python tts_check.py`; if the beep fails it is the audio device: `pactl set-default-sink` |
| Sound comes from the monitor | Default sink is HDMI: plug headphones into the monitor jack or switch the sink |
| VLM talks but never "Stop" | Fast path gets no records: check `ls -la /dev/shm/fused.jsonl` is growing |
| "Stop" comes late | GPU shared with Ollama: compare detector fps in the log with and without the VLM |
| Wrong camera | `--camera 1` |
| Description mode says "Camera view unclear" | Look at `last_described.jpg` (exactly what the model saw) |
| Leftover audio files | Should never happen; `/dev/shm/va_tts` is emptied at start and exit |

---

## 5. Tuning knobs

All are constants at the top of the file named.

| File | Constant | Meaning |
|---|---|---|
| `fastpath.py` | `DEPTH_SCALE`, `DEPTH_OFFSET_M` | Depth correction `z_true = scale·z + offset`. Measured: about 1 m overestimate → offset −1.0. Refit with a tape measure at 1, 2, 3, 5 m |
| `fastpath.py` | `STOP_ZONE_M` (1.5), `WARN_ZONE_M` (3.0), `PATH_HALF_WIDTH_M` (0.7) | When "Stop" / path warnings fire |
| `fastpath.py` | `APPROACH_MPS` (0.3) | Closing speed that counts as approaching |
| `fastpath.py` | `TAGS`, `RULES` | Object tags: vehicle, ground, overhead, landmark. Add a class name to a tag set to change its rules |
| `fastpath.py` | `REPEAT_S`, `KIND_REPEAT_S` | How soon the same warning may repeat |
| `vlm_loop.py` | `MODEL`, `DESCRIBE_MODEL` | VLMs for navigation / description |
| `vlm_loop.py` | `PROMPT`, `OVERVIEW_PROMPT`, `SECTION_PROMPT`, `MORE_PROMPT` | What the VLM is asked |
| `vlm_loop.py` | `DESCRIBE_SECTIONS` (True) | Overview + zoomed left/right halves; False = one single description |
| `vlm_loop.py` | `COOLDOWN_S`, `SIMILARITY` | Repeat suppression for VLM sentences |
| `run_all.py` | `--tts-cmd` default | `espeak-ng -s 165 -w {out} {text}` (`-s` = speaking rate) |

---

## 6. Known limitations (say these honestly in the demo)

- Depth is **uncalibrated** apart from the −1 m correction; camera field of view (70°) is assumed.
- The corridor check looks at a band above the floor, so low obstacles rely on detection (the "ground" tag).
- Approach speed lags by a few frames (smoothing), and walking toward a static object also reads as "approaching".
- Traffic light **colour is never announced**. A wrong "green" is the worst possible error.
- The VLM can still occasionally invent details, especially in description mode.
- Description mode assumes the user is standing still: static obstacles are not announced there. Switch back with `n` before walking.

---

## How it fits together

```
webcam ─► webcam_detection.py ─► /dev/shm/fused.jsonl ─► fastpath.py ─┐
            (YOLO + depth +        /dev/shm/latest.jpg ─► vlm_loop.py ─┤
             fuser)                                       (Ollama VLM) │
                                                                       ▼
                                                    Speaker (prioritizer)
                                                    hazards interrupt, newest wins,
                                                    stale dropped, repeats suppressed
                                                                       │
                                                    tts_files.py ─► espeak-ng ─► headphones
```
