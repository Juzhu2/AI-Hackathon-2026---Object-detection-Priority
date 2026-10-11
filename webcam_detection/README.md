# Webcam detection

Standalone copy of the supplied ZIP. The window shows annotated video on top
and a separate panel with detection details below. Press Q or Escape to quit.

From this folder:

```powershell
python -m pip install -r requirements.txt
python webcam_detection.py
```

For optional depth estimation, install `requirements-depth.txt` and add `--depth`.
Model weights are downloaded on first use and are excluded from Git.
The archive did not include `prioritizer.py`, so `--speak` and `--tts` are unavailable.
No virtual environment, cache, or generated output was included in the archive.

Validation: detection and BoT-SORT buffer tracking both processed a three-frame
test video successfully using the existing `yolo26n.pt` weights on CPU. The text
panel was checked for empty detections, depth values, and long wrapped captions.
Live camera, dual-model detection, and depth inference were not verified.

To process a video without opening a window:

```powershell
python webcam_detection.py --source path/to/video.mp4 --no-display
```
