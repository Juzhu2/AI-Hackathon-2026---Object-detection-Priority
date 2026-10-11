"""Show live YOLO26 or YOLO-World detection with BoT-SORT object IDs."""

import argparse
import sys
import json
import time
from contextlib import ExitStack
from pathlib import Path
from tracking_utils import tracking_config, frame_record, draw_debug


PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_LABELS = PROJECT_ROOT / "labels.txt"
# Default model type; override per run with --model-type.
MODEL_TYPE = "yolo26"
# Larger models than n/s: much better on small objects like a computer mouse.
DEFAULT_MODELS = {"yolo26": "yolo26m.pt", "world": "yolov8l-worldv2.pt"}


def load_custom_labels(path):
    """Read one object name per line, ignoring blanks and duplicate names."""
    labels = []
    seen = set()
    for line in Path(path).read_text(encoding="utf-8-sig").splitlines():
        label = line.strip()
        if label and label.casefold() not in seen:
            labels.append(label)
            seen.add(label.casefold())
    if not labels:
        raise ValueError(f"No object labels found in {path}")
    return labels


def merge_labels(baseline, custom):
    """Preserve the baseline vocabulary when adding case-insensitive unique prompts."""
    labels = []
    seen = set()
    for label in [*baseline, *custom]:
        if label.casefold() not in seen:
            labels.append(label)
            seen.add(label.casefold())
    return labels


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", default=None, help="Weights name or local path (default depends on --model-type)")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="YOLO-World only: TXT file with one object name per line")
    parser.add_argument("--include-coco", action="store_true",
                        help="YOLO-World only: also add the 80 COCO names to the vocabulary "
                             "(off by default; a short vocabulary scores much higher)")
    # Open-vocabulary scores are low, so the default is much lower than before (was 0.4).
    parser.add_argument("--confidence", type=float, default=0.15, help="Minimum confidence (0-1)")
    parser.add_argument("--imgsz", type=int, default=960,
                        help="Inference image size. Higher helps small objects (default: 960)")
    parser.add_argument("--width", type=int, default=1280, help="Requested webcam capture width")
    parser.add_argument("--height", type=int, default=720, help="Requested webcam capture height")
    parser.add_argument("--model-type", choices=["yolo26", "world"], default=MODEL_TYPE)
    parser.add_argument("--source", type=Path, help="Video file for repeatable comparisons instead of webcam")
    parser.add_argument("--profile", choices=["default", "buffer", "reid"], default="reid")
    parser.add_argument("--track-buffer", type=int, help="Override lost-track buffer in processed frames")
    parser.add_argument("--detect-only", action="store_true", help="Run detector independently without tracker")
    parser.add_argument("--no-display", action="store_true", help="Process without a desktop window")
    parser.add_argument("--json-output", type=Path, help="Write session metadata and frame records as JSON Lines")
    parser.add_argument("--debug", action="store_true", help="Log per-frame classes, IDs, confidence, counts, and FPS")
    args = parser.parse_args()
    if not 0 <= args.confidence <= 1:
        parser.error("--confidence must be between 0 and 1")
    if args.track_buffer is not None and args.track_buffer < 1:
        parser.error("--track-buffer must be positive")
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    # Pick the default weights for the chosen model type.
    if args.model is None:
        args.model = DEFAULT_MODELS[args.model_type]
    # Simple model names use our bundled weights; explicit paths remain supported.
    if Path(args.model).name == args.model:
        args.model = str(WEIGHTS_DIR / args.model)
    return args


def main():
    args = parse_args()
    try:
        import cv2
    except ImportError:
        print("Install dependencies first: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1

    camera = None
    resources = ExitStack()
    window = f"VisionAssist - {args.model_type} + {args.profile} (Q or Esc to quit)"
    try:
        # Ultralytics downloads named pretrained weights on first use.
        if args.model_type == "yolo26":
            from ultralytics.models.yolo.model import YOLO

            model = YOLO(args.model)
            print("Using YOLO26 with the model's trained classes (COCO 80 by default).")
            print("Note: stairs is not a COCO class; use --model-type world for it.")
        elif args.model_type == "world":
            from ultralytics.models.yolo.model import YOLOWorld

            labels = load_custom_labels(args.labels)
            if args.include_coco:
                from ultralytics.utils import ROOT, YAML

                coco_names = YAML.load(ROOT / "cfg/datasets/coco8.yaml")["names"]
                baseline = list(coco_names.values()) if isinstance(coco_names, dict) else coco_names
                labels = merge_labels(baseline, labels)
            model = YOLOWorld(args.model)
            # set_classes replaces the vocabulary with exactly these prompts.
            model.set_classes(labels)
            print(f"Detecting {len(labels)} classes from {args.labels.name}: {labels[:10]}")
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        tracker_path, config = (None, {}) if args.detect_only else resources.enter_context(
            tracking_config(args.profile, args.track_buffer, args.confidence)
        )
        import ultralytics
        metadata = {"type": "session", "ultralytics_version": ultralytics.__version__,
                    "model": args.model, "model_type": args.model_type,
                    "profile": "detect-only" if args.detect_only else args.profile,
                    "tracking_config": config, "confidence": args.confidence,
                    "imgsz": args.imgsz,
                    "source": str(args.source) if args.source else f"camera:{args.camera}"}
        print(json.dumps(metadata))
        output = None
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            output = resources.enter_context(args.json_output.open("w", encoding="utf-8"))
            output.write(json.dumps(metadata) + "\n")
        camera = cv2.VideoCapture(str(args.source) if args.source else args.camera)
        if not camera.isOpened():
            raise RuntimeError(
                f"Cannot open camera {args.camera}. Check camera permissions, "
                "close other camera apps, or try --camera 1."
            )
        if not args.source:
            # Higher capture resolution keeps small objects large enough to detect.
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
            actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera resolution: {actual_w}x{actual_h}")

        if not args.no_display:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        frame_index, label_cache = 0, {}
        started = previous = time.perf_counter()
        fps = 0.0
        runtime_logged = False
        while True:
            success, frame = camera.read()
            if not success:
                if args.source:
                    break
                raise RuntimeError("Could not read a frame from the camera.")

            # Keep tracker state between frames so same-class objects have separate IDs.
            inference_started = time.perf_counter()
            if args.detect_only:
                result = model.predict(frame, conf=args.confidence, imgsz=args.imgsz,
                                       verbose=False)[0]
            else:
                result = model.track(frame, persist=True, tracker=tracker_path,
                                     conf=args.confidence, imgsz=args.imgsz,
                                     verbose=False)[0]
            finished = time.perf_counter()
            instant_fps = 1 / max(finished - previous, 1e-9)
            fps = instant_fps if frame_index == 0 else 0.9 * fps + 0.1 * instant_fps
            previous = finished
            frame_index += 1
            timestamp = camera.get(cv2.CAP_PROP_POS_MSEC) / 1000 if args.source else finished - started
            record = frame_record(result, frame_index, timestamp, fps,
                                  (finished - inference_started) * 1000, label_cache)
            if not args.detect_only and not runtime_logged:
                actual = vars(model.predictor.trackers[0].args).copy()
                actual.pop("device", None)
                runtime = {"type": "tracking_runtime", "tracking_config": actual}
                print(json.dumps(runtime))
                if output:
                    output.write(json.dumps(runtime) + "\n")
                runtime_logged = True
            if output:
                output.write(json.dumps(record) + "\n")
            if args.debug:
                print(json.dumps(record))
            if args.no_display:
                continue
            image = draw_debug(cv2, result.plot(), record, metadata["profile"], config)
            cv2.imshow(window, image)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
        return 0
    except KeyboardInterrupt:
        return 0
    except Exception as error:
        print(f"Unable to run webcam detection: {error}", file=sys.stderr)
        return 1
    finally:
        if camera is not None:
            camera.release()
        resources.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    sys.exit(main())