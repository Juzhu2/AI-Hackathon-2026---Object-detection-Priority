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
# Keep exactly one of these model settings uncommented.
#MODEL_TYPE, DEFAULT_MODEL = "yolo26", "yolo26n.pt"
MODEL_TYPE, DEFAULT_MODEL = "world", "yolov8s-worldv2.pt"


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
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Weights name or local path for the selected model type")
    parser.add_argument("--labels", type=Path, default=DEFAULT_LABELS, help="YOLO-World only: TXT file with one object name per line")
    parser.add_argument("--confidence", type=float, default=0.4, help="Minimum confidence (0–1)")
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
    # Preserve the comment switch while allowing either model to be selected per run.
    if args.model == DEFAULT_MODEL and args.model_type != MODEL_TYPE:
        args.model = "yolo26n.pt" if args.model_type == "yolo26" else "yolov8s-worldv2.pt"
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
        elif args.model_type == "world":
            from ultralytics.models.yolo.model import YOLOWorld
            from ultralytics.utils import ROOT, YAML

            custom_labels = load_custom_labels(args.labels)
            coco_names = YAML.load(ROOT / "cfg/datasets/coco8.yaml")["names"]
            baseline = list(coco_names.values()) if isinstance(coco_names, dict) else coco_names
            labels = merge_labels(baseline, custom_labels)
            model = YOLOWorld(args.model)
            # set_classes replaces the vocabulary, so include the COCO baseline too.
            model.set_classes(labels)
            print(f"Detecting {len(labels)} classes (COCO + custom prompts from {args.labels.name}).")
        else:
            raise ValueError(f"Unknown model type: {args.model_type}")
        tracker_path, config = (None, {}) if args.detect_only else resources.enter_context(
            tracking_config(args.profile, args.track_buffer)
        )
        import ultralytics
        metadata = {"type": "session", "ultralytics_version": ultralytics.__version__,
                    "model": args.model, "model_type": args.model_type,
                    "profile": "detect-only" if args.detect_only else args.profile,
                    "tracking_config": config, "confidence": args.confidence,
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
                result = model.predict(frame, conf=args.confidence, verbose=False)[0]
            else:
                result = model.track(frame, persist=True, tracker=tracker_path,
                                     conf=args.confidence, verbose=False)[0]
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
