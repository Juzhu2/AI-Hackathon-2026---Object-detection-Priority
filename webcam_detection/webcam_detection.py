"""Show live YOLO26 and/or YOLO-World detection with BoT-SORT object IDs.

Default mode "both": YOLO26 detects its trained (COCO) classes and YOLO-World adds
the extra objects from labels.txt that COCO does not have (e.g. staircase).
"""

import argparse
import sys
import json
import time
from contextlib import ExitStack
from pathlib import Path
from tracking_utils import tracking_config, frame_record
from display_utils import compose_display


PROJECT_ROOT = Path(__file__).resolve().parent
WEIGHTS_DIR = PROJECT_ROOT / "weights"
DEFAULT_LABELS = PROJECT_ROOT / "labels.txt"
# Default model type; override per run with --model-type (yolo26, world, or both).
MODEL_TYPE = "both"
# YOLO-World track IDs are shifted by this much so they never collide with YOLO26 IDs.
WORLD_ID_OFFSET = 100000
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


def drop_overlaps(world_result, primary_result, iou_thresh=0.6):
    """Remove YOLO-World boxes that mostly duplicate a YOLO26 box (e.g. "mouse" vs "computer mouse")."""
    wb, pb = world_result.boxes, primary_result.boxes
    if wb is None or pb is None or len(wb) == 0 or len(pb) == 0:
        return world_result
    from torchvision.ops import box_iou

    keep = box_iou(wb.xyxy, pb.xyxy).max(dim=1).values < iou_thresh
    return world_result[keep]


def merge_records(records, names):
    """Combine per-detector frame records into one. World IDs are offset to stay unique."""
    merged = records[0]
    for item in merged["objects"]:
        item["source"] = names[0]
    for record, name in zip(records[1:], names[1:]):
        for item in record["objects"]:
            item["source"] = name
            if item["track_id"] is not None:
                item["track_id"] += WORLD_ID_OFFSET
                base = item["object_id"].rpartition("_")[0]
                item["object_id"] = f"{base}_{item['track_id']}"
            merged["objects"].append(item)
    merged["tracked_count"] = sum(item["track_id"] is not None for item in merged["objects"])
    return merged


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
    parser.add_argument("--device", default=None,
                        help="Torch device, e.g. 0 or cpu (default: GPU 0 if CUDA is available)")
    parser.add_argument("--no-half", action="store_true", help="Disable FP16 inference on GPU")
    parser.add_argument("--depth", action="store_true",
                        help="Run Depth Anything V2 and add per-object distance")
    parser.add_argument("--depth-mode", choices=["indoor", "outdoor", "relative"], default="indoor",
                        help="indoor/outdoor = metric (meters); relative = 0-1 proximity (default: indoor)")
    parser.add_argument("--depth-size", choices=["small", "base", "large"], default="small",
                        help="Depth model size (default: small)")
    parser.add_argument("--depth-input-size", type=int, default=518,
                        help="Depth model input size, longer side (default: 518)")
    parser.add_argument("--depth-every", type=int, default=1,
                        help="Run depth every N frames and reuse the last map in between (default: 1)")
    parser.add_argument("--show-depth", action="store_true", help="Open a second window with the depth map")
    parser.add_argument("--fuse", action="store_true",
                        help="Add 3D position, bearing and smoothing per object (needs --depth with a metric mode)")
    parser.add_argument("--clean-json", type=Path,
                        help="Write a clean, formatted JSON file of the fused output (needs --fuse)")
    parser.add_argument("--table", action="store_true",
                        help="Print a readable geometry table every frame (needs --fuse)")
    parser.add_argument("--hfov", type=float, default=70.0,
                        help="Camera horizontal field of view in degrees (default: 70)")
    parser.add_argument("--speak", action="store_true",
                        help="Run the prioritizer and print what would be spoken (needs --fuse)")
    parser.add_argument("--tts", action="store_true",
                        help="With --speak: also read utterances aloud (pip install pyttsx3)")
    parser.add_argument("--width", type=int, default=1280, help="Requested webcam capture width")
    parser.add_argument("--height", type=int, default=720, help="Requested webcam capture height")
    parser.add_argument("--model-type", choices=["yolo26", "world", "both"], default=MODEL_TYPE,
                        help="both = YOLO26 for COCO classes + YOLO-World for extra labels (default)")
    parser.add_argument("--world-model", default=None,
                        help="YOLO-World weights, used in --model-type both (default: yolov8l-worldv2.pt)")
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
    if args.fuse and (not args.depth or args.depth_mode == "relative"):
        parser.error("--fuse requires --depth with --depth-mode indoor or outdoor")
    if args.clean_json and not args.fuse:
        parser.error("--clean-json requires --fuse (and --depth)")
    if args.table and not args.fuse:
        parser.error("--table requires --fuse")
    if args.speak and not args.fuse:
        parser.error("--speak requires --fuse (and --depth)")
    if args.tts and not args.speak:
        parser.error("--tts requires --speak")
    if args.speak and not (PROJECT_ROOT / "prioritizer.py").is_file():
        parser.error("--speak requires prioritizer.py, which was not included in this archive")
    if args.depth_every < 1:
        parser.error("--depth-every must be positive")
    if args.imgsz < 32:
        parser.error("--imgsz must be at least 32")
    # Pick the default weights for the chosen model type.
    # --model is the YOLO26 weights in yolo26/both mode, and the YOLO-World weights in world mode.
    if args.model is None:
        args.model = DEFAULT_MODELS["world" if args.model_type == "world" else "yolo26"]
    if args.world_model is None:
        args.world_model = DEFAULT_MODELS["world"]
    # Simple model names use our bundled weights; explicit paths remain supported.
    if Path(args.model).name == args.model:
        args.model = str(WEIGHTS_DIR / args.model)
    if Path(args.world_model).name == args.world_model:
        args.world_model = str(WEIGHTS_DIR / args.world_model)
    return args


def main():
    args = parse_args()
    try:
        import cv2
    except ImportError:
        print("Install dependencies first: python -m pip install -r requirements.txt", file=sys.stderr)
        return 1

    import torch

    cuda_ok = torch.cuda.is_available()
    device = args.device if args.device is not None else (0 if cuda_ok else "cpu")
    use_half = str(device) != "cpu" and not args.no_half
    # Ultralytics >= 8.4.80 uses quantize=16 instead of the deprecated half=True.
    precision = {"quantize": 16} if use_half else {}
    print(f"torch {torch.__version__} | CUDA available: {cuda_ok} | device: {device} | fp16: {use_half}")
    if not cuda_ok:
        print("WARNING: running on CPU. On the GX10 install the cu130 torch build "
              "(pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130).")

    camera = None
    resources = ExitStack()
    window = f"VisionAssist - {args.model_type} + {args.profile} (Q or Esc to quit)"
    try:
        # Ultralytics downloads named pretrained weights on first use.
        detectors = []  # (name, model) pairs; run in order each frame
        world_weights = args.model if args.model_type == "world" else args.world_model
        yolo = None
        if args.model_type in ("yolo26", "both"):
            from ultralytics.models.yolo.model import YOLO

            yolo = YOLO(args.model)
            detectors.append(("yolo26", yolo))
            print("YOLO26 ready with its trained classes (COCO 80 by default).")
        if args.model_type in ("world", "both"):
            from ultralytics.models.yolo.model import YOLOWorld

            labels = load_custom_labels(args.labels)
            if args.model_type == "both":
                # YOLO26 already covers its trained classes; YOLO-World only adds the rest.
                known = {str(name).casefold() for name in yolo.names.values()}
                skipped = [label for label in labels if label.casefold() in known]
                labels = [label for label in labels if label.casefold() not in known]
                if skipped:
                    print(f"Skipping labels YOLO26 already detects: {skipped}")
            elif args.include_coco:
                from ultralytics.utils import ROOT, YAML

                coco_names = YAML.load(ROOT / "cfg/datasets/coco8.yaml")["names"]
                baseline = list(coco_names.values()) if isinstance(coco_names, dict) else coco_names
                labels = merge_labels(baseline, labels)
            if labels:
                world = YOLOWorld(world_weights)
                # set_classes replaces the vocabulary with exactly these prompts.
                world.set_classes(labels)
                detectors.append(("world", world))
                print(f"YOLO-World ready with {len(labels)} classes from {args.labels.name}: {labels[:10]}")
            else:
                print(f"WARNING: {args.labels.name} has nothing beyond YOLO26's classes; "
                      "running YOLO26 only.")
        if not detectors:
            raise ValueError("No detector could be loaded.")
        label_caches = {name: {} for name, _ in detectors}
        depth_est = None
        if args.depth:
            from depth_utils import (CORRIDOR_COLS, CORRIDOR_ROWS, DepthEstimator, add_depth,
                                     colorize, corridor_min_depth)

            depth_est = DepthEstimator(args.depth_size, args.depth_mode, device,
                                       use_half, args.depth_input_size)
            print(f"Depth model: {depth_est.model_id} ({args.depth_mode})")
        fuser = None
        if args.fuse:
            from geometry_fuser import GeometryFuser, format_table

            fuser = GeometryFuser(args.hfov)
            print(f"Geometry fuser on (hfov={args.hfov} deg).")
        clean_writer = None
        prioritizer = speak_queue = None
        if args.speak:
            from prioritizer import Prioritizer, UtteranceQueue, make_pyttsx3_speaker

            prioritizer = Prioritizer()
            speak_queue = UtteranceQueue(make_pyttsx3_speaker() if args.tts else None).start()
            resources.callback(speak_queue.stop)
            print("Prioritizer on" + (" with text-to-speech." if args.tts else " (printing SPEAK lines)."))
        tracker_path, config = (None, {}) if args.detect_only else resources.enter_context(
            tracking_config(args.profile, args.track_buffer, args.confidence)
        )
        import ultralytics
        metadata = {"type": "session", "ultralytics_version": ultralytics.__version__,
                    "model": args.model, "model_type": args.model_type,
                    "detectors": [name for name, _ in detectors],
                    "world_model": world_weights if args.model_type != "yolo26" else None,
                    "profile": "detect-only" if args.detect_only else args.profile,
                    "tracking_config": config, "confidence": args.confidence,
                    "imgsz": args.imgsz,
                    "source": str(args.source) if args.source else f"camera:{args.camera}"}
        metadata["depth"] = (
            {"model": depth_est.model_id, "mode": args.depth_mode, "every": args.depth_every}
            if depth_est else None
        )
        metadata["geometry"] = {"hfov_deg": args.hfov} if fuser else None
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

        if args.clean_json:
            from geometry_fuser import CleanJsonWriter

            frame_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            meta = {
                "schema": "visionassist.fused/2",
                "units": "meters, degrees, seconds",
                "coordinates": "camera frame: x right+, y up+, z forward along the optical axis; "
                               "bearing right+",
                "camera": {"width": frame_w, "height": frame_h, "hfov_deg": args.hfov,
                           "intrinsics_calibrated": False,
                           "note": "hfov is an assumed value; pinhole model, square pixels"},
                "depth": {
                    "model": depth_est.model_id, "mode": args.depth_mode,
                    "input_size": args.depth_input_size, "run_every_n_frames": args.depth_every,
                    "calibration": {
                        "status": "uncalibrated", "validated_against": None,
                        "note": "Pretrained Depth Anything V2 metric model used as-is, with no "
                                "per-camera or per-scene calibration. Accuracy has not been "
                                "measured on this camera; check against a tape measure before "
                                "relying on absolute values."}},
                "detectors": [name for name, _ in detectors],
                "detector_confidence_floor": args.confidence,
                "track_id_note": "YOLO-World track ids start at 100000",
                "smoothing": fuser.describe(),
                "corridor": {
                    "field": "corridor_min_z_m",
                    "definition": "5th-percentile metric depth inside a strip straight ahead; "
                                  "no class needed. Misses obstacles below the strip.",
                    "columns_fraction": list(CORRIDOR_COLS), "rows_fraction": list(CORRIDOR_ROWS)},
                "clock": {
                    "t": "seconds since the script started (or video time with --source)",
                    "capture_unix_s": "time.time() just after the frame was read; wall clock, "
                                      "comparable across machines if their clocks are synced",
                    "capture_monotonic_s": "time.monotonic() at the same moment; comparable only "
                                           "between processes on this machine",
                    "latency_ms": "capture to fused record ready (before the prioritizer); excludes "
                                  "camera and driver buffering before read() returned"},
                "frames": "one entry per processed frame, including frames with no objects",
            }
            args.clean_json.parent.mkdir(parents=True, exist_ok=True)
            clean_writer = CleanJsonWriter(args.clean_json, meta)
            resources.callback(clean_writer.close)  # finishes the JSON on exit or Ctrl+C
            print(f"Writing clean JSON to {args.clean_json}")

        if not args.no_display:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        frame_index = 0
        started = previous = time.perf_counter()
        fps = 0.0
        runtime_logged = False
        depth_map = None
        last_said = ("", -1e9)  # (text, record time) for the on-screen caption
        while True:
            success, frame = camera.read()
            captured_unix, captured_mono = time.time(), time.monotonic()
            if not success:
                if args.source:
                    break
                raise RuntimeError("Could not read a frame from the camera.")

            # Keep tracker state between frames so same-class objects have separate IDs.
            inference_started = time.perf_counter()
            results = []
            for name, detector in detectors:
                if args.detect_only:
                    res = detector.predict(frame, conf=args.confidence, imgsz=args.imgsz,
                                           device=device, verbose=False, **precision)[0]
                else:
                    res = detector.track(frame, persist=True, tracker=tracker_path,
                                         conf=args.confidence, imgsz=args.imgsz,
                                         device=device, verbose=False, **precision)[0]
                results.append((name, res))
            if len(results) == 2:  # drop YOLO-World boxes that duplicate YOLO26 ones
                results[1] = (results[1][0], drop_overlaps(results[1][1], results[0][1]))
            finished = time.perf_counter()
            instant_fps = 1 / max(finished - previous, 1e-9)
            fps = instant_fps if frame_index == 0 else 0.9 * fps + 0.1 * instant_fps
            previous = finished
            frame_index += 1
            timestamp = camera.get(cv2.CAP_PROP_POS_MSEC) / 1000 if args.source else finished - started
            latency_ms = (finished - inference_started) * 1000
            record = merge_records(
                [frame_record(res, frame_index, timestamp, fps, latency_ms, label_caches[name])
                 for name, res in results],
                [name for name, _ in results],
            )
            record["capture_unix_s"] = round(captured_unix, 3)
            record["capture_monotonic_s"] = round(captured_mono, 3)
            if depth_est is not None:
                depth_started = time.perf_counter()
                if depth_map is None or frame_index % args.depth_every == 0:
                    depth_map = depth_est.predict(frame)
                add_depth(record, depth_map, frame.shape, depth_est.mode)
                record["corridor_min_z_m"] = corridor_min_depth(depth_map, depth_est.mode)
                if fuser is not None:
                    fuser.update(record, frame.shape)
                    record["latency_ms"] = (time.monotonic() - captured_mono) * 1000
                    if args.table:
                        print(format_table(record), end="\n\n")
                    if clean_writer is not None:
                        clean_writer.write(record)
                if prioritizer is not None:
                    spoken = prioritizer.update(record)
                    record["utterances"] = [u.text for u in spoken]
                    for utt in spoken:
                        speak_queue.put(utt)
                        last_said = (utt.text, record["timestamp_seconds"])
                record["depth_ms"] = (time.perf_counter() - depth_started) * 1000
            if not args.detect_only and not runtime_logged:
                actual = vars(detectors[0][1].predictor.trackers[0].args).copy()
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
            image = results[0][1].plot()
            for _, extra in results[1:]:
                image = extra.plot(img=image)
            if depth_est is not None:
                if args.show_depth:
                    cv2.imshow("Depth", colorize(depth_map, frame.shape, depth_est.mode))
            caption = last_said[0] if timestamp - last_said[1] < 4 else ""
            image = compose_display(cv2, image, record, metadata["profile"], config, caption)
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
