"""
YOLOv8 Fire Detection wrapper.

Provides a clean interface for loading the trained YOLO model
and running inference on images, video files, or webcam.

Usage:
    python -m src.yolo_detector --source path/to/image_or_video.mp4
    python -m src.yolo_detector --source 0  (webcam)
"""
import os
import argparse
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_WEIGHTS = os.path.join(PROJECT_ROOT, "models", "yolo_best.pt")
DEFAULT_CONF = 0.45
DEFAULT_IOU = 0.5
DEVICE = "mps"


def load_model(weights_path=DEFAULT_WEIGHTS):
    """Load YOLO model from weights file."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"YOLO weights not found at {weights_path}. "
            f"Train first with: python yolo_train.py"
        )
    return YOLO(weights_path)


def detect(source, weights_path=DEFAULT_WEIGHTS, conf=DEFAULT_CONF,
           iou=DEFAULT_IOU, save=True, show=False):
    """Run fire detection on an image, video, or webcam feed."""
    model = load_model(weights_path)
    results = model.predict(
        source=source, conf=conf, iou=iou,
        save=save, show=show, device=DEVICE
    )
    return results


def detect_frame(frame, model=None, weights_path=DEFAULT_WEIGHTS,
                 conf=DEFAULT_CONF, iou=DEFAULT_IOU):
    """
    Run detection on a single OpenCV frame (numpy array).
    Returns dict with fire_detected, max_confidence, num_detections, boxes.
    """
    if model is None:
        model = load_model(weights_path)

    results = model.predict(
        source=frame, conf=conf, iou=iou,
        save=False, show=False, device=DEVICE, verbose=False
    )

    boxes = []
    if len(results) > 0 and results[0].boxes is not None:
        for box in results[0].boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            boxes.append([*xyxy, confidence])

    max_conf = max([b[4] for b in boxes], default=0.0)

    return {
        "fire_detected": len(boxes) > 0,
        "max_confidence": max_conf,
        "num_detections": len(boxes),
        "boxes": boxes
    }


def validate(weights_path=DEFAULT_WEIGHTS):
    """Run validation on the test set. Returns metrics dict."""
    model = load_model(weights_path)
    data_yaml = os.path.join(PROJECT_ROOT, "data", "fire_detection", "data.yaml")
    results = model.val(
        data=data_yaml, imgsz=640, conf=0.25, iou=0.5, device=DEVICE
    )
    return {
        "mAP50": float(results.box.map50),
        "mAP50_95": float(results.box.map),
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Fire Detector")
    parser.add_argument("--source", type=str, help="Image/video path or 0 for webcam")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--validate", action="store_true", help="Run validation")
    args = parser.parse_args()

    if args.validate:
        metrics = validate()
        print("YOLO Validation Results:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
    elif args.source:
        source = int(args.source) if args.source.isdigit() else args.source
        results = detect(source, conf=args.conf, show=True)
        for r in results:
            print(f"Detections: {len(r.boxes)} fire regions found")
    else:
        parser.print_help()
