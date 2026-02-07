"""
YOLOv8 Fire Detection - Validation Script.
Usage: python yolo_test.py
"""
import os
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    weights = os.path.join(PROJECT_ROOT, "models", "yolo_best.pt")
    data_yaml = os.path.join(PROJECT_ROOT, "data", "fire_detection", "data.yaml")

    model = YOLO(weights)
    results = model.val(
        data=data_yaml, imgsz=640, conf=0.25, iou=0.5, device="mps"
    )

    print("=" * 40)
    print("YOLO VALIDATION RESULTS")
    print("=" * 40)
    print(f"mAP@0.5:      {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    print(f"Precision:     {results.box.mp:.4f}")
    print(f"Recall:        {results.box.mr:.4f}")


if __name__ == "__main__":
    main()
