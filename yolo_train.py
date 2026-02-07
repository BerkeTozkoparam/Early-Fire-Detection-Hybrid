"""
YOLOv8 Fire Detection - Training Script.
Adapted from BerkeTozkoparam/Fire-Detection-with-YOLOv8-.

Usage: python yolo_train.py
"""
import os
from ultralytics import YOLO

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))


def main():
    model = YOLO(os.path.join(PROJECT_ROOT, "models", "yolov8n.pt"))
    data_yaml = os.path.join(PROJECT_ROOT, "data", "fire_detection", "data.yaml")

    model.train(
        data=data_yaml,
        epochs=50,
        imgsz=640,
        batch=8,
        device="mps",
        project=os.path.join(PROJECT_ROOT, "runs"),
        name="fire_detect"
    )


if __name__ == "__main__":
    main()
