"""
Early Fire Detection - Hybrid System
Combines YOLOv8 visual detection with IoT sensor ML.

Usage:
    python main.py --demo           # Full demonstration
    python main.py --train-sensor   # Train sensor model only
    python main.py --hybrid         # Hybrid detection demo
    python main.py --test-yolo      # Validate YOLO model only
"""
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.sensor_model import (
    load_and_preprocess, train_model, save_model, predict as sensor_predict
)
from src.hybrid_detector import fuse_predictions, format_result


def demo_sensor():
    """Train sensor model and run demo predictions."""
    print("\n" + "=" * 60)
    print("  PART 1: IoT Sensor Fire Detection")
    print("=" * 60)

    X, y, feature_names = load_and_preprocess()
    model, scaler = train_model(X, y, feature_names)
    save_model(model, scaler)

    print("\n--- Demo Predictions ---")

    fire_sample = {
        "Temperature[C]": 50.0, "Humidity[%]": 18.0,
        "TVOC[ppb]": 8000, "eCO2[ppm]": 5000,
        "Raw H2": 10500, "Raw Ethanol": 16000,
        "Pressure[hPa]": 934.0, "PM1.0": 100.0, "PM2.5": 200.0,
        "NC0.5": 500.0, "NC1.0": 80.0, "NC2.5": 10.0,
    }
    print(f"Fire scenario:   {sensor_predict(fire_sample)}")

    safe_sample = {
        "Temperature[C]": 22.0, "Humidity[%]": 55.0,
        "TVOC[ppb]": 50, "eCO2[ppm]": 420,
        "Raw H2": 12500, "Raw Ethanol": 19000,
        "Pressure[hPa]": 940.0, "PM1.0": 1.0, "PM2.5": 2.0,
        "NC0.5": 5.0, "NC1.0": 1.0, "NC2.5": 0.1,
    }
    print(f"Normal scenario: {sensor_predict(safe_sample)}")


def demo_hybrid():
    """Demonstrate hybrid fusion with simulated inputs."""
    print("\n" + "=" * 60)
    print("  HYBRID DETECTION FUSION DEMO")
    print("=" * 60)

    scenarios = [
        (
            "Both detect fire (CRITICAL)",
            {"fire_detected": True, "max_confidence": 0.92, "num_detections": 2, "boxes": []},
            {"prediction": 1, "probability": 0.97, "label": "FIRE DETECTED"},
        ),
        (
            "Only sensor detects (smoke but no visible flame)",
            {"fire_detected": False, "max_confidence": 0.0, "num_detections": 0, "boxes": []},
            {"prediction": 1, "probability": 0.88, "label": "FIRE DETECTED"},
        ),
        (
            "Only camera detects (outdoor fire, no nearby sensors)",
            {"fire_detected": True, "max_confidence": 0.85, "num_detections": 1, "boxes": []},
            {"prediction": 0, "probability": 0.12, "label": "No fire"},
        ),
        (
            "Neither detects fire (SAFE)",
            {"fire_detected": False, "max_confidence": 0.0, "num_detections": 0, "boxes": []},
            {"prediction": 0, "probability": 0.05, "label": "No fire"},
        ),
    ]

    for name, visual, sensor in scenarios:
        print(f"\n--- Scenario: {name} ---")
        result = fuse_predictions(visual, sensor)
        print(format_result(result))


def demo_yolo():
    """Run YOLO validation."""
    print("\n" + "=" * 60)
    print("  YOLOv8 Visual Fire Detection")
    print("=" * 60)
    from src.yolo_detector import validate
    metrics = validate()
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")


def run_full_demo():
    """Run sensor training + hybrid demo."""
    demo_sensor()
    demo_hybrid()
    print("\n[Note] YOLO validation requires Fire-Detection image dataset.")
    print("[Note] Run 'python main.py --test-yolo' separately if dataset is available.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Early Fire Detection - Hybrid System")
    parser.add_argument("--demo", action="store_true", help="Full demonstration")
    parser.add_argument("--train-sensor", action="store_true", help="Train sensor model")
    parser.add_argument("--test-yolo", action="store_true", help="Validate YOLO model")
    parser.add_argument("--hybrid", action="store_true", help="Hybrid fusion demo")
    args = parser.parse_args()

    if args.train_sensor:
        demo_sensor()
    elif args.test_yolo:
        demo_yolo()
    elif args.hybrid:
        demo_hybrid()
    elif args.demo:
        run_full_demo()
    else:
        parser.print_help()
