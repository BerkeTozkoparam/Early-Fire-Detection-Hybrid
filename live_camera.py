"""
Live Camera Fire Detection with YOLOv8 + IoT Sensor Hybrid System.

Opens webcam feed, runs YOLOv8 fire detection on each frame,
and displays real-time results with hybrid risk assessment.

Controls:
    Q - Quit
    S - Toggle sensor simulation (normal/smoke/fire)
    SPACE - Screenshot

Run: python live_camera.py
"""
import os
import sys
import cv2
import time
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.yolo_detector import load_model, detect_frame
from src.sensor_model import predict as sensor_predict, MODEL_PATH
from src.hybrid_detector import fuse_predictions, RiskLevel

# --- Sensor simulation presets ---
SENSOR_SCENARIOS = {
    0: {
        "name": "Normal",
        "data": {
            "Temperature[C]": 22.0, "Humidity[%]": 55.0,
            "TVOC[ppb]": 50, "eCO2[ppm]": 420,
            "Raw H2": 12500, "Raw Ethanol": 19000,
            "Pressure[hPa]": 939.0, "PM1.0": 1.0, "PM2.5": 2.0,
            "NC0.5": 5.0, "NC1.0": 1.0, "NC2.5": 0.1,
        }
    },
    1: {
        "name": "Duman",
        "data": {
            "Temperature[C]": 32.0, "Humidity[%]": 35.0,
            "TVOC[ppb]": 5000, "eCO2[ppm]": 2000,
            "Raw H2": 11500, "Raw Ethanol": 17500,
            "Pressure[hPa]": 936.0, "PM1.0": 50.0, "PM2.5": 80.0,
            "NC0.5": 300.0, "NC1.0": 50.0, "NC2.5": 5.0,
        }
    },
    2: {
        "name": "Yangin",
        "data": {
            "Temperature[C]": 55.0, "Humidity[%]": 15.0,
            "TVOC[ppb]": 25000, "eCO2[ppm]": 10000,
            "Raw H2": 10800, "Raw Ethanol": 15500,
            "Pressure[hPa]": 933.0, "PM1.0": 300.0, "PM2.5": 500.0,
            "NC0.5": 5000.0, "NC1.0": 800.0, "NC2.5": 100.0,
        }
    },
}

RISK_COLORS = {
    RiskLevel.SAFE: (0, 200, 0),        # Green
    RiskLevel.LOW: (0, 200, 200),        # Yellow
    RiskLevel.MEDIUM: (0, 165, 255),     # Orange
    RiskLevel.HIGH: (0, 100, 255),       # Red-orange
    RiskLevel.CRITICAL: (0, 0, 255),     # Red
}

RISK_LABELS_TR = {
    RiskLevel.SAFE: "GUVENLI",
    RiskLevel.LOW: "DUSUK RISK",
    RiskLevel.MEDIUM: "ORTA RISK",
    RiskLevel.HIGH: "YUKSEK RISK",
    RiskLevel.CRITICAL: "YANGIN ALARMI!",
}


def draw_detection_boxes(frame, boxes):
    """Draw fire detection bounding boxes on frame."""
    for box in boxes:
        x1, y1, x2, y2, conf = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4]
        # Red box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
        # Label background
        label = f"FIRE {conf:.0%}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - h - 10), (x1 + w + 10, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return frame


def draw_hud(frame, hybrid_result, sensor_scenario_name, fps, frame_count):
    """Draw the heads-up display overlay."""
    h, w = frame.shape[:2]
    risk_color = RISK_COLORS.get(hybrid_result.risk_level, (255, 255, 255))
    risk_label = RISK_LABELS_TR.get(hybrid_result.risk_level, "UNKNOWN")

    # Top bar background
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 110), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Title
    cv2.putText(frame, "EARLY FIRE DETECTION - LIVE",
                (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Risk level indicator
    cv2.rectangle(frame, (w - 320, 10), (w - 10, 55), risk_color, -1)
    cv2.putText(frame, risk_label,
                (w - 310, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Metrics line
    metrics_y = 75
    cv2.putText(frame, f"Kamera: {hybrid_result.visual_confidence:.0%}",
                (15, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 2)
    cv2.putText(frame, f"Sensor: {hybrid_result.sensor_probability:.0%}",
                (220, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (100, 200, 255), 2)
    cv2.putText(frame, f"Combined: {hybrid_result.combined_score:.0%}",
                (420, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, risk_color, 2)
    cv2.putText(frame, f"FPS: {fps:.0f}",
                (w - 120, metrics_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 2)

    # Sensor scenario
    cv2.putText(frame, f"Sensor Sim: {sensor_scenario_name}  [S: degistir]",
                (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # Bottom bar - controls
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, h - 35), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
    cv2.putText(frame, "Q: Cikis  |  S: Sensor Degistir  |  SPACE: Screenshot",
                (15, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

    # Fire warning flash
    if hybrid_result.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
        if frame_count % 20 < 10:  # Blink effect
            cv2.rectangle(frame, (0, 0), (w, 5), (0, 0, 255), -1)
            cv2.rectangle(frame, (0, h - 5), (w, h), (0, 0, 255), -1)
            cv2.rectangle(frame, (0, 0), (5, h), (0, 0, 255), -1)
            cv2.rectangle(frame, (w - 5, 0), (w, h), (0, 0, 255), -1)

    return frame


def main():
    print("=" * 50)
    print("  LIVE FIRE DETECTION")
    print("=" * 50)

    # Load YOLO model
    print("YOLOv8 modeli yukleniyor...")
    try:
        yolo_model = load_model()
        print("YOLO modeli yuklendi!")
    except FileNotFoundError:
        print("UYARI: Egitilmis YOLO modeli bulunamadi.")
        print("Varsayilan yolov8n.pt kullanilacak (genel nesne tespiti).")
        from ultralytics import YOLO
        yolo_model = YOLO(os.path.join(PROJECT_ROOT, "models", "yolov8n.pt"))

    # Check sensor model
    if not os.path.exists(MODEL_PATH):
        print("Sensor modeli bulunamadi. Once 'python main.py --train-sensor' calistirin.")
        return

    print("Sensor modeli yuklendi!")
    print("\nKamera aciliyor...")

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("HATA: Kamera acilamadi!")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    sensor_idx = 0  # Current sensor scenario
    frame_count = 0
    prev_time = time.time()
    fps = 0

    print("Kamera aktif! Kontroller:")
    print("  Q     - Cikis")
    print("  S     - Sensor senaryosunu degistir")
    print("  SPACE - Ekran goruntusu al")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame okunamadi!")
            break

        frame_count += 1

        # Calculate FPS
        curr_time = time.time()
        if curr_time - prev_time >= 1.0:
            fps = frame_count / (curr_time - prev_time)
            frame_count = 0
            prev_time = curr_time

        # --- YOLOv8 Detection ---
        visual_result = detect_frame(frame, model=yolo_model, conf=0.35)

        # --- Sensor Prediction ---
        scenario = SENSOR_SCENARIOS[sensor_idx]
        sensor_result = sensor_predict(scenario["data"])

        # --- Hybrid Fusion ---
        hybrid_result = fuse_predictions(visual_result, sensor_result)

        # --- Draw on frame ---
        frame = draw_detection_boxes(frame, visual_result["boxes"])
        frame = draw_hud(frame, hybrid_result, scenario["name"], fps, frame_count)

        # Show frame
        cv2.imshow("Early Fire Detection - Live", frame)

        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
        elif key == ord('s') or key == ord('S'):
            sensor_idx = (sensor_idx + 1) % len(SENSOR_SCENARIOS)
            new_name = SENSOR_SCENARIOS[sensor_idx]["name"]
            print(f"Sensor senaryosu: {new_name}")
        elif key == ord(' '):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            ss_path = os.path.join(PROJECT_ROOT, f"screenshot_{ts}.png")
            cv2.imwrite(ss_path, frame)
            print(f"Screenshot kaydedildi: {ss_path}")

    cap.release()
    cv2.destroyAllWindows()
    print("Kamera kapatildi.")


if __name__ == "__main__":
    main()
