# Early Fire Detection - Hybrid System

Hybrid early fire detection system combining **YOLOv8 visual detection** with **IoT sensor-based ML** for maximum accuracy and reliability.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-FF4B4B?logo=streamlit)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Object_Detection-00FFFF?logo=yolo)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikit-learn)

## Overview

This project fuses two complementary approaches to fire detection:

| Component | Method | Accuracy |
|-----------|--------|----------|
| **Visual Detection** | YOLOv8 object detection on camera frames | Real-time fire/smoke detection |
| **Sensor Detection** | Random Forest on IoT sensor data | 99.98% accuracy |
| **Hybrid Fusion** | Weighted combination (40% visual + 60% sensor) | Multi-level risk assessment |

### Risk Levels

The hybrid system outputs 5 risk levels:

- **SAFE** - No fire detected
- **LOW** - Minor anomaly, monitoring recommended
- **MEDIUM** - Potential threat detected
- **HIGH** - Fire likely, immediate action needed
- **CRITICAL** - Both systems confirm fire

## Features

- **Live Camera Detection** - Real-time fire detection via webcam (WebRTC)
- **Image Upload** - Analyze uploaded images for fire
- **IoT Sensor Analysis** - Predict fire risk from 15 sensor readings
- **Hybrid Fusion** - Combined visual + sensor risk assessment
- **EDA Dashboard** - Exploratory data analysis visualizations
- **Model Training** - Train sensor model directly from the web UI

## Project Structure

```
Early fire/
├── app.py                     # Streamlit web application
├── main.py                    # CLI entry point
├── live_camera.py             # Standalone OpenCV camera script
├── yolo_train.py              # YOLOv8 training script
├── yolo_test.py               # YOLOv8 validation script
├── requirements.txt           # Python dependencies
├── packages.txt               # System dependencies (Streamlit Cloud)
├── data/
│   └── smoke_detection_iot.csv  # IoT sensor dataset (62,630 samples)
├── models/                    # Trained model artifacts
│   ├── yolo_best.pt           # Trained YOLOv8 weights
│   └── sensor_model.joblib    # Trained Random Forest model
├── notebooks/
│   └── eda_sensor_data.py     # EDA visualizations
└── src/
    ├── __init__.py
    ├── sensor_model.py        # IoT sensor ML pipeline
    ├── yolo_detector.py       # YOLOv8 wrapper
    └── hybrid_detector.py     # Fusion logic
```

## Installation

```bash
# Clone the repository
git clone https://github.com/BerkeTozkoparam/Early-Fire-Detection-Hybrid.git
cd Early-Fire-Detection-Hybrid

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Web UI (Streamlit)

```bash
streamlit run app.py
```

Opens at `http://localhost:8501` with pages for live camera, sensor detection, hybrid detection, EDA, and model training.

### CLI

```bash
# Full demo
python main.py --demo

# Train sensor model only
python main.py --train-sensor

# Validate YOLO model
python main.py --test-yolo

# Hybrid fusion demo
python main.py --hybrid
```

### Standalone Camera

```bash
python live_camera.py
```

## Dataset

The IoT sensor dataset (`smoke_detection_iot.csv`) contains **62,630 samples** with 15 sensor features:

| Feature | Description |
|---------|-------------|
| Temperature[C] | Ambient temperature |
| Humidity[%] | Relative humidity |
| TVOC[ppb] | Total Volatile Organic Compounds |
| eCO2[ppm] | Equivalent CO2 concentration |
| Raw H2 | Raw hydrogen sensor reading |
| Raw Ethanol | Raw ethanol sensor reading |
| Pressure[hPa] | Atmospheric pressure |
| PM1.0, PM2.5 | Particulate matter |
| NC0.5 - NC2.5 | Number concentration of particles |
| Fire Alarm | Target variable (0/1) |

## Model Performance

### Sensor Model (Random Forest)
- **Accuracy**: 99.98%
- **Precision**: 99.98%
- **Recall**: 99.99%
- **F1 Score**: 99.98%
- **ROC AUC**: 1.00

### Hybrid Fusion
- Weighted scoring: 40% visual + 60% sensor
- Override rules for high-confidence single-detector alerts
- Agreement bonus when both detectors confirm fire

## Tech Stack

- **Computer Vision**: PyTorch, Ultralytics YOLOv8, OpenCV
- **Machine Learning**: scikit-learn, Random Forest
- **Web UI**: Streamlit, streamlit-webrtc
- **Data Analysis**: pandas, NumPy, matplotlib, seaborn

## License

MIT

## Author

**Berke Baran Tozkoparan**
