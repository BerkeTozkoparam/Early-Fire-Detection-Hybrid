"""
Early Fire Detection - Hybrid System Web Interface
Streamlit app combining IoT sensor ML + YOLOv8 visual detection.

Run: streamlit run app.py
"""
import os
import sys
import cv2
import av
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.sensor_model import (
    load_and_preprocess, train_model, save_model, load_model,
    predict as sensor_predict, MODEL_PATH, EXPECTED_FEATURES
)
from src.yolo_detector import load_model as load_yolo, detect_frame
from src.hybrid_detector import fuse_predictions, RiskLevel

# --- Page Config ---
st.set_page_config(
    page_title="Early Fire Detection System",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
    }
    .risk-critical {
        background: linear-gradient(135deg, #ff0000, #cc0000);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff6600, #cc5200);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffaa00, #cc8800);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
    }
    .risk-low {
        background: linear-gradient(135deg, #ffdd00, #ccb100);
        color: black; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
    }
    .risk-safe {
        background: linear-gradient(135deg, #00cc44, #009933);
        color: white; padding: 20px; border-radius: 12px;
        text-align: center; font-size: 24px; font-weight: bold;
    }
    .metric-card {
        background: #1e1e2e; padding: 15px; border-radius: 10px;
        text-align: center; border: 1px solid #333;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_yolo_model():
    """Load and cache YOLOv8 model."""
    weights = os.path.join(PROJECT_ROOT, "models", "yolo_best.pt")
    if not os.path.exists(weights):
        weights = os.path.join(PROJECT_ROOT, "models", "yolov8n.pt")
    from ultralytics import YOLO
    return YOLO(weights)


# --- Sensor simulation presets ---
SENSOR_PRESETS = {
    "Normal Ortam": {
        "data": {
            "Temperature[C]": 22.0, "Humidity[%]": 55.0,
            "TVOC[ppb]": 50, "eCO2[ppm]": 420,
            "Raw H2": 12500, "Raw Ethanol": 19000,
            "Pressure[hPa]": 939.0, "PM1.0": 1.0, "PM2.5": 2.0,
            "NC0.5": 5.0, "NC1.0": 1.0, "NC2.5": 0.1,
        }
    },
    "Duman Algilandi": {
        "data": {
            "Temperature[C]": 32.0, "Humidity[%]": 35.0,
            "TVOC[ppb]": 5000, "eCO2[ppm]": 2000,
            "Raw H2": 11500, "Raw Ethanol": 17500,
            "Pressure[hPa]": 936.0, "PM1.0": 50.0, "PM2.5": 80.0,
            "NC0.5": 300.0, "NC1.0": 50.0, "NC2.5": 5.0,
        }
    },
    "Yangin Algilandi": {
        "data": {
            "Temperature[C]": 55.0, "Humidity[%]": 15.0,
            "TVOC[ppb]": 25000, "eCO2[ppm]": 10000,
            "Raw H2": 10800, "Raw Ethanol": 15500,
            "Pressure[hPa]": 933.0, "PM1.0": 300.0, "PM2.5": 500.0,
            "NC0.5": 5000.0, "NC1.0": 800.0, "NC2.5": 100.0,
        }
    },
}

RISK_COLORS_BGR = {
    RiskLevel.SAFE: (0, 200, 0),
    RiskLevel.LOW: (0, 200, 200),
    RiskLevel.MEDIUM: (0, 165, 255),
    RiskLevel.HIGH: (0, 100, 255),
    RiskLevel.CRITICAL: (0, 0, 255),
}

RISK_LABELS_TR = {
    RiskLevel.SAFE: "GUVENLI",
    RiskLevel.LOW: "DUSUK RISK",
    RiskLevel.MEDIUM: "ORTA RISK",
    RiskLevel.HIGH: "YUKSEK RISK",
    RiskLevel.CRITICAL: "YANGIN ALARMI!",
}


def get_risk_html(risk_level, score):
    """Return styled HTML for risk level display."""
    labels = {
        RiskLevel.CRITICAL: ("🚨 CRITICAL - YANGIN ALARMI!", "risk-critical"),
        RiskLevel.HIGH: ("⚠️ HIGH RISK - Yüksek Yangın Riski", "risk-high"),
        RiskLevel.MEDIUM: ("🔶 MEDIUM RISK - Orta Seviye Risk", "risk-medium"),
        RiskLevel.LOW: ("🔸 LOW RISK - Düşük Risk", "risk-low"),
        RiskLevel.SAFE: ("✅ SAFE - Güvenli", "risk-safe"),
    }
    label, css_class = labels.get(risk_level, ("Unknown", "risk-safe"))
    return f'<div class="{css_class}">{label}<br><span style="font-size:16px">Combined Score: {score:.3f}</span></div>'


def ensure_model_trained():
    """Train model if not already saved."""
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Model eğitiliyor... (ilk çalıştırmada bir kere olur)"):
            X, y, feature_names = load_and_preprocess()
            model, scaler = train_model(X, y, feature_names)
            save_model(model, scaler)
    return True


def page_sensor_detection():
    """IoT Sensor-based fire detection page."""
    st.header("📡 IoT Sensör Tabanlı Yangın Tespiti")
    st.markdown("IoT sensör değerlerini girerek yangın olasılığını hesaplayın.")

    ensure_model_trained()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("🌡️ Ortam")
        temp = st.slider("Sıcaklık (°C)", -25.0, 65.0, 22.0, 0.5)
        humidity = st.slider("Nem (%)", 10.0, 80.0, 50.0, 0.5)
        pressure = st.slider("Basınç (hPa)", 930.0, 945.0, 939.0, 0.1)

    with col2:
        st.subheader("💨 Gaz Sensörleri")
        tvoc = st.slider("TVOC (ppb)", 0, 60000, 100, 50)
        eco2 = st.slider("eCO2 (ppm)", 400, 60000, 420, 50)
        raw_h2 = st.slider("Raw H2", 10000, 14000, 12500, 10)
        raw_ethanol = st.slider("Raw Ethanol", 15000, 22000, 19000, 10)

    with col3:
        st.subheader("🌫️ Partikül Madde")
        pm1 = st.number_input("PM1.0", 0.0, 1000.0, 1.0, 0.5)
        pm25 = st.number_input("PM2.5", 0.0, 1000.0, 2.0, 0.5)
        nc05 = st.number_input("NC0.5", 0.0, 60000.0, 5.0, 1.0)
        nc1 = st.number_input("NC1.0", 0.0, 55000.0, 1.0, 1.0)
        nc25 = st.number_input("NC2.5", 0.0, 35000.0, 0.1, 0.1)

    st.markdown("---")

    if st.button("🔍 Yangın Analizi Yap", type="primary", use_container_width=True):
        sensor_data = {
            "Temperature[C]": temp, "Humidity[%]": humidity,
            "TVOC[ppb]": tvoc, "eCO2[ppm]": eco2,
            "Raw H2": raw_h2, "Raw Ethanol": raw_ethanol,
            "Pressure[hPa]": pressure, "PM1.0": pm1, "PM2.5": pm25,
            "NC0.5": nc05, "NC1.0": nc1, "NC2.5": nc25,
        }

        result = sensor_predict(sensor_data)

        st.markdown("---")

        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Tahmin", result["label"])
        with col_b:
            st.metric("Yangın Olasılığı", f"{result['probability']:.1%}")
        with col_c:
            st.metric("Sınıf", f"{'🔥 Fire' if result['prediction'] == 1 else '✅ Safe'}")

        if result["prediction"] == 1:
            st.error("🚨 YANGIN TESPİT EDİLDİ! Sensör verileri yangın göstergeleri içeriyor.")
        else:
            st.success("✅ Güvenli - Sensör verileri normal aralıkta.")

    # Preset scenarios
    st.markdown("---")
    st.subheader("📋 Hazır Senaryolar")
    preset_col1, preset_col2, preset_col3 = st.columns(3)

    with preset_col1:
        if st.button("🔥 Yangın Senaryosu", use_container_width=True):
            fire_data = {
                "Temperature[C]": 55.0, "Humidity[%]": 15.0,
                "TVOC[ppb]": 25000, "eCO2[ppm]": 10000,
                "Raw H2": 10800, "Raw Ethanol": 15500,
                "Pressure[hPa]": 933.0, "PM1.0": 300.0, "PM2.5": 500.0,
                "NC0.5": 5000.0, "NC1.0": 800.0, "NC2.5": 100.0,
            }
            r = sensor_predict(fire_data)
            st.error(f"🔥 Yangın! Olasılık: {r['probability']:.1%}")

    with preset_col2:
        if st.button("🌤️ Normal Ortam", use_container_width=True):
            normal_data = {
                "Temperature[C]": 22.0, "Humidity[%]": 55.0,
                "TVOC[ppb]": 50, "eCO2[ppm]": 420,
                "Raw H2": 12500, "Raw Ethanol": 19000,
                "Pressure[hPa]": 939.0, "PM1.0": 1.0, "PM2.5": 2.0,
                "NC0.5": 5.0, "NC1.0": 1.0, "NC2.5": 0.1,
            }
            r = sensor_predict(normal_data)
            st.success(f"✅ Güvenli. Olasılık: {r['probability']:.1%}")

    with preset_col3:
        if st.button("🌫️ Duman Senaryosu", use_container_width=True):
            smoke_data = {
                "Temperature[C]": 30.0, "Humidity[%]": 35.0,
                "TVOC[ppb]": 5000, "eCO2[ppm]": 2000,
                "Raw H2": 11500, "Raw Ethanol": 17500,
                "Pressure[hPa]": 936.0, "PM1.0": 50.0, "PM2.5": 80.0,
                "NC0.5": 300.0, "NC1.0": 50.0, "NC2.5": 5.0,
            }
            r = sensor_predict(smoke_data)
            if r["prediction"] == 1:
                st.warning(f"⚠️ Duman tespit edildi! Olasılık: {r['probability']:.1%}")
            else:
                st.info(f"🔶 Olasılık: {r['probability']:.1%}")


def page_hybrid_detection():
    """Hybrid detection fusion page."""
    st.header("🔗 Hibrit Yangın Algılama Sistemi")
    st.markdown(
        "Kamera (YOLOv8) ve IoT sensör verilerini birleştirerek "
        "çok katmanlı yangın risk değerlendirmesi yapın."
    )

    ensure_model_trained()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📹 Kamera (YOLOv8) Simülasyonu")
        visual_detected = st.checkbox("Kamera yangın algıladı mı?", value=False)
        visual_conf = st.slider("Kamera güven skoru", 0.0, 1.0, 0.0, 0.01)
        visual_count = st.number_input("Algılanan bölge sayısı", 0, 20, 0)

    with col2:
        st.subheader("📡 Sensör Verileri")
        sensor_mode = st.radio(
            "Sensör girişi",
            ["Manuel değer gir", "Hazır senaryo seç"]
        )

        if sensor_mode == "Hazır senaryo seç":
            scenario = st.selectbox("Senaryo", [
                "Normal ortam", "Duman algılandı", "Yangın algılandı"
            ])
            scenario_map = {
                "Normal ortam": {"prediction": 0, "probability": 0.05, "label": "No fire"},
                "Duman algılandı": {"prediction": 1, "probability": 0.75, "label": "FIRE DETECTED"},
                "Yangın algılandı": {"prediction": 1, "probability": 0.97, "label": "FIRE DETECTED"},
            }
            sensor_result = scenario_map[scenario]
            st.info(f"Sensör olasılığı: {sensor_result['probability']:.0%}")
        else:
            sensor_prob = st.slider("Sensör yangın olasılığı", 0.0, 1.0, 0.05, 0.01)
            sensor_pred = 1 if sensor_prob >= 0.5 else 0
            sensor_result = {
                "prediction": sensor_pred,
                "probability": sensor_prob,
                "label": "FIRE DETECTED" if sensor_pred == 1 else "No fire"
            }

    st.markdown("---")

    if st.button("⚡ Hibrit Analiz Çalıştır", type="primary", use_container_width=True):
        visual_result = {
            "fire_detected": visual_detected,
            "max_confidence": visual_conf if visual_detected else 0.0,
            "num_detections": visual_count if visual_detected else 0,
            "boxes": []
        }

        result = fuse_predictions(visual_result, sensor_result)

        st.markdown("---")
        st.markdown(get_risk_html(result.risk_level, result.combined_score), unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            st.metric("Combined Score", f"{result.combined_score:.3f}")
        with col_b:
            st.metric("Kamera", f"{'🔥 Evet' if result.visual_fire_detected else '❌ Hayır'}")
        with col_c:
            st.metric("Kamera Güven", f"{result.visual_confidence:.1%}")
        with col_d:
            st.metric("Sensör Olasılık", f"{result.sensor_probability:.1%}")

        # Fusion breakdown
        st.markdown("### 📊 Füzyon Detayları")
        fig, ax = plt.subplots(figsize=(8, 3))
        categories = ["Kamera (40%)", "Sensör (60%)", "Combined"]
        values = [
            result.visual_confidence * 0.4,
            result.sensor_probability * 0.6,
            result.combined_score
        ]
        colors = ["#3498db", "#e67e22", "#e74c3c" if result.combined_score > 0.5 else "#2ecc71"]
        bars = ax.barh(categories, values, color=colors, height=0.5)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Score")
        ax.axvline(x=0.5, color="red", linestyle="--", alpha=0.5, label="Threshold")
        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                    f"{val:.3f}", va="center", fontsize=11)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()


def page_eda():
    """EDA & Dataset visualization page."""
    st.header("📊 Veri Analizi (EDA)")

    data_path = os.path.join(PROJECT_ROOT, "data", "smoke_detection_iot.csv")
    df = pd.read_csv(data_path)
    df = df.drop(columns=[df.columns[0], "UTC", "CNT"], errors="ignore")

    # Summary stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Toplam Kayıt", f"{len(df):,}")
    with col2:
        st.metric("Özellik Sayısı", f"{len(df.columns) - 1}")
    with col3:
        fire_rate = df["Fire Alarm"].mean()
        st.metric("Yangın Oranı", f"{fire_rate:.1%}")
    with col4:
        st.metric("Eksik Veri", "0")

    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Sınıf Dağılımı", "🔥 Korelasyon", "📈 Dağılımlar", "📋 Veri Tablosu"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(6, 4))
        counts = df["Fire Alarm"].value_counts().sort_index()
        bars = ax.bar(["No Fire (0)", "Fire (1)"], counts.values,
                       color=["steelblue", "tomato"])
        for bar, val in zip(bars, counts.values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 200,
                    f"{val:,}", ha="center", fontsize=12, fontweight="bold")
        ax.set_title("Sınıf Dağılımı: Fire Alarm", fontsize=14)
        ax.set_ylabel("Kayıt Sayısı")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab2:
        fig, ax = plt.subplots(figsize=(12, 10))
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, ax=ax)
        ax.set_title("Özellik Korelasyon Matrisi", fontsize=14)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with tab3:
        features = [c for c in df.columns if c != "Fire Alarm"]
        selected = st.multiselect(
            "Görüntülenecek özellikler:",
            features,
            default=["Temperature[C]", "Humidity[%]", "TVOC[ppb]", "Pressure[hPa]"]
        )
        if selected:
            fig, axes = plt.subplots(1, len(selected), figsize=(5 * len(selected), 4))
            if len(selected) == 1:
                axes = [axes]
            for i, feat in enumerate(selected):
                for label, color in [(0, "steelblue"), (1, "tomato")]:
                    subset = df[df["Fire Alarm"] == label][feat]
                    axes[i].hist(subset, bins=50, alpha=0.6,
                                 label=f"Alarm={label}", color=color)
                axes[i].set_title(feat, fontsize=11)
                axes[i].legend(fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

    with tab4:
        st.dataframe(df.head(100), use_container_width=True)
        st.markdown(f"*İlk 100 kayıt gösteriliyor (toplam: {len(df):,})*")

    # Feature importances if model exists
    fi_path = os.path.join(PROJECT_ROOT, "models", "feature_importances.png")
    cm_path = os.path.join(PROJECT_ROOT, "models", "confusion_matrix.png")
    if os.path.exists(fi_path) or os.path.exists(cm_path):
        st.markdown("---")
        st.subheader("🤖 Model Sonuçları")
        mcol1, mcol2 = st.columns(2)
        if os.path.exists(fi_path):
            with mcol1:
                st.image(fi_path, caption="Özellik Önemleri (Random Forest)")
        if os.path.exists(cm_path):
            with mcol2:
                st.image(cm_path, caption="Karmaşıklık Matrisi")


def page_model_training():
    """Model training page."""
    st.header("🤖 Model Eğitimi")

    st.markdown("""
    Bu sayfada sensör modelini eğitebilir ve performans metriklerini görebilirsiniz.

    **Model:** Random Forest Classifier
    - 200 ağaç, max_depth=20
    - Stratified 80/20 split
    - 5-fold cross-validation
    """)

    if st.button("🚀 Modeli Eğit", type="primary", use_container_width=True):
        progress = st.progress(0, text="Veri yükleniyor...")

        progress.progress(10, text="Veri ön işleme...")
        X, y, feature_names = load_and_preprocess()

        progress.progress(30, text="Model eğitiliyor...")
        model, scaler = train_model(X, y, feature_names)

        progress.progress(80, text="Model kaydediliyor...")
        save_model(model, scaler)

        progress.progress(100, text="Tamamlandı!")
        st.success("Model başarıyla eğitildi ve kaydedildi!")

        # Show results
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_test_sc = sc.fit_transform(X_train)
        X_test_sc = sc.transform(X_test)
        y_pred = model.predict(X_test_sc)

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
        with col2:
            st.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
        with col3:
            st.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
        with col4:
            st.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")


def page_live_camera():
    """Live camera fire detection page."""
    st.header("📹 Canlı Kamera - Yangın Tespiti")
    st.markdown("Kameranızı kullanarak gerçek zamanlı YOLOv8 yangın tespiti yapın.")

    ensure_model_trained()

    # Sidebar sensor scenario selection
    sensor_scenario = st.selectbox(
        "Sensör Simülasyonu",
        list(SENSOR_PRESETS.keys()),
        index=0,
        help="Gerçek IoT sensör olmadığı için simüle edilen sensör verisi"
    )
    conf_threshold = st.slider("YOLO Güven Eşiği", 0.1, 0.95, 0.60, 0.05,
                               help="Düşük = daha fazla algılama ama yanlış alarm. Yüksek = kesin algılama.")
    process_every_n = st.slider("Her N. frame'de analiz", 1, 10, 3, 1,
                                help="Performans için frame atlama. 1=her frame, 5=her 5. frame")

    st.markdown("---")

    tab1, tab2 = st.tabs(["📹 Canlı Kamera (WebRTC)", "📷 Fotoğraf Çek / Yükle"])

    # ---- TAB 1: Live WebRTC Camera ----
    with tab1:
        st.markdown("Kameranızı açmak için **START** butonuna basın.")

        class FireDetector(VideoProcessorBase):
            def __init__(self):
                self.model = get_yolo_model()
                self.conf = conf_threshold
                self.sensor_name = sensor_scenario
                self.frame_count = 0
                self.skip = process_every_n
                self.last_boxes = []
                self.last_hybrid = None

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                img = frame.to_ndarray(format="bgr24")
                self.frame_count += 1

                # Only run YOLO every N frames for performance
                if self.frame_count % self.skip == 0 or self.last_hybrid is None:
                    # Resize for faster inference
                    h_orig, w_orig = img.shape[:2]
                    scale = min(640 / w_orig, 640 / h_orig, 1.0)
                    if scale < 1.0:
                        small = cv2.resize(img, (int(w_orig * scale), int(h_orig * scale)))
                    else:
                        small = img

                    results = self.model.predict(
                        source=small, conf=self.conf, iou=0.5,
                        save=False, show=False, verbose=False
                    )

                    boxes = []
                    if len(results) > 0 and results[0].boxes is not None:
                        for box in results[0].boxes:
                            xyxy = box.xyxy[0].cpu().numpy()
                            confidence = float(box.conf[0].cpu().numpy())
                            # Scale boxes back to original size
                            if scale < 1.0:
                                xyxy = xyxy / scale
                            boxes.append([*xyxy, confidence])

                    self.last_boxes = boxes

                    max_conf = max([b[4] for b in boxes], default=0.0)
                    visual_result = {
                        "fire_detected": len(boxes) > 0,
                        "max_confidence": max_conf,
                        "num_detections": len(boxes),
                        "boxes": boxes,
                    }

                    s_data = SENSOR_PRESETS[self.sensor_name]["data"]
                    sensor_result = sensor_predict(s_data)
                    self.last_hybrid = fuse_predictions(visual_result, sensor_result)

                boxes = self.last_boxes
                hybrid = self.last_hybrid
                risk_color = RISK_COLORS_BGR.get(hybrid.risk_level, (255, 255, 255))
                risk_label = RISK_LABELS_TR.get(hybrid.risk_level, "?")

                # Draw bounding boxes
                for b in boxes:
                    x1, y1, x2, y2, c = int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4]
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                    lbl = f"FIRE {c:.0%}"
                    (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 10, y1), (0, 0, 255), -1)
                    cv2.putText(img, lbl, (x1 + 5, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                h, w = img.shape[:2]

                # Top HUD bar
                overlay = img.copy()
                cv2.rectangle(overlay, (0, 0), (w, 90), (0, 0, 0), -1)
                cv2.addWeighted(overlay, 0.7, img, 0.3, 0, img)

                cv2.putText(img, "EARLY FIRE DETECTION - LIVE",
                            (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Risk badge
                cv2.rectangle(img, (w - 300, 8), (w - 10, 48), risk_color, -1)
                cv2.putText(img, risk_label,
                            (w - 290, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                # Metrics
                cv2.putText(img, f"Kamera: {hybrid.visual_confidence:.0%}",
                            (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 2)
                cv2.putText(img, f"Sensor: {hybrid.sensor_probability:.0%}",
                            (200, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 200, 255), 2)
                cv2.putText(img, f"Combined: {hybrid.combined_score:.0%}",
                            (390, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.5, risk_color, 2)
                cv2.putText(img, f"Sensor: {self.sensor_name}",
                            (15, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

                # Fire border flash
                if hybrid.risk_level in (RiskLevel.CRITICAL, RiskLevel.HIGH):
                    if self.frame_count % 20 < 10:
                        cv2.rectangle(img, (0, 0), (w, 4), (0, 0, 255), -1)
                        cv2.rectangle(img, (0, h - 4), (w, h), (0, 0, 255), -1)
                        cv2.rectangle(img, (0, 0), (4, h), (0, 0, 255), -1)
                        cv2.rectangle(img, (w - 4, 0), (w, h), (0, 0, 255), -1)

                return av.VideoFrame.from_ndarray(img, format="bgr24")

        webrtc_streamer(
            key="fire-detection-live",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=FireDetector,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.markdown("""
        > **Nasıl test edilir:** Telefonunuzdan bir yangın fotoğrafı/videosu açıp kameraya gösterin.
        > Sensör simülasyonunu yukarıdaki dropdown'dan değiştirebilirsiniz.
        """)

    # ---- TAB 2: Image Upload / Camera Capture ----
    with tab2:
        st.markdown("Bir görüntü yükleyin veya kameranızdan fotoğraf çekin.")

        input_col1, input_col2 = st.columns(2)

        with input_col1:
            uploaded = st.file_uploader(
                "Görüntü yükle", type=["jpg", "jpeg", "png", "webp"],
                help="Yangın içeren veya normal bir görüntü yükleyin"
            )

        with input_col2:
            camera_photo = st.camera_input("Kameradan çek")

        # Pick whichever is available
        img_source = camera_photo if camera_photo is not None else uploaded

        if img_source is not None:
            pil_img = Image.open(img_source).convert("RGB")
            img_array = np.array(pil_img)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

            with st.spinner("YOLOv8 analiz ediyor..."):
                model = get_yolo_model()
                visual_result = detect_frame(img_bgr, model=model, conf=conf_threshold)

                s_data = SENSOR_PRESETS[sensor_scenario]["data"]
                sensor_result = sensor_predict(s_data)
                hybrid = fuse_predictions(visual_result, sensor_result)

            # Draw boxes on image
            annotated = img_bgr.copy()
            for b in visual_result["boxes"]:
                x1, y1, x2, y2, c = int(b[0]), int(b[1]), int(b[2]), int(b[3]), b[4]
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 3)
                lbl = f"FIRE {c:.0%}"
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(annotated, (x1, y1 - th - 12), (x1 + tw + 12, y1), (0, 0, 255), -1)
                cv2.putText(annotated, lbl, (x1 + 6, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            # Display
            res_col1, res_col2 = st.columns([2, 1])

            with res_col1:
                st.image(annotated_rgb, caption="YOLOv8 Algılama Sonucu", use_container_width=True)

            with res_col2:
                st.markdown(get_risk_html(hybrid.risk_level, hybrid.combined_score),
                            unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.metric("Algılanan Yangın Bölgesi", visual_result["num_detections"])
                st.metric("Kamera Güven", f"{visual_result['max_confidence']:.1%}")
                st.metric("Sensör Olasılık", f"{sensor_result['probability']:.1%}")
                st.metric("Sensör Modu", sensor_scenario)

            if visual_result["fire_detected"]:
                st.error(
                    f"🚨 {visual_result['num_detections']} yangın bölgesi tespit edildi! "
                    f"(max güven: {visual_result['max_confidence']:.0%})"
                )
            else:
                st.success("✅ Görüntüde yangın tespit edilmedi.")


def page_about():
    """About page."""
    st.header("ℹ️ Proje Hakkında")

    st.markdown("""
    ## Early Fire Detection - Hybrid System

    Bu proje, erken yangın tespiti için iki farklı yaklaşımı birleştiren hibrit bir sistemdir:

    ### 🔍 Yaklaşım 1: Görüntü Tabanlı (YOLOv8)
    - **Model:** YOLOv8 (You Only Look Once v8)
    - **Veri:** Kamera/video görüntüleri
    - **Görev:** Gerçek zamanlı yangın bölgesi tespiti
    - **Performans:** mAP@0.5: ~0.86, Precision: ~0.87

    ### 📡 Yaklaşım 2: Sensör Tabanlı (Random Forest)
    - **Model:** Random Forest Classifier (200 ağaç)
    - **Veri:** 62,630 IoT sensör kaydı
    - **Özellikler:** Sıcaklık, Nem, TVOC, eCO2, H2, Ethanol, Basınç, PM
    - **Performans:** Accuracy: ~99.98%, F1: ~99.99%

    ### 🔗 Hibrit Füzyon
    - Ağırlıklı birleştirme: %40 görsel + %60 sensör
    - Risk seviyeleri: SAFE → LOW → MEDIUM → HIGH → CRITICAL
    - Çift onay: Her iki sistem de algılarsa → CRITICAL

    ---

    ### 🏗️ Teknolojiler
    - Python, Streamlit, scikit-learn
    - YOLOv8 (Ultralytics), OpenCV
    - Pandas, Matplotlib, Seaborn

    ---

    **Geliştirici:** Berke Baran Tozkoparan
    **GitHub:** [BerkeTozkoparam](https://github.com/BerkeTozkoparam)
    """)


# --- Main App ---
def main():
    with st.sidebar:
        st.title("🔥 Early Fire Detection")
        st.markdown("Hibrit Yangın Algılama Sistemi")
        st.markdown("---")

        page = st.radio(
            "Sayfa Seçin",
            [
                "📹 Canlı Kamera",
                "📡 Sensör Tespiti",
                "🔗 Hibrit Algılama",
                "📊 Veri Analizi",
                "🤖 Model Eğitimi",
                "ℹ️ Hakkında"
            ],
            index=0
        )

        st.markdown("---")
        st.markdown(
            "<div style='text-align:center; color:#888; font-size:12px;'>"
            "Berke Baran Tozkoparan<br>"
            "Early Fire Detection v1.0"
            "</div>",
            unsafe_allow_html=True
        )

    if page == "📹 Canlı Kamera":
        page_live_camera()
    elif page == "📡 Sensör Tespiti":
        page_sensor_detection()
    elif page == "🔗 Hibrit Algılama":
        page_hybrid_detection()
    elif page == "📊 Veri Analizi":
        page_eda()
    elif page == "🤖 Model Eğitimi":
        page_model_training()
    elif page == "ℹ️ Hakkında":
        page_about()


if __name__ == "__main__":
    main()
