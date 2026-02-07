"""
IoT Sensor-based Fire Detection Model.

Trains a Random Forest classifier on smoke_detection_iot.csv data.
Supports training, evaluation, saving/loading, and single-sample prediction.

Usage:
    python -m src.sensor_model --train
    python -m src.sensor_model --predict
"""
import os
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "smoke_detection_iot.csv")
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "sensor_model.joblib")
SCALER_PATH = os.path.join(PROJECT_ROOT, "models", "sensor_scaler.joblib")

DROP_COLUMNS = ["UTC", "CNT"]
TARGET = "Fire Alarm"
TEST_SIZE = 0.2
RANDOM_STATE = 42

EXPECTED_FEATURES = [
    "Temperature[C]", "Humidity[%]", "TVOC[ppb]", "eCO2[ppm]",
    "Raw H2", "Raw Ethanol", "Pressure[hPa]", "PM1.0", "PM2.5",
    "NC0.5", "NC1.0", "NC2.5"
]


def load_and_preprocess(filepath=DATA_PATH):
    """Load CSV, drop non-features, return X and y."""
    df = pd.read_csv(filepath)

    if df.columns[0] == "" or df.columns[0].startswith("Unnamed"):
        df = df.drop(columns=[df.columns[0]])

    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns])

    y = df[TARGET].values
    X = df.drop(columns=[TARGET])

    feature_names = list(X.columns)
    return X.values, y, feature_names


def train_model(X, y, feature_names=None):
    """Split data, scale features, train Random Forest, evaluate."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    print("Training Random Forest...")
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    print("\n" + "=" * 50)
    print("SENSOR MODEL EVALUATION RESULTS")
    print("=" * 50)
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, y_prob):.4f}")
    print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Fire", "Fire"], yticklabels=["No Fire", "Fire"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Sensor Model - Confusion Matrix")
    plt.tight_layout()
    cm_path = os.path.join(PROJECT_ROOT, "models", "confusion_matrix.png")
    fig.savefig(cm_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {cm_path}")

    # Feature importance
    if feature_names is not None:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        print("\nTop Feature Importances:")
        for i in range(min(10, len(feature_names))):
            print(f"  {feature_names[indices[i]]:<25s} {importances[indices[i]]:.4f}")

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(range(len(feature_names)),
                importances[indices[::-1]],
                color="steelblue")
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels([feature_names[i] for i in indices[::-1]])
        ax.set_xlabel("Importance")
        ax.set_title("Random Forest Feature Importances")
        plt.tight_layout()
        fi_path = os.path.join(PROJECT_ROOT, "models", "feature_importances.png")
        fig.savefig(fi_path, dpi=150)
        plt.close()
        print(f"Feature importance plot saved to {fi_path}")

    # Cross-validation
    print("\nRunning 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="f1")
    print(f"CV F1 scores: {cv_scores}")
    print(f"CV F1 mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    return model, scaler


def save_model(model, scaler):
    """Save trained model and scaler to disk."""
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print(f"Scaler saved to {SCALER_PATH}")


def load_model():
    """Load trained model and scaler from disk."""
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict(sensor_readings: dict) -> dict:
    """
    Predict fire probability from a dictionary of sensor readings.

    Args:
        sensor_readings: dict with keys matching feature names

    Returns:
        dict with "prediction" (0 or 1) and "probability" (float 0-1)
    """
    model, scaler = load_model()

    values = [sensor_readings.get(f, 0.0) for f in EXPECTED_FEATURES]
    X = np.array(values).reshape(1, -1)
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]
    probability = model.predict_proba(X_scaled)[0][1]

    return {
        "prediction": int(prediction),
        "probability": float(probability),
        "label": "FIRE DETECTED" if prediction == 1 else "No fire"
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IoT Sensor Fire Detection Model")
    parser.add_argument("--train", action="store_true", help="Train and save the model")
    parser.add_argument("--predict", action="store_true", help="Run a demo prediction")
    args = parser.parse_args()

    if args.train:
        X, y, feature_names = load_and_preprocess()
        model, scaler = train_model(X, y, feature_names)
        save_model(model, scaler)
    elif args.predict:
        sample = {
            "Temperature[C]": 45.0, "Humidity[%]": 20.0,
            "TVOC[ppb]": 5000, "eCO2[ppm]": 2000,
            "Raw H2": 11000, "Raw Ethanol": 17000,
            "Pressure[hPa]": 935.0, "PM1.0": 50.0, "PM2.5": 80.0,
            "NC0.5": 300.0, "NC1.0": 50.0, "NC2.5": 5.0,
        }
        result = predict(sample)
        print(f"Sensor prediction: {result}")
    else:
        parser.print_help()
