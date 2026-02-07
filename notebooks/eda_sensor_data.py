"""
Exploratory Data Analysis for IoT Fire Detection Sensor Data.
Run: python notebooks/eda_sensor_data.py
Saves plots to notebooks/ directory.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(os.path.dirname(OUTPUT_DIR), "data", "smoke_detection_iot.csv")


def load_data():
    df = pd.read_csv(DATA_PATH)
    df = df.drop(columns=[df.columns[0], "UTC", "CNT"], errors="ignore")
    return df


def plot_class_distribution(df):
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Fire Alarm"].value_counts().sort_index()
    counts.plot(kind="bar", ax=ax, color=["steelblue", "tomato"])
    ax.set_title("Class Distribution: Fire Alarm")
    ax.set_xlabel("Fire Alarm")
    ax.set_ylabel("Count")
    ax.set_xticklabels(["No Fire (0)", "Fire (1)"], rotation=0)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
    plt.close()


def plot_correlation_heatmap(df):
    fig, ax = plt.subplots(figsize=(12, 10))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150)
    plt.close()


def plot_feature_distributions(df):
    features = [c for c in df.columns if c != "Fire Alarm"]
    fig, axes = plt.subplots(4, 3, figsize=(15, 16))
    axes = axes.flatten()
    for i, feat in enumerate(features):
        if i >= len(axes):
            break
        for label, color in [(0, "steelblue"), (1, "tomato")]:
            subset = df[df["Fire Alarm"] == label][feat]
            axes[i].hist(subset, bins=50, alpha=0.6, label=f"Alarm={label}", color=color)
        axes[i].set_title(feat, fontsize=10)
        axes[i].legend(fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Feature Distributions by Fire Alarm Class", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_distributions.png"), dpi=150)
    plt.close()


def plot_boxplots(df):
    key_features = ["Temperature[C]", "Humidity[%]", "TVOC[ppb]", "eCO2[ppm]", "PM2.5", "Raw H2"]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    for i, feat in enumerate(key_features):
        df.boxplot(column=feat, by="Fire Alarm", ax=axes[i])
        axes[i].set_title(feat)
    plt.suptitle("Key Features by Fire Alarm Class", fontsize=14)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "feature_boxplots.png"), dpi=150)
    plt.close()


def print_summary(df):
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Shape: {df.shape}")
    print(f"\nClass distribution:")
    print(df["Fire Alarm"].value_counts())
    print(f"\nFire rate: {df['Fire Alarm'].mean():.1%}")
    print(f"\nBasic statistics:")
    print(df.describe().round(2))
    print(f"\nMissing values:\n{df.isnull().sum()}")


if __name__ == "__main__":
    df = load_data()
    print_summary(df)
    plot_class_distribution(df)
    plot_correlation_heatmap(df)
    plot_feature_distributions(df)
    plot_boxplots(df)
    print(f"\nPlots saved to {OUTPUT_DIR}/")
