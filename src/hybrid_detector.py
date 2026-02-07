"""
Hybrid Fire Detection System.

Combines YOLOv8 visual detection with IoT sensor ML predictions
to produce a unified fire risk assessment.

Fusion strategy:
    - Weighted combination: 40% visual + 60% sensor
    - Override: high-confidence single detector escalates
    - Agreement bonus: both detect fire -> CRITICAL
"""
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    SAFE = "SAFE"
    LOW = "LOW_RISK"
    MEDIUM = "MEDIUM_RISK"
    HIGH = "HIGH_RISK"
    CRITICAL = "CRITICAL"


@dataclass
class HybridResult:
    risk_level: RiskLevel
    combined_score: float
    visual_fire_detected: bool
    visual_confidence: float
    sensor_fire_detected: bool
    sensor_probability: float
    message: str


VISUAL_WEIGHT = 0.4
SENSOR_WEIGHT = 0.6

THRESHOLD_LOW = 0.3
THRESHOLD_MEDIUM = 0.5
THRESHOLD_HIGH = 0.7
THRESHOLD_CRITICAL = 0.85


def fuse_predictions(visual_result: dict, sensor_result: dict) -> HybridResult:
    """
    Combine visual (YOLO) and sensor (ML) predictions.

    Args:
        visual_result: dict from yolo_detector.detect_frame()
        sensor_result: dict from sensor_model.predict()

    Returns:
        HybridResult with combined assessment
    """
    v_conf = visual_result.get("max_confidence", 0.0)
    v_detected = visual_result.get("fire_detected", False)

    s_prob = sensor_result.get("probability", 0.0)
    s_detected = sensor_result.get("prediction", 0) == 1

    # Weighted combination
    combined = (VISUAL_WEIGHT * v_conf) + (SENSOR_WEIGHT * s_prob)

    # Override: high-confidence single detector escalates
    if v_detected and v_conf > 0.8:
        combined = max(combined, 0.75)
    if s_detected and s_prob > 0.9:
        combined = max(combined, 0.75)

    # Both detectors agree -> escalate
    if v_detected and s_detected:
        combined = max(combined, 0.85)

    # Determine risk level
    if combined >= THRESHOLD_CRITICAL:
        risk = RiskLevel.CRITICAL
    elif combined >= THRESHOLD_HIGH:
        risk = RiskLevel.HIGH
    elif combined >= THRESHOLD_MEDIUM:
        risk = RiskLevel.MEDIUM
    elif combined >= THRESHOLD_LOW:
        risk = RiskLevel.LOW
    else:
        risk = RiskLevel.SAFE

    # Build message
    parts = []
    if v_detected:
        parts.append(
            f"Visual: {visual_result['num_detections']} fire region(s) "
            f"detected (conf={v_conf:.2f})"
        )
    else:
        parts.append("Visual: No fire detected in frame")

    if s_detected:
        parts.append(f"Sensor: Fire indicators present (prob={s_prob:.2f})")
    else:
        parts.append(f"Sensor: Normal readings (prob={s_prob:.2f})")

    return HybridResult(
        risk_level=risk,
        combined_score=combined,
        visual_fire_detected=v_detected,
        visual_confidence=v_conf,
        sensor_fire_detected=s_detected,
        sensor_probability=s_prob,
        message=" | ".join(parts)
    )


def format_result(result: HybridResult) -> str:
    """Pretty-print a hybrid detection result."""
    return "\n".join([
        "=" * 60,
        "  HYBRID FIRE DETECTION RESULT",
        "=" * 60,
        f"  Risk Level:       {result.risk_level.value}",
        f"  Combined Score:   {result.combined_score:.3f}",
        f"  Visual Detection: {'YES' if result.visual_fire_detected else 'no'} "
        f"(confidence: {result.visual_confidence:.3f})",
        f"  Sensor Detection: {'YES' if result.sensor_fire_detected else 'no'} "
        f"(probability: {result.sensor_probability:.3f})",
        "-" * 60,
        f"  {result.message}",
        "=" * 60,
    ])
