from __future__ import annotations

from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parents[1]
GAIE_MODEL_PATH = BASE_DIR / "ml" / "models" / "gaie_best_model.joblib"

FEATURE_NAMES = [
    "c_probability",
    "m_probability",
    "x_probability",
    "solar_wind_speed_kms",
    "proton_density_pcm3",
    "xray_flux",
    "kp_index",
    "satellite_altitude_km",
    "radiation_dose_rate",
    "payload_criticality",
    "comms_dependency",
    "battery_margin_pct",
]


def _coerce_features(features: dict[str, Any]) -> dict[str, float]:
    values: dict[str, float] = {}
    for name in FEATURE_NAMES:
        try:
            values[name] = float(features.get(name, 0.0))
        except (TypeError, ValueError):
            values[name] = 0.0
    return values


def _fallback_score(values: dict[str, float]) -> float:
    radiation_pressure = min(values["radiation_dose_rate"] / 180.0, 1.0)
    flare_pressure = (
        (values["c_probability"] / 100.0) * 0.28
        + (values["m_probability"] / 100.0) * 0.36
        + (values["x_probability"] / 100.0) * 0.48
    )
    geomagnetic_pressure = min(values["kp_index"] / 9.0, 1.0) * 0.22
    mission_exposure = (
        min(values["payload_criticality"] / 5.0, 1.0) * 0.16
        + min(values["comms_dependency"] / 5.0, 1.0) * 0.10
        + max(0.0, 1.0 - values["battery_margin_pct"] / 100.0) * 0.12
    )
    plasma_pressure = min(values["solar_wind_speed_kms"] / 850.0, 1.0) * 0.08
    plasma_pressure += min(values["proton_density_pcm3"] / 35.0, 1.0) * 0.06
    return max(0.0, min(flare_pressure + geomagnetic_pressure + mission_exposure + radiation_pressure * 0.22 + plasma_pressure, 1.0))


def _risk_from_score(score: float) -> str:
    if score >= 0.66:
        return "HIGH"
    if score >= 0.38:
        return "MEDIUM"
    return "LOW"


def predict_engineering_risk(features: dict[str, Any]) -> dict[str, Any]:
    values = _coerce_features(features)
    provider = "fallback_rules"
    probability = _fallback_score(values)

    if GAIE_MODEL_PATH.exists():
        try:
            import joblib
            import pandas as pd

            model = joblib.load(GAIE_MODEL_PATH)
            frame = pd.DataFrame([{name: values[name] for name in FEATURE_NAMES}])
            if hasattr(model, "predict_proba"):
                probabilities = model.predict_proba(frame)[0]
                classes = list(getattr(model, "classes_", ["LOW", "MEDIUM", "HIGH"]))
                class_scores = {str(label).upper(): float(probabilities[index]) for index, label in enumerate(classes)}
                probability = max(class_scores.get("HIGH", 0.0), class_scores.get("MEDIUM", 0.0) * 0.72)
            else:
                risk = str(model.predict(frame)[0]).upper()
                probability = {"LOW": 0.22, "MEDIUM": 0.52, "HIGH": 0.82}.get(risk, probability)
            provider = "gaie_best_model"
        except Exception:
            provider = "fallback_rules_model_unavailable"

    risk = _risk_from_score(probability)
    return {
        "risk_level": risk,
        "score": round(probability, 3),
        "provider": provider,
        "features_used": values,
        "explanation": (
            "Predicao tabular GAIE para risco operacional de engenharia espacial, combinando "
            "probabilidades de flare, telemetria ambiental e criticidade da missao."
        ),
    }

