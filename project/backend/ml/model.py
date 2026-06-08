from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps, ImageStat

BASE_DIR = Path(__file__).resolve().parents[1]
CLASSIFIER_MODEL_PATH = BASE_DIR / "ml" / "models" / "solar_flare_classifier_131.bin"
CMX_MODEL_PATH = BASE_DIR / "ml" / "models" / "solar_flare_cmx_131.bin"
BINARY_MODEL_PATH = BASE_DIR / "ml" / "models" / "solar_flare_cnn.bin"
MODEL_MODE = os.getenv("HERMES_MODEL_MODE", "classifier").strip().lower()


def _resolve_model_path() -> Path:
    if MODEL_MODE in {"forecast", "cmx", "spaceweatherlive"}:
        candidates = (CMX_MODEL_PATH, CLASSIFIER_MODEL_PATH, BINARY_MODEL_PATH)
    elif MODEL_MODE in {"classifier", "acv", "classification"}:
        candidates = (CLASSIFIER_MODEL_PATH, CMX_MODEL_PATH, BINARY_MODEL_PATH)
    else:
        candidates = (CLASSIFIER_MODEL_PATH, CMX_MODEL_PATH, BINARY_MODEL_PATH)
    for path in candidates:
        if path.exists():
            return path
    return CLASSIFIER_MODEL_PATH


MODEL_PATH = _resolve_model_path()


def _build_tiny_binary_model(nn: Any) -> Any:
    class TinySolarCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 24, kernel_size=3, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, kernel_size=3, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 96, kernel_size=3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(96, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return TinySolarCNN()


def _build_cmx_model(nn: Any) -> Any:
    class CMXCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 24, 3, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 96, 3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(96, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 3),
            )

        def forward(self, x):
            return self.head(self.features(x))

    return CMXCNN()


def _build_forecast_cmx_model(nn: Any) -> Any:
    class ForecastCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 24, 3, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(48, 96, 3, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(96, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 192),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(192, 3),
            )

        def forward(self, x):
            return self.head(self.features(x))

    return ForecastCNN()


def _build_forecast_cmx_ensemble_model(torch: Any, nn: Any, member_states: list[dict[str, Any]]) -> Any:
    class ForecastEnsemble(nn.Module):
        def __init__(self, states: list[dict[str, Any]]):
            super().__init__()
            self.members = nn.ModuleList()
            for state in states:
                model = _build_forecast_cmx_model(nn)
                model.load_state_dict(state)
                self.members.append(model)

        def forward(self, x):
            probabilities = [torch.sigmoid(member(x)) for member in self.members]
            mean_probability = torch.stack(probabilities, dim=0).mean(dim=0)
            return torch.logit(mean_probability.clamp(1e-6, 1 - 1e-6))

    return ForecastEnsemble(member_states)


def _build_acv_tiny_model(nn: Any) -> Any:
    class SolarShieldTinyCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 4 * 4, 64),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(64, 3),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return SolarShieldTinyCNN()


def _build_acv_deep_model(nn: Any) -> Any:
    class SolarShieldDeepCNN(nn.Module):
        def __init__(self):
            super().__init__()

            def block(in_channels: int, out_channels: int, dropout: float):
                return nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Dropout2d(dropout),
                )

            self.features = nn.Sequential(
                block(1, 24, 0.05),
                block(24, 48, 0.10),
                block(48, 96, 0.15),
                block(96, 128, 0.20),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(128 * 4 * 4, 160),
                nn.ReLU(),
                nn.Dropout(0.35),
                nn.Linear(160, 3),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    return SolarShieldDeepCNN()


def load_model() -> Any | None:
    model_path = _resolve_model_path()
    print(f"MODEL PATH: {model_path}, EXISTS: {model_path.exists()}")
    if not model_path.exists():
        return None
    try:
        import torch
        import torch.nn as nn

        checkpoint = torch.load(model_path, map_location="cpu")
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            architecture = checkpoint.get("architecture")
            if architecture == "acv_tiny_cnn":
                model = _build_acv_tiny_model(nn)
            elif architecture == "acv_deep_cnn":
                model = _build_acv_deep_model(nn)
            elif architecture == "forecast_cmx_cnn":
                model = _build_forecast_cmx_model(nn)
            elif architecture == "forecast_cmx_ensemble":
                model = _build_forecast_cmx_ensemble_model(torch, nn, checkpoint.get("member_state_dicts", []))
            elif architecture == "cmx_cnn":
                model = _build_cmx_model(nn)
            elif architecture == "tiny_solar_cnn":
                model = _build_tiny_binary_model(nn)
            else:
                return None
            if architecture != "forecast_cmx_ensemble":
                model.load_state_dict(checkpoint["model_state_dict"])
            model.hermes_metadata = {
                "architecture": architecture,
                "classes": checkpoint.get("classes", []),
                "metrics": checkpoint.get("metrics"),
                "val_accuracy": checkpoint.get("val_accuracy"),
                "image_size": checkpoint.get("image_size"),
                "label_policy": checkpoint.get("label_policy"),
                "channel": checkpoint.get("channel"),
                "path": str(model_path),
                "model_mode": MODEL_MODE,
                "parameter_count": checkpoint.get("parameter_count"),
                "training_source": checkpoint.get("training_source"),
            }
        else:
            model = checkpoint
            model.hermes_metadata = {"architecture": "unknown", "path": str(model_path)}
        model.eval()
        return model
    except Exception as e:
        print(f"MODEL LOAD EXCEPTION: {e}")
        return None


def preprocess_image(image_bytes: bytes) -> Image.Image:
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image.resize((224, 224))


def _demo_heuristic(image: Image.Image) -> tuple[float, float, dict[str, float]]:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32) / 255.0
    stat = ImageStat.Stat(gray)
    brightness = stat.mean[0] / 255.0
    contrast = min(stat.stddev[0] / 80.0, 1.0)

    edges = gray.filter(ImageFilter.FIND_EDGES)
    edge_density = float(np.asarray(edges, dtype=np.float32).mean() / 255.0)
    hot_pixels = float((arr > 0.82).mean())

    score = 0.18 + (brightness * 0.28) + (contrast * 0.22) + (edge_density * 0.22) + (hot_pixels * 0.35)
    score = float(max(0.03, min(score, 0.98)))
    confidence = float(max(0.62, min(0.96, 0.68 + abs(score - 0.5) * 0.42 + contrast * 0.12)))
    features = {
        "brightness": round(brightness, 4),
        "contrast": round(contrast, 4),
        "edge_density": round(edge_density, 4),
        "hot_pixels": round(hot_pixels, 4),
    }
    return score, confidence, features


def _probabilities_from_score(score: float) -> dict[str, float]:
    c_prob = float(max(0.0, min(score, 1.0)))
    m_prob = float(max(0.0, min((c_prob - 0.42) / 1.8, 0.65)))
    x_prob = float(max(0.0, min((c_prob - 0.72) / 3.2, 0.18)))
    return {"C": round(c_prob, 3), "M": round(m_prob, 3), "X": round(x_prob, 3)}


def _risk_from_probabilities(probabilities: dict[str, float]) -> str:
    if probabilities["X"] >= 0.1 or probabilities["M"] >= 0.45 or probabilities["C"] >= 0.75:
        return "HIGH"
    if probabilities["M"] >= 0.2 or probabilities["C"] >= 0.4:
        return "MEDIUM"
    return "LOW"


def _flare_probabilities_from_classes(class_probabilities: dict[str, float]) -> dict[str, float]:
    low = class_probabilities.get("LOW", 0.0)
    medium = class_probabilities.get("MEDIUM", 0.0)
    high = class_probabilities.get("HIGH", 0.0)
    c_prob = medium + high
    m_prob = high
    x_prob = high * 0.25
    return {
        "C": round(float(max(0.0, min(c_prob, 1.0))), 3),
        "M": round(float(max(0.0, min(m_prob, 1.0))), 3),
        "X": round(float(max(0.0, min(x_prob, 1.0))), 3),
    }


def predict_flare_risk(image_bytes: bytes, model: Any | None = None, forecast_calibration: bool = False) -> dict[str, Any]:
    image = preprocess_image(image_bytes)
    probabilities: dict[str, float] | None = None
    class_probabilities: dict[str, float] | None = None

    if model is not None:
        try:
            import torch

            metadata = getattr(model, "hermes_metadata", {})
            architecture = metadata.get("architecture", "")
            is_classifier = architecture in {"acv_tiny_cnn", "acv_deep_cnn"}
            image_size = int(metadata.get("image_size") or 128)
            gray = ImageOps.autocontrast(image.convert("L")).resize((image_size, image_size))
            array = np.asarray(gray, dtype=np.float32) / 255.0
            if is_classifier:
                array = (array - 0.5) / 0.5
            tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                logits = model(tensor)
                if is_classifier:
                    output = torch.softmax(logits, dim=1).cpu().numpy().reshape(-1)
                else:
                    output = torch.sigmoid(logits).cpu().numpy().reshape(-1)
            if is_classifier:
                classes = metadata.get("classes") or ["LOW", "MEDIUM", "HIGH"]
                class_probabilities = {
                    str(classes[index]).upper(): round(float(output[index]), 3)
                    for index in range(min(output.size, len(classes)))
                }
                for label in ("LOW", "MEDIUM", "HIGH"):
                    class_probabilities.setdefault(label, 0.0)
                probabilities = _flare_probabilities_from_classes(class_probabilities)
                risk = max(class_probabilities, key=class_probabilities.get)
                score = float(
                    (class_probabilities["LOW"] * 0.15)
                    + (class_probabilities["MEDIUM"] * 0.55)
                    + (class_probabilities["HIGH"] * 0.92)
                )
                confidence = float(max(class_probabilities.values()))
                features = {
                    "model": architecture,
                    "metrics": metadata.get("metrics"),
                    "channel": metadata.get("channel"),
                    "classes": metadata.get("classes"),
                    "label_policy": metadata.get("label_policy"),
                    "training_source": metadata.get("training_source"),
                    "calibration": "class_probabilities_from_sdobenchmark_peak_flux",
                    "flare_probability_note": "C/M/X are derived from LOW/MEDIUM/HIGH classes for UI comparison.",
                }
                explanation = {
                    "LOW": "A CNN classificou a imagem como baixo risco visual, abaixo da faixa C-class no criterio do dataset.",
                    "MEDIUM": "A CNN identificou padroes compativeis com atividade C-class e recomenda acompanhamento operacional.",
                    "HIGH": "A CNN classificou a imagem na faixa M/X-class operacional, exigindo monitoramento e mitigacao prioritaria.",
                }[risk]
                return {
                    "risk_level": risk,
                    "score": round(score, 3),
                    "confidence": round(confidence, 3),
                    "class_probabilities": class_probabilities,
                    "flare_probabilities": probabilities,
                    "explanation_short": explanation,
                    "features": features,
                    "mode": "trained",
                }
            if output.size >= 3:
                probabilities = {
                    "C": round(float(output[0]), 3),
                    "M": round(float(output[1]), 3),
                    "X": round(float(output[2]), 3),
                }
                score = probabilities["C"]
                confidence = float(np.mean([max(value, 1.0 - value) for value in probabilities.values()]))
            else:
                score = float(output[0])
                confidence = float(max(score, 1.0 - score))
            features = {
                "model": metadata.get("architecture", "trained_cnn"),
                "metrics": metadata.get("metrics"),
                "channel": metadata.get("channel"),
                "training_source": metadata.get("training_source"),
            }
        except Exception:
            score, confidence, features = _demo_heuristic(image)
    else:
        score, confidence, features = _demo_heuristic(image)

    if probabilities is None:
        probabilities = _probabilities_from_score(score)
        features = {**features, "calibration": "none_raw_model_score"}
    else:
        training_source = getattr(model, "hermes_metadata", {}).get("training_source")
        uses_wayback_forecast = bool(training_source and "Wayback SpaceWeatherLive forecast" in training_source)
        if forecast_calibration:
            features = {
                **features,
                "calibration": "disabled_external_calibration",
                "raw_flare_probabilities": probabilities,
            }
        elif uses_wayback_forecast:
            features = {**features, "calibration": "none_wayback_forecast_model"}
        elif training_source:
            features = {**features, "calibration": "none_historical_donki_model"}
        else:
            features = {**features, "calibration": "none_model_outputs_cmx"}
        score = probabilities["C"]
        confidence = float(np.mean([max(value, 1.0 - value) for value in probabilities.values()]))

    risk = _risk_from_probabilities(probabilities)
    explanation = {
        "LOW": "Atividade visual solar estavel, sem sinais fortes de regioes ativas intensas.",
        "MEDIUM": "Ha indicios moderados de atividade magnetica e regioes brilhantes que exigem acompanhamento.",
        "HIGH": "A imagem sugere atividade intensa e maior probabilidade de evento solar relevante nas proximas 24 horas.",
    }[risk]

    return {
        "risk_level": risk,
        "score": round(score, 3),
        "confidence": round(confidence, 3),
        "class_probabilities": class_probabilities,
        "flare_probabilities": probabilities,
        "explanation_short": explanation,
        "features": features,
        "mode": "trained" if model is not None else "demo",
    }
