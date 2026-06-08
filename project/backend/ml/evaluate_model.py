from __future__ import annotations

import argparse
import json
from pathlib import Path


def evaluate(dataset_dir: Path, model_path: Path) -> dict[str, float]:
    try:
        import torch
        import torch.nn as nn
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Instale torch, pillow e numpy para avaliar o modelo.") from exc

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

    checkpoint = torch.load(model_path, map_location="cpu")
    model = TinySolarCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    samples = []
    for label, target in (("low", 0.0), ("high", 1.0)):
        for path in sorted((dataset_dir / label).glob("*.jpg")):
            samples.append((path, target))
    if not samples:
        raise SystemExit(f"Nenhuma imagem encontrada em {dataset_dir}")

    correct = 0
    tp = tn = fp = fn = 0
    probabilities = []
    with torch.no_grad():
        for path, target in samples:
            image = Image.open(path).convert("L").resize((128, 128))
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
            probability = float(torch.sigmoid(model(tensor)).item())
            prediction = 1.0 if probability >= 0.5 else 0.0
            probabilities.append(probability)
            correct += int(prediction == target)
            if prediction == 1.0 and target == 1.0:
                tp += 1
            elif prediction == 0.0 and target == 0.0:
                tn += 1
            elif prediction == 1.0 and target == 0.0:
                fp += 1
            else:
                fn += 1

    total = len(samples)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics = {
        "accuracy": correct / total,
        "precision_high": precision,
        "recall_high": recall,
        "f1_high": 2 * precision * recall / max(precision + recall, 1e-9),
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "mean_probability": sum(probabilities) / total,
    }
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Avalia o modelo treinado no split de teste.")
    parser.add_argument("--dataset-dir", default="data/processed/test", type=Path)
    parser.add_argument("--model-path", default="ml/models/solar_flare_cnn.pt", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.dataset_dir, args.model_path), indent=2))

