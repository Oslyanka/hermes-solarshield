from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


THRESHOLDS = {"C": 1e-6, "M": 1e-5, "X": 1e-4}


def _sample_dir(split_dir: Path, sample_id: str) -> Path:
    active_region, folder = sample_id.split("_", 1)
    return split_dir / active_region / folder


def _collect_samples(split_dir: Path, channel: str, max_images: int | None, seed: int) -> list[tuple[Path, list[float]]]:
    metadata_path = split_dir / "meta_data.csv"
    rows = list(csv.DictReader(metadata_path.open(newline="", encoding="utf-8")))
    samples: list[tuple[Path, list[float]]] = []
    for row in rows:
        peak_flux = float(row["peak_flux"])
        targets = [1.0 if peak_flux >= threshold else 0.0 for threshold in THRESHOLDS.values()]
        folder = _sample_dir(split_dir, row["id"])
        if not folder.exists():
            continue
        pattern = "*.jpg" if channel.lower() == "all" else f"*__{channel}.jpg"
        for path in sorted(folder.glob(pattern)):
            samples.append((path, targets))
    random.Random(seed).shuffle(samples)
    return samples[:max_images] if max_images else samples


def train(raw_dir: Path, output: Path, channel: str, max_epochs: int, target_accuracy: float, max_images: int, batch_size: int) -> dict[str, float]:
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from PIL import Image
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise SystemExit("Instale torch, pillow e numpy para treinar o modelo C/M/X.") from exc

    dataset_root = raw_dir / "SDOBenchmark_example" if (raw_dir / "SDOBenchmark_example").exists() else raw_dir
    training_dir = dataset_root / "training"
    test_dir = dataset_root / "test"
    if not training_dir.exists():
        raise SystemExit(f"SDOBenchmark nao encontrado em {dataset_root}")

    all_train = _collect_samples(training_dir, channel, max_images, seed=42)
    split = int(len(all_train) * 0.82)
    train_samples = all_train[:split]
    val_samples = all_train[split:]
    test_samples = _collect_samples(test_dir, channel, max(max_images // 4, 1), seed=7) if test_dir.exists() else []

    class CMXDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            path, targets = self.samples[index]
            image = Image.open(path).convert("L").resize((128, 128))
            array = np.asarray(image, dtype=np.float32) / 255.0
            return torch.from_numpy(array).unsqueeze(0), torch.tensor(targets, dtype=torch.float32)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = CMXDataset(train_samples)
    val_set = CMXDataset(val_samples)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)
    model = CMXCNN().to(device)

    target_matrix = torch.tensor([targets for _, targets in train_samples], dtype=torch.float32)
    positives = target_matrix.sum(dim=0)
    negatives = len(train_samples) - positives
    pos_weight = (negatives / torch.clamp(positives, min=1)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    output.parent.mkdir(parents=True, exist_ok=True)
    best_accuracy = 0.0
    best_metrics: dict[str, float] = {}
    history = []

    print(f"Treinando C/M/X em {device}: train={len(train_samples)} val={len(val_samples)} test={len(test_samples)} channel={channel}")
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        correct = torch.zeros(3)
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                logits = model(images)
                val_loss += float(criterion(logits, targets).item())
                predictions = (torch.sigmoid(logits) >= 0.5).float().cpu()
                correct += (predictions == targets.cpu()).sum(dim=0)
                total += targets.shape[0]

        per_class = correct / max(total, 1)
        macro_accuracy = float(per_class.mean().item())
        metrics = {
            "macro_accuracy": macro_accuracy,
            "c_accuracy": float(per_class[0].item()),
            "m_accuracy": float(per_class[1].item()),
            "x_accuracy": float(per_class[2].item()),
            "train_loss": train_loss / max(len(train_loader), 1),
            "val_loss": val_loss / max(len(val_loader), 1),
        }
        history.append({"epoch": epoch, **metrics})
        print(
            f"Epoch {epoch}/{max_epochs} macro={metrics['macro_accuracy']:.4f} "
            f"C={metrics['c_accuracy']:.4f} M={metrics['m_accuracy']:.4f} X={metrics['x_accuracy']:.4f}"
        )

        if macro_accuracy > best_accuracy:
            best_accuracy = macro_accuracy
            best_metrics = metrics
            torch.save(
                {
                    "architecture": "cmx_cnn",
                    "model_state_dict": model.state_dict(),
                    "classes": ["C", "M", "X"],
                    "thresholds": THRESHOLDS,
                    "channel": channel,
                    "metrics": best_metrics,
                },
                output,
            )
            print(f"Novo melhor modelo C/M/X salvo em {output}")

        if best_accuracy >= target_accuracy:
            print(f"Alvo atingido: macro_accuracy={best_accuracy:.4f}")
            break

    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina modelo multi-label para probabilidades C/M/X.")
    parser.add_argument("--raw-dir", default="data/sdobenchmark", type=Path)
    parser.add_argument("--output", default="ml/models/solar_flare_cmx.pt", type=Path)
    parser.add_argument("--channel", default="all")
    parser.add_argument("--max-epochs", default=80, type=int)
    parser.add_argument("--target-accuracy", default=0.90, type=float)
    parser.add_argument("--max-images", default=16000, type=int)
    parser.add_argument("--batch-size", default=64, type=int)
    args = parser.parse_args()
    print(json.dumps(train(args.raw_dir, args.output, args.channel, args.max_epochs, args.target_accuracy, args.max_images, args.batch_size), indent=2))

