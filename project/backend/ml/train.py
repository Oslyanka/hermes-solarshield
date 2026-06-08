from __future__ import annotations

import argparse
import json
from pathlib import Path


def train(
    dataset_dir: Path,
    max_epochs: int,
    output: Path,
    target_accuracy: float,
    batch_size: int,
    learning_rate: float,
) -> dict[str, float]:
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    if not train_dir.exists() or not val_dir.exists():
        raise SystemExit(
            "Dataset processado nao encontrado.\n"
            f"Esperado: {train_dir} e {val_dir}\n"
            "Rode: python ml/prepare_dataset.py --raw-dir data/sdobenchmark --output-dir data/processed"
        )

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, Dataset
        from PIL import Image
        import numpy as np
    except ImportError as exc:
        raise SystemExit("Para treinar o modelo real, instale: pip install torch pillow numpy") from exc

    class SolarImageDataset(Dataset):
        def __init__(self, root: Path):
            self.samples = []
            for label, target in (("low", 0.0), ("high", 1.0)):
                for path in sorted((root / label).glob("*.jpg")):
                    self.samples.append((path, target))
            if not self.samples:
                raise SystemExit(f"Nenhuma imagem encontrada em {root}")

        def __len__(self):
            return len(self.samples)

        def __getitem__(self, index):
            path, target = self.samples[index]
            image = Image.open(path).convert("L").resize((128, 128))
            array = np.asarray(image, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(array).unsqueeze(0)
            return tensor, torch.tensor([target], dtype=torch.float32)

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = SolarImageDataset(train_dir)
    val_set = SolarImageDataset(val_dir)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=0)

    model = TinySolarCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    positives = sum(1 for _, target in train_set.samples if target == 1.0)
    negatives = len(train_set) - positives
    pos_weight = torch.tensor([negatives / max(positives, 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_accuracy = 0.0
    best_loss = float("inf")
    history = []
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Treinando em {device} com {len(train_set)} imagens de treino e {len(val_set)} de validacao")
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                predictions = (torch.sigmoid(logits) >= 0.5).float()
                correct += int((predictions == labels).sum().item())
                total += int(labels.numel())
                val_loss += float(loss.item())

        train_loss /= max(len(train_loader), 1)
        val_loss /= max(len(val_loader), 1)
        val_accuracy = correct / max(total, 1)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_accuracy})
        print(
            f"Epoch {epoch}/{max_epochs} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f}"
        )

        if val_accuracy > best_accuracy or (val_accuracy == best_accuracy and val_loss < best_loss):
            best_accuracy = val_accuracy
            best_loss = val_loss
            torch.save(
                {
                    "architecture": "tiny_solar_cnn",
                    "model_state_dict": model.state_dict(),
                    "image_size": 128,
                    "input_channels": 1,
                    "classes": ["low", "high"],
                    "val_accuracy": best_accuracy,
                },
                output,
            )
            print(f"Novo melhor modelo salvo em {output} com val_accuracy={best_accuracy:.4f}")

        if best_accuracy >= target_accuracy:
            print(f"Alvo atingido: val_accuracy={best_accuracy:.4f} >= {target_accuracy:.4f}")
            break

    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return {"best_accuracy": best_accuracy, "best_loss": best_loss}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina uma CNN para risco de solar flare ate atingir acuracia alvo.")
    parser.add_argument("--dataset-dir", default="data/processed", type=Path)
    parser.add_argument("--max-epochs", default=30, type=int)
    parser.add_argument("--target-accuracy", default=0.90, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--output", default="ml/models/solar_flare_cnn.pt", type=Path)
    args = parser.parse_args()
    metrics = train(
        dataset_dir=args.dataset_dir,
        max_epochs=args.max_epochs,
        output=args.output,
        target_accuracy=args.target_accuracy,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    print(json.dumps(metrics, indent=2))
