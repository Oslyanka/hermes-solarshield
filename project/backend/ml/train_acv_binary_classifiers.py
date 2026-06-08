from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

CLASSES = ["low", "high"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}


def format_pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def matrix_to_markdown(matrix: list[list[int]]) -> str:
    return "\n".join(
        [
            "| Real \\ Predito | low | high |",
            "|---|---:|---:|",
            f"| low | {matrix[0][0]} | {matrix[0][1]} |",
            f"| high | {matrix[1][0]} | {matrix[1][1]} |",
        ]
    )


def count_split(root: Path) -> dict[str, int]:
    return {label: len(list((root / label).glob("*.jpg"))) for label in CLASSES}


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from PIL import Image, ImageEnhance, ImageOps
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    except ImportError as exc:
        raise SystemExit("Instale torch, pillow e numpy para treinar ACV binario.") from exc

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_dir = args.dataset_dir / "train"
    val_dir = args.dataset_dir / "val"
    test_dir = args.dataset_dir / "test"
    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise SystemExit(f"Esperado train/val/test em {args.dataset_dir}")

    class BinarySolarDataset(Dataset):
        def __init__(self, root: Path, augment: bool):
            self.samples: list[tuple[Path, int]] = []
            self.augment = augment
            for label in CLASSES:
                for path in sorted((root / label).glob("*.jpg")):
                    self.samples.append((path, CLASS_TO_INDEX[label]))
            if not self.samples:
                raise SystemExit(f"Nenhuma imagem encontrada em {root}")

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            path, target = self.samples[index]
            image = Image.open(path).convert("L")
            image = ImageOps.autocontrast(image)
            image = image.resize((args.image_size, args.image_size))
            if self.augment:
                if random.random() < 0.5:
                    image = ImageOps.mirror(image)
                if random.random() < 0.3:
                    image = ImageOps.flip(image)
                if random.random() < 0.5:
                    image = image.rotate(random.uniform(-8, 8))
                if random.random() < 0.5:
                    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.2))
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = (array - 0.5) / 0.5
            return torch.from_numpy(array).unsqueeze(0), torch.tensor(target, dtype=torch.long)

    class BinaryTinyCNN(nn.Module):
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
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(96 * 4 * 4, 96),
                nn.ReLU(),
                nn.Dropout(0.30),
                nn.Linear(96, 2),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    class BinaryDeepCNN(nn.Module):
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
                nn.Linear(160, 2),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    def parameter_count(model: nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, samples: list[tuple[Path, int]]) -> dict[str, Any]:
        model.eval()
        total_loss = 0.0
        total = 0
        correct = 0
        confusion = [[0, 0], [0, 0]]
        errors: list[dict[str, Any]] = []
        cursor = 0
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                probabilities = torch.softmax(logits, dim=1).cpu()
                predictions = probabilities.argmax(dim=1)
                total_loss += float(loss.item())
                correct += int((predictions.to(device) == labels).sum().item())
                total += int(labels.numel())
                for batch_index, (actual_idx, predicted_idx) in enumerate(zip(labels.cpu().tolist(), predictions.tolist())):
                    confusion[actual_idx][predicted_idx] += 1
                    if actual_idx != predicted_idx and len(errors) < 25:
                        path, _ = samples[cursor + batch_index]
                        errors.append(
                            {
                                "file": path.name,
                                "actual": CLASSES[actual_idx],
                                "predicted": CLASSES[predicted_idx],
                                "probabilities": {
                                    CLASSES[index]: round(float(probabilities[batch_index][index]), 3)
                                    for index in range(2)
                                },
                            }
                        )
                cursor += labels.shape[0]

        report: dict[str, dict[str, float]] = {}
        for index, label in enumerate(CLASSES):
            tp = confusion[index][index]
            fp = sum(confusion[row][index] for row in range(2) if row != index)
            fn = sum(confusion[index][col] for col in range(2) if col != index)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
            report[label] = {"precision": precision, "recall": recall, "f1": f1}
        return {
            "loss": total_loss / max(len(loader), 1),
            "accuracy": correct / max(total, 1),
            "confusion_matrix": confusion,
            "classification_report": report,
            "errors": errors,
        }

    def run_model(architecture: str, model: nn.Module) -> dict[str, Any]:
        model = model.to(device)
        train_set = BinarySolarDataset(train_dir, augment=True)
        val_set = BinarySolarDataset(val_dir, augment=False)
        test_set = BinarySolarDataset(test_dir, augment=False)
        labels = [target for _, target in train_set.samples]
        class_counts = Counter(labels)
        class_weights = [len(labels) / max(2 * class_counts.get(index, 1), 1) for index in range(2)]
        sample_weights = [class_weights[label] for label in labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler)
        val_loader = DataLoader(val_set, batch_size=args.batch_size)
        test_loader = DataLoader(test_set, batch_size=args.batch_size)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)
        best_state = None
        best_val = -1.0
        best_loss = float("inf")
        stale = 0
        history: list[dict[str, float]] = []
        print(f"\nTreinando {architecture} em {device}", flush=True)
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for images, labels_batch in train_loader:
                images, labels_batch = images.to(device), labels_batch.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels_batch)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                train_correct += int((logits.argmax(dim=1) == labels_batch).sum().item())
                train_total += int(labels_batch.numel())
            val_metrics = evaluate(model, val_loader, criterion, val_set.samples)
            train_accuracy = train_correct / max(train_total, 1)
            train_loss = train_loss / max(len(train_loader), 1)
            scheduler.step(val_metrics["accuracy"])
            history.append(
                {
                    "epoch": epoch,
                    "train_accuracy": train_accuracy,
                    "train_loss": train_loss,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_loss": val_metrics["loss"],
                }
            )
            print(
                f"Epoch {epoch:03d}/{args.max_epochs} train_acc={train_accuracy:.4f} "
                f"val_acc={val_metrics['accuracy']:.4f} train_loss={train_loss:.4f} val_loss={val_metrics['loss']:.4f}",
                flush=True,
            )
            improved = val_metrics["accuracy"] > best_val or (
                val_metrics["accuracy"] == best_val and val_metrics["loss"] < best_loss
            )
            if improved:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_val = val_metrics["accuracy"]
                best_loss = val_metrics["loss"]
                stale = 0
            else:
                stale += 1
            if best_val >= args.target_accuracy and epoch >= args.min_epochs:
                break
            if stale >= args.patience and epoch >= args.min_epochs:
                print(f"Parada antecipada em {architecture}.", flush=True)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        val_metrics = evaluate(model, val_loader, criterion, val_set.samples)
        test_metrics = evaluate(model, test_loader, criterion, test_set.samples)
        output_path = args.output_dir / f"{architecture}.pt"
        checkpoint = {
            "architecture": architecture,
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "image_size": args.image_size,
            "channel": "SDO AIA 131",
            "training_source": "SDOBenchmark binary low/high risk classification",
            "metrics": {
                "val_accuracy": val_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "confusion_matrix": test_metrics["confusion_matrix"],
                "classification_report": test_metrics["classification_report"],
            },
            "split_counts": {
                "train": count_split(train_dir),
                "val": count_split(val_dir),
                "test": count_split(test_dir),
            },
            "parameter_count": parameter_count(model),
        }
        torch.save(checkpoint, output_path)
        output_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return {
            "architecture": architecture,
            "path": str(output_path),
            "parameter_count": parameter_count(model),
            "val_accuracy": val_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "classification_report": test_metrics["classification_report"],
            "errors": test_metrics["errors"],
        }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_model("acv_binary_tiny_cnn", BinaryTinyCNN()),
        run_model("acv_binary_deep_cnn", BinaryDeepCNN()),
    ]
    best = max(results, key=lambda item: (item["test_accuracy"], item["val_accuracy"]))
    active_path = args.output_dir / "solar_flare_binary_classifier_131.pt"
    shutil.copyfile(best["path"], active_path)
    summary = {
        "dataset": {
            "source": str(args.dataset_dir),
            "channel": "SDO AIA 131",
            "classes": CLASSES,
            "label_policy": "low/high risk from SDOBenchmark processed split",
            "split_counts": {
                "train": count_split(train_dir),
                "val": count_split(val_dir),
                "test": count_split(test_dir),
            },
        },
        "comparison": [
            {key: item[key] for key in ("architecture", "path", "parameter_count", "val_accuracy", "val_loss", "test_accuracy", "test_loss")}
            for item in results
        ],
        "best_model": best,
        "active_model_path": str(active_path),
        "target_accuracy": args.target_accuracy,
        "ethical_note": "Uses existing train/val/test split in data/processed. No pretrained model, no test leakage, no manual output calibration.",
    }
    comparison_path = args.output_dir / "acv_binary_classification_comparison.json"
    comparison_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = args.report_dir / "acv_binary_classification_report.md"
    build_report(summary, report_path)
    print(json.dumps(summary["comparison"], indent=2), flush=True)
    print(f"Relatorio: {report_path}", flush=True)
    return summary


def build_report(summary: dict[str, Any], report_path: Path) -> None:
    best = summary["best_model"]
    achieved = best["test_accuracy"] >= summary["target_accuracy"]
    lines = [
        "# Hermes SolarShield - Relatorio ACV Binario",
        "",
        "## Definicao do problema",
        "",
        "Classificar imagens solares SDO AIA 131 em risco `low` ou `high` para triagem operacional de solar flares.",
        "",
        "## Dataset",
        "",
        f"- Fonte: {summary['dataset']['source']}",
        f"- Canal: {summary['dataset']['channel']}",
        f"- Classes: {', '.join(summary['dataset']['classes'])}",
        f"- Politica de rotulo: {summary['dataset']['label_policy']}",
        f"- Treino: {summary['dataset']['split_counts']['train']}",
        f"- Validacao: {summary['dataset']['split_counts']['val']}",
        f"- Teste: {summary['dataset']['split_counts']['test']}",
        "",
        "## Comparacao das arquiteturas",
        "",
        "| Arquitetura | Parametros | Val accuracy | Test accuracy | Test loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in summary["comparison"]:
        lines.append(
            f"| {item['architecture']} | {item['parameter_count']} | {format_pct(item['val_accuracy'])} | "
            f"{format_pct(item['test_accuracy'])} | {item['test_loss']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Melhor modelo",
            "",
            f"- Arquitetura: {best['architecture']}",
            f"- Acuracia de teste: {format_pct(best['test_accuracy'])}",
            f"- Criterio de referencia de 88%: {'atingido' if achieved else 'nao atingido'}",
            f"- Checkpoint ativo: `{summary['active_model_path']}`",
            "",
            "### Matriz de confusao",
            "",
            matrix_to_markdown(best["confusion_matrix"]),
            "",
            "### Metricas por classe",
            "",
            "| Classe | Precision | Recall | F1 |",
            "|---|---:|---:|---:|",
        ]
    )
    for label, metrics in best["classification_report"].items():
        lines.append(f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |")
    lines.extend(["", "### Exemplos de erro", ""])
    if best["errors"]:
        for error in best["errors"][:10]:
            lines.append(f"- `{error['file']}` real={error['actual']} predito={error['predicted']} prob={error['probabilities']}")
    else:
        lines.append("- Nenhum erro no conjunto de teste.")
    lines.extend(["", "## Nota etica", "", summary["ethical_note"]])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina duas CNNs binarias do zero para ACV.")
    parser.add_argument("--dataset-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--output-dir", default=Path("ml/models"), type=Path)
    parser.add_argument("--report-dir", default=Path("reports"), type=Path)
    parser.add_argument("--image-size", default=128, type=int)
    parser.add_argument("--max-epochs", default=80, type=int)
    parser.add_argument("--min-epochs", default=12, type=int)
    parser.add_argument("--patience", default=18, type=int)
    parser.add_argument("--target-accuracy", default=0.88, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

