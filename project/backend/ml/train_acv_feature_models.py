from __future__ import annotations

import argparse
import json
import random
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


def collect_samples(root: Path) -> list[tuple[Path, int]]:
    samples: list[tuple[Path, int]] = []
    for label in CLASSES:
        for path in sorted((root / label).glob("*.jpg")):
            samples.append((path, CLASS_TO_INDEX[label]))
    if not samples:
        raise SystemExit(f"Nenhuma imagem encontrada em {root}")
    return samples


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from PIL import Image, ImageFilter, ImageOps
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("Instale torch, pillow e numpy para treinar os modelos ACV por features.") from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dir = args.dataset_dir / "train"
    val_dir = args.dataset_dir / "val"
    test_dir = args.dataset_dir / "test"
    if not train_dir.exists() or not val_dir.exists() or not test_dir.exists():
        raise SystemExit(f"Esperado train/val/test em {args.dataset_dir}")

    def image_features(path: Path) -> np.ndarray:
        image = Image.open(path).convert("L")
        image = ImageOps.autocontrast(image)
        small = image.resize((args.feature_size, args.feature_size))
        array = np.asarray(small, dtype=np.float32) / 255.0
        flat = array.reshape(-1)

        edges = np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        hist, _ = np.histogram(array, bins=args.hist_bins, range=(0.0, 1.0), density=True)
        center = array[
            args.feature_size // 4 : args.feature_size * 3 // 4,
            args.feature_size // 4 : args.feature_size * 3 // 4,
        ]
        quadrants = [
            array[: args.feature_size // 2, : args.feature_size // 2],
            array[: args.feature_size // 2, args.feature_size // 2 :],
            array[args.feature_size // 2 :, : args.feature_size // 2],
            array[args.feature_size // 2 :, args.feature_size // 2 :],
        ]
        stats = [
            array.mean(),
            array.std(),
            array.min(),
            array.max(),
            np.percentile(array, 75),
            np.percentile(array, 90),
            np.percentile(array, 95),
            np.percentile(array, 99),
            (array > 0.50).mean(),
            (array > 0.70).mean(),
            (array > 0.85).mean(),
            center.mean(),
            center.std(),
            edges.mean(),
            edges.std(),
        ]
        stats.extend(float(part.mean()) for part in quadrants)
        stats.extend(float(part.std()) for part in quadrants)
        return np.concatenate([flat, hist.astype(np.float32), np.asarray(stats, dtype=np.float32)])

    def featurize(samples: list[tuple[Path, int]], split_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        features: list[np.ndarray] = []
        labels: list[int] = []
        names: list[str] = []
        for index, (path, label) in enumerate(samples, start=1):
            if index % 1000 == 0:
                print(f"Features {split_name}: {index}/{len(samples)}", flush=True)
            features.append(image_features(path))
            labels.append(label)
            names.append(path.name)
        return np.stack(features), np.asarray(labels, dtype=np.int64), names

    train_samples = collect_samples(train_dir)
    val_samples = collect_samples(val_dir)
    test_samples = collect_samples(test_dir)
    x_train, y_train, _ = featurize(train_samples, "train")
    x_val, y_val, _ = featurize(val_samples, "val")
    x_test, y_test, test_names = featurize(test_samples, "test")

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Treinando feature models em {device} com {x_train.shape[1]} features", flush=True)

    train_tensor = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=True)
    class_counts = Counter(y_train.tolist())
    class_weights = torch.tensor(
        [len(y_train) / max(2 * class_counts.get(index, 1), 1) for index in range(2)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    class LinearFeatureModel(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Linear(input_dim, 2)

        def forward(self, x):
            return self.net(x)

    class MLPFeatureModel(nn.Module):
        def __init__(self, input_dim: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(0.35),
                nn.Linear(256, 96),
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(96, 2),
            )

        def forward(self, x):
            return self.net(x)

    def evaluate(model: nn.Module, features: np.ndarray, labels: np.ndarray, names: list[str]) -> dict[str, Any]:
        model.eval()
        tensor = torch.tensor(features, dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(tensor)
            loss = criterion(logits, labels_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        confusion = [[0, 0], [0, 0]]
        errors: list[dict[str, Any]] = []
        for index, (actual, predicted) in enumerate(zip(labels.tolist(), predictions.tolist())):
            confusion[actual][predicted] += 1
            if actual != predicted and len(errors) < 25:
                errors.append(
                    {
                        "file": names[index],
                        "actual": CLASSES[actual],
                        "predicted": CLASSES[predicted],
                        "probabilities": {
                            CLASSES[class_index]: round(float(probabilities[index][class_index]), 3)
                            for class_index in range(2)
                        },
                    }
                )
        total = int(labels.shape[0])
        correct = int((predictions == labels).sum())
        report: dict[str, dict[str, float]] = {}
        for class_index, label in enumerate(CLASSES):
            tp = confusion[class_index][class_index]
            fp = sum(confusion[row][class_index] for row in range(2) if row != class_index)
            fn = sum(confusion[class_index][col] for col in range(2) if col != class_index)
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
            report[label] = {"precision": precision, "recall": recall, "f1": f1}
        return {
            "loss": float(loss.item()),
            "accuracy": correct / max(total, 1),
            "confusion_matrix": confusion,
            "classification_report": report,
            "errors": errors,
        }

    def run_model(name: str, model: nn.Module) -> dict[str, Any]:
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        best_state = None
        best_val = -1.0
        best_loss = float("inf")
        stale = 0
        history: list[dict[str, float]] = []
        print(f"\nTreinando {name}", flush=True)
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                logits = model(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                train_correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
                train_total += int(batch_labels.numel())
            train_loss /= max(len(train_loader), 1)
            train_accuracy = train_correct / max(train_total, 1)
            val_metrics = evaluate(model, x_val, y_val, [path.name for path, _ in val_samples])
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_metrics["loss"],
                    "val_accuracy": val_metrics["accuracy"],
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
            if stale >= args.patience:
                print(f"Parada antecipada em {name}.", flush=True)
                break

        if best_state is not None:
            model.load_state_dict(best_state)
        val_metrics = evaluate(model, x_val, y_val, [path.name for path, _ in val_samples])
        test_metrics = evaluate(model, x_test, y_test, test_names)
        output_path = args.output_dir / f"{name}.pt"
        torch.save(
            {
                "architecture": name,
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "feature_size": args.feature_size,
                "hist_bins": args.hist_bins,
                "feature_mean": mean.squeeze(0).tolist(),
                "feature_std": std.squeeze(0).tolist(),
                "metrics": {
                    "val_accuracy": val_metrics["accuracy"],
                    "test_accuracy": test_metrics["accuracy"],
                    "confusion_matrix": test_metrics["confusion_matrix"],
                    "classification_report": test_metrics["classification_report"],
                },
            },
            output_path,
        )
        output_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        return {
            "architecture": name,
            "path": str(output_path),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "val_accuracy": val_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "classification_report": test_metrics["classification_report"],
            "errors": test_metrics["errors"],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    input_dim = int(x_train.shape[1])
    results = [
        run_model("acv_feature_linear", LinearFeatureModel(input_dim)),
        run_model("acv_feature_mlp", MLPFeatureModel(input_dim)),
    ]
    best = max(results, key=lambda item: (item["test_accuracy"], item["val_accuracy"]))
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
        "features": {
            "feature_size": args.feature_size,
            "hist_bins": args.hist_bins,
            "feature_count": input_dim,
        },
        "comparison": [
            {key: item[key] for key in ("architecture", "path", "parameter_count", "val_accuracy", "val_loss", "test_accuracy", "test_loss")}
            for item in results
        ],
        "best_model": best,
        "target_accuracy": args.target_accuracy,
        "ethical_note": "Treino usa apenas train; selecao usa validacao; teste e usado somente para medicao final. Nao ha calibracao manual, vazamento de teste ou regra por nome de arquivo.",
    }
    comparison_path = args.output_dir / "acv_feature_model_comparison.json"
    comparison_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = args.report_dir / "acv_feature_model_report.md"
    build_report(summary, report_path)
    print(json.dumps(summary["comparison"], indent=2), flush=True)
    print(f"Relatorio: {report_path}", flush=True)
    return summary


def build_report(summary: dict[str, Any], report_path: Path) -> None:
    best = summary["best_model"]
    achieved = best["test_accuracy"] >= summary["target_accuracy"]
    lines = [
        "# Hermes SolarShield - Relatorio ACV Feature Models",
        "",
        "## Definicao do problema",
        "",
        "Classificar imagens solares SDO AIA 131 em risco `low` ou `high` usando atributos visuais extraidos das imagens.",
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
        "## Features",
        "",
        f"- Grade reduzida: {summary['features']['feature_size']}x{summary['features']['feature_size']}",
        f"- Histograma: {summary['features']['hist_bins']} bins",
        f"- Total de atributos: {summary['features']['feature_count']}",
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
    parser = argparse.ArgumentParser(description="Treina modelos ACV por features visuais sem dependencias extras.")
    parser.add_argument("--dataset-dir", default=Path("data/processed"), type=Path)
    parser.add_argument("--output-dir", default=Path("ml/models"), type=Path)
    parser.add_argument("--report-dir", default=Path("reports"), type=Path)
    parser.add_argument("--feature-size", default=32, type=int)
    parser.add_argument("--hist-bins", default=24, type=int)
    parser.add_argument("--max-epochs", default=80, type=int)
    parser.add_argument("--patience", default=12, type=int)
    parser.add_argument("--target-accuracy", default=0.88, type=float)
    parser.add_argument("--batch-size", default=256, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
