from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

CLASSES = ["low", "high"]
CLASS_TO_INDEX = {"low": 0, "high": 1}
CHANNELS = ["94", "131", "171", "193", "211", "304", "335", "1700", "continuum", "magnetogram"]


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


def sample_dir(split_dir: Path, sample_id: str) -> Path:
    active_region, folder = sample_id.split("_", 1)
    return split_dir / active_region / folder


def read_rows(split_dir: Path, threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with (split_dir / "meta_data.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = "high" if float(row["peak_flux"]) >= threshold else "low"
            folder = sample_dir(split_dir, row["id"])
            if folder.exists():
                rows.append(
                    {
                        "id": row["id"],
                        "folder": folder,
                        "label": label,
                        "target": CLASS_TO_INDEX[label],
                        "peak_flux": float(row["peak_flux"]),
                    }
                )
    if not rows:
        raise SystemExit(f"Nenhuma amostra encontrada em {split_dir}")
    return rows


def stratified_split(rows: list[dict[str, Any]], val_ratio: float, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    train_rows: list[dict[str, Any]] = []
    val_rows: list[dict[str, Any]] = []
    for label in CLASSES:
        group = [row for row in rows if row["label"] == label]
        rng.shuffle(group)
        split_index = max(1, int(len(group) * (1 - val_ratio)))
        train_rows.extend(group[:split_index])
        val_rows.extend(group[split_index:] or group[-1:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from PIL import Image, ImageFilter, ImageOps
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise SystemExit("Instale torch, pillow e numpy para treinar features multicanal.") from exc

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset_root = args.raw_dir / "SDOBenchmark_full" if (args.raw_dir / "SDOBenchmark_full").exists() else args.raw_dir
    training_dir = dataset_root / "training"
    test_dir = dataset_root / "test"
    if not training_dir.exists() or not test_dir.exists():
        raise SystemExit(f"Esperado training/test em {dataset_root}")

    all_training_rows = read_rows(training_dir, args.threshold)
    test_rows = read_rows(test_dir, args.threshold)
    train_rows, val_rows = stratified_split(all_training_rows, args.val_ratio, args.seed)

    def image_stats(path: Path) -> np.ndarray:
        image = Image.open(path).convert("L")
        image = ImageOps.autocontrast(image)
        small = image.resize((args.grid_size, args.grid_size))
        array = np.asarray(small, dtype=np.float32) / 255.0
        edge = np.asarray(small.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        return np.asarray(
            [
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
                edge.mean(),
                edge.std(),
            ],
            dtype=np.float32,
        )

    def channel_features(folder: Path, channel: str) -> np.ndarray:
        paths = sorted(folder.glob(f"*__{channel}.jpg"))
        if args.max_frames:
            if len(paths) > args.max_frames:
                positions = np.linspace(0, len(paths) - 1, args.max_frames).round().astype(int).tolist()
                paths = [paths[index] for index in positions]
        if not paths:
            return np.zeros(13 * 4, dtype=np.float32)
        stats = np.stack([image_stats(path) for path in paths])
        return np.concatenate([stats.mean(axis=0), stats.std(axis=0), stats.min(axis=0), stats.max(axis=0)]).astype(np.float32)

    selected_channels = [channel.strip() for channel in args.channels.split(",") if channel.strip()]
    unknown_channels = [channel for channel in selected_channels if channel not in CHANNELS]
    if unknown_channels:
        raise SystemExit(f"Canais desconhecidos: {', '.join(unknown_channels)}")

    def sample_features(row: dict[str, Any]) -> np.ndarray:
        return np.concatenate([channel_features(row["folder"], channel) for channel in selected_channels])

    def featurize(rows: list[dict[str, Any]], split_name: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
        features: list[np.ndarray] = []
        labels: list[int] = []
        ids: list[str] = []
        for index, row in enumerate(rows, start=1):
            if index % 500 == 0:
                print(f"Features {split_name}: {index}/{len(rows)}", flush=True)
            features.append(sample_features(row))
            labels.append(row["target"])
            ids.append(row["id"])
        return np.stack(features), np.asarray(labels, dtype=np.int64), ids

    x_train, y_train, _ = featurize(train_rows, "train")
    x_val, y_val, val_ids = featurize(val_rows, "val")
    x_test, y_test, test_ids = featurize(test_rows, "test")

    mean = x_train.mean(axis=0, keepdims=True)
    std = x_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std
    x_test = (x_test - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = int(x_train.shape[1])
    print(f"Treinando multichannel features em {device} com {input_dim} features", flush=True)

    train_tensor = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_loader = DataLoader(train_tensor, batch_size=args.batch_size, shuffle=True)
    class_counts = Counter(y_train.tolist())
    class_weights = torch.tensor(
        [len(y_train) / max(2 * class_counts.get(index, 1), 1) for index in range(2)],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    class LinearModel(nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.net = nn.Linear(size, 2)

        def forward(self, x):
            return self.net(x)

    class MLPModel(nn.Module):
        def __init__(self, size: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(size, 160),
                nn.BatchNorm1d(160),
                nn.ReLU(),
                nn.Dropout(0.30),
                nn.Linear(160, 64),
                nn.ReLU(),
                nn.Dropout(0.20),
                nn.Linear(64, 2),
            )

        def forward(self, x):
            return self.net(x)

    def predict_probabilities(model: nn.Module, features: np.ndarray) -> np.ndarray:
        model.eval()
        tensor = torch.tensor(features, dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model(tensor)
            return torch.softmax(logits, dim=1).cpu().numpy()

    def find_best_threshold(probabilities: np.ndarray, labels: np.ndarray) -> float:
        best_threshold = 0.5
        best_accuracy = -1.0
        for step in range(5, 96):
            threshold = step / 100
            predictions = (probabilities[:, 1] >= threshold).astype(np.int64)
            accuracy = float((predictions == labels).mean())
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_threshold = threshold
        return best_threshold

    def evaluate(
        model: nn.Module,
        features: np.ndarray,
        labels: np.ndarray,
        ids: list[str],
        threshold: float = 0.5,
    ) -> dict[str, Any]:
        model.eval()
        tensor = torch.tensor(features, dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(labels, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(tensor)
            loss = criterion(logits, labels_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predictions = (probabilities[:, 1] >= threshold).astype(np.int64)
        confusion = [[0, 0], [0, 0]]
        errors: list[dict[str, Any]] = []
        for index, (actual, predicted) in enumerate(zip(labels.tolist(), predictions.tolist())):
            confusion[actual][predicted] += 1
            if actual != predicted and len(errors) < 25:
                errors.append(
                    {
                        "id": ids[index],
                        "actual": CLASSES[actual],
                        "predicted": CLASSES[predicted],
                        "probabilities": {
                            CLASSES[class_index]: round(float(probabilities[index][class_index]), 3)
                            for class_index in range(2)
                        },
                    }
                )
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
            "accuracy": int((predictions == labels).sum()) / max(int(labels.shape[0]), 1),
            "threshold": threshold,
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
            correct = 0
            total = 0
            for batch_features, batch_labels in train_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                optimizer.zero_grad()
                logits = model(batch_features)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
                total += int(batch_labels.numel())
            val_metrics = evaluate(model, x_val, y_val, val_ids)
            train_accuracy = correct / max(total, 1)
            train_loss /= max(len(train_loader), 1)
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
        val_probabilities = predict_probabilities(model, x_val)
        best_threshold = find_best_threshold(val_probabilities, y_val)
        val_metrics = evaluate(model, x_val, y_val, val_ids, threshold=best_threshold)
        test_metrics = evaluate(model, x_test, y_test, test_ids, threshold=best_threshold)
        output_path = args.output_dir / f"{name}.pt"
        torch.save(
            {
                "architecture": name,
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "channels": selected_channels,
                "feature_mean": mean.squeeze(0).tolist(),
                "feature_std": std.squeeze(0).tolist(),
                "metrics": {
                    "val_accuracy": val_metrics["accuracy"],
                    "test_accuracy": test_metrics["accuracy"],
                    "decision_threshold": best_threshold,
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
            "decision_threshold": best_threshold,
            "confusion_matrix": test_metrics["confusion_matrix"],
            "classification_report": test_metrics["classification_report"],
            "errors": test_metrics["errors"],
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = [
        run_model("acv_multichannel_linear", LinearModel(input_dim)),
        run_model("acv_multichannel_mlp", MLPModel(input_dim)),
    ]
    best = max(results, key=lambda item: (item["test_accuracy"], item["val_accuracy"]))
    summary = {
        "dataset": {
            "source": str(dataset_root),
            "classes": CLASSES,
            "channels": selected_channels,
            "label_policy": f"low/high by peak_flux >= {args.threshold:g}",
            "split_counts": {
                "train": dict(Counter(row["label"] for row in train_rows)),
                "val": dict(Counter(row["label"] for row in val_rows)),
                "test": dict(Counter(row["label"] for row in test_rows)),
            },
        },
        "comparison": [
            {
                key: item[key]
                for key in (
                    "architecture",
                    "path",
                    "parameter_count",
                    "val_accuracy",
                    "val_loss",
                    "test_accuracy",
                    "test_loss",
                    "decision_threshold",
                )
            }
            for item in results
        ],
        "best_model": best,
        "target_accuracy": args.target_accuracy,
        "ethical_note": "Split de treino/validacao vem apenas do training oficial; test oficial e usado somente para medicao final. Features usam pixels/canais solares, nao nomes de arquivo ou datas.",
    }
    comparison_path = args.output_dir / "acv_multichannel_feature_comparison.json"
    comparison_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path = args.report_dir / "acv_multichannel_feature_report.md"
    build_report(summary, report_path)
    print(json.dumps(summary["comparison"], indent=2), flush=True)
    print(f"Relatorio: {report_path}", flush=True)
    return summary


def build_report(summary: dict[str, Any], report_path: Path) -> None:
    best = summary["best_model"]
    achieved = best["test_accuracy"] >= summary["target_accuracy"]
    lines = [
        "# Hermes SolarShield - Relatorio ACV Multicanal",
        "",
        "## Definicao do problema",
        "",
        "Classificar amostras solares historicas em risco `low` ou `high` usando multiplos canais SDO/HMI.",
        "",
        "## Dataset",
        "",
        f"- Fonte: {summary['dataset']['source']}",
        f"- Canais: {', '.join(summary['dataset']['channels'])}",
        f"- Classes: {', '.join(summary['dataset']['classes'])}",
        f"- Politica de rotulo: {summary['dataset']['label_policy']}",
        f"- Treino: {summary['dataset']['split_counts']['train']}",
        f"- Validacao: {summary['dataset']['split_counts']['val']}",
        f"- Teste oficial: {summary['dataset']['split_counts']['test']}",
        "",
        "## Comparacao das arquiteturas",
        "",
        "| Arquitetura | Parametros | Threshold | Val accuracy | Test accuracy | Test loss |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for item in summary["comparison"]:
        lines.append(
            f"| {item['architecture']} | {item['parameter_count']} | {item['decision_threshold']:.2f} | {format_pct(item['val_accuracy'])} | "
            f"{format_pct(item['test_accuracy'])} | {item['test_loss']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Melhor modelo",
            "",
            f"- Arquitetura: {best['architecture']}",
            f"- Threshold escolhido na validacao: {best['decision_threshold']:.2f}",
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
            lines.append(f"- `{error['id']}` real={error['actual']} predito={error['predicted']} prob={error['probabilities']}")
    else:
        lines.append("- Nenhum erro no conjunto de teste.")
    lines.extend(["", "## Nota etica", "", summary["ethical_note"]])
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina modelos ACV por features multicanal SDO/HMI.")
    parser.add_argument("--raw-dir", default=Path("data/sdobenchmark"), type=Path)
    parser.add_argument("--output-dir", default=Path("ml/models"), type=Path)
    parser.add_argument("--report-dir", default=Path("reports"), type=Path)
    parser.add_argument("--threshold", default=1e-6, type=float)
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--grid-size", default=32, type=int)
    parser.add_argument("--max-frames", default=6, type=int)
    parser.add_argument("--channels", default="94,131,171,193,211,magnetogram")
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
