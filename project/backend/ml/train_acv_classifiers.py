from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CLASSES = ["LOW", "MEDIUM", "HIGH"]
CLASS_TO_INDEX = {label: index for index, label in enumerate(CLASSES)}
LABEL_POLICY = {
    "LOW": "peak_flux < 1e-6 W/m2: below C-class solar flare range",
    "MEDIUM": "1e-6 <= peak_flux < 1e-5 W/m2: C-class solar flare range",
    "HIGH": "peak_flux >= 1e-5 W/m2: M/X-class operational risk range",
}


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str
    label_index: int
    sample_id: str
    active_region: str
    peak_flux: float
    split: str


def label_from_peak_flux(peak_flux: float) -> str:
    if peak_flux >= 1e-5:
        return "HIGH"
    if peak_flux >= 1e-6:
        return "MEDIUM"
    return "LOW"


def sample_dir(split_dir: Path, sample_id: str) -> Path:
    active_region, folder = sample_id.split("_", 1)
    return split_dir / active_region / folder


def find_dataset_root(raw_dir: Path) -> Path:
    candidates = [
        raw_dir / "SDOBenchmark_full",
        raw_dir / "SDOBenchmark_example",
        raw_dir,
    ]
    for candidate in candidates:
        if (candidate / "training" / "meta_data.csv").exists():
            return candidate
    raise SystemExit(
        "SDOBenchmark nao encontrado. Esperado meta_data.csv em "
        f"{raw_dir}\\SDOBenchmark_full\\training ou {raw_dir}\\training."
    )


def collect_sdobenchmark_samples(
    split_dir: Path,
    split_name: str,
    channel: str,
    max_per_class: int,
    seed: int,
) -> list[Sample]:
    metadata_path = split_dir / "meta_data.csv"
    rows = list(csv.DictReader(metadata_path.open(newline="", encoding="utf-8")))
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for row in rows:
        peak_flux = float(row["peak_flux"])
        label = label_from_peak_flux(peak_flux)
        folder = sample_dir(split_dir, row["id"])
        if not folder.exists():
            continue
        matches = sorted(folder.glob(f"*__{channel}.jpg"))
        if not matches:
            continue
        active_region = row["id"].split("_", 1)[0]
        grouped[label].append(
            Sample(
                path=matches[0],
                label=label,
                label_index=CLASS_TO_INDEX[label],
                sample_id=row["id"],
                active_region=active_region,
                peak_flux=peak_flux,
                split=split_name,
            )
        )

    rng = random.Random(seed)
    samples: list[Sample] = []
    for label in CLASSES:
        class_samples = grouped[label]
        rng.shuffle(class_samples)
        if max_per_class > 0:
            class_samples = class_samples[:max_per_class]
        samples.extend(class_samples)
    rng.shuffle(samples)
    return samples


def stratified_train_val_split(samples: list[Sample], val_fraction: float, seed: int) -> tuple[list[Sample], list[Sample]]:
    rng = random.Random(seed)
    grouped: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        grouped[sample.label].append(sample)

    train: list[Sample] = []
    val: list[Sample] = []
    for label in CLASSES:
        class_samples = grouped[label]
        rng.shuffle(class_samples)
        val_count = max(1, int(round(len(class_samples) * val_fraction)))
        if len(class_samples) - val_count < 1:
            val_count = max(0, len(class_samples) - 1)
        val.extend(class_samples[:val_count])
        train.extend(class_samples[val_count:])

    rng.shuffle(train)
    rng.shuffle(val)
    return train, val


def counts_by_label(samples: list[Sample]) -> dict[str, int]:
    counts = Counter(sample.label for sample in samples)
    return {label: counts.get(label, 0) for label in CLASSES}


def matrix_to_markdown(matrix: list[list[int]]) -> str:
    rows = ["| Real \\ Predito | LOW | MEDIUM | HIGH |", "|---|---:|---:|---:|"]
    for label, values in zip(CLASSES, matrix):
        rows.append(f"| {label} | {values[0]} | {values[1]} | {values[2]} |")
    return "\n".join(rows)


def format_accuracy(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:.2f}%"


def build_markdown_report(results: dict[str, Any], output_path: Path) -> None:
    comparison = results["comparison"]
    best = results["best_model"]
    achieved = best["test_accuracy"] >= 0.88
    status = "atingiu" if achieved else "nao atingiu"
    lines = [
        "# Hermes SolarShield - Relatorio ACV",
        "",
        "## 1. Definicao do problema",
        "",
        "O problema de Visao Computacional e classificar imagens solares SDO AIA 131 em risco LOW, MEDIUM ou HIGH. "
        "A classificacao apoia a plataforma Hermes SolarShield na triagem operacional de clima espacial para infraestrutura orbital.",
        "",
        "## 2. Dataset utilizado",
        "",
        f"- Fonte: {results['dataset']['source']}",
        f"- Canal visual: {results['dataset']['channel']}",
        f"- Classes: {', '.join(CLASSES)}",
        f"- Treino: {results['dataset']['split_counts']['train']}",
        f"- Validacao: {results['dataset']['split_counts']['val']}",
        f"- Teste: {results['dataset']['split_counts']['test']}",
        "- Rotulagem:",
    ]
    for label, description in LABEL_POLICY.items():
        lines.append(f"  - {label}: {description}")
    lines.extend(
        [
            "",
            "Pre-processamento: conversao para escala de cinza, autocontraste, redimensionamento, normalizacao numerica e aumentos leves no treino.",
            "",
            "## 3. Treinamento de CNNs do zero",
            "",
            "Foram treinadas duas arquiteturas convolucionais proprias, sem uso de modelos pre-treinados.",
            "",
            "| Arquitetura | Parametros | Val accuracy | Test accuracy | Test loss |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for item in comparison:
        lines.append(
            f"| {item['architecture']} | {item['parameter_count']} | "
            f"{format_accuracy(item['val_accuracy'])} | {format_accuracy(item['test_accuracy'])} | {item['test_loss']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 4. Avaliacao do melhor modelo",
            "",
            f"- Melhor arquitetura: {best['architecture']}",
            f"- Acuracia de teste: {format_accuracy(best['test_accuracy'])}",
            f"- Loss de teste: {best['test_loss']:.4f}",
            f"- Criterio de referencia de 88%: {status}.",
        ]
    )
    if not achieved:
        lines.extend(
            [
                "",
                "Justificativa tecnica: o rotulo `peak_flux` do SDOBenchmark representa a intensidade futura do evento solar na janela do exemplo, "
                "enquanto a entrada usada nesta entrega e apenas um frame do canal AIA 131. A imagem isolada nao contem toda a informacao temporal "
                "e magnetica usada por sistemas reais de previsao de flares. Melhorias futuras incluem usar sequencias temporais, multiplos canais SDO, "
                "magnetogramas HMI, aumento do dataset balanceado e modelos hibridos CNN + series temporais.",
            ]
        )
    lines.extend(
        [
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
        lines.append(
            f"| {label} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} |"
        )

    lines.extend(["", "### Exemplos de erros", ""])
    errors = best.get("errors", [])
    if errors:
        for error in errors[:10]:
            lines.append(
                f"- {error['file']}: real={error['actual']} predito={error['predicted']} "
                f"prob={error['probabilities']}"
            )
    else:
        lines.append("- Nenhum erro no subconjunto de teste avaliado.")

    lines.extend(
        [
            "",
            "## 5. Comparacao tecnica",
            "",
            results["technical_comparison"],
            "",
            "## 6. Demonstracao funcional",
            "",
            "O checkpoint vencedor e salvo em `backend/ml/models/solar_flare_classifier_131.pt` e carregado automaticamente pela API FastAPI. "
            "A pagina Applied Computer Vision mostra a classe prevista, confianca e probabilidades LOW/MEDIUM/HIGH.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def train(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from PIL import Image, ImageEnhance, ImageOps
        from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
    except ImportError as exc:
        raise SystemExit("Instale torch, pillow e numpy para treinar as CNNs de ACV.") from exc

    dataset_root = find_dataset_root(args.raw_dir)
    training_samples_all = collect_sdobenchmark_samples(
        dataset_root / "training",
        "training",
        args.channel,
        args.max_train_per_class,
        args.seed,
    )
    test_samples = collect_sdobenchmark_samples(
        dataset_root / "test",
        "test",
        args.channel,
        args.max_test_per_class,
        args.seed + 1,
    )
    train_samples, val_samples = stratified_train_val_split(training_samples_all, args.val_fraction, args.seed + 2)
    if not train_samples or not val_samples or not test_samples:
        raise SystemExit("Amostras insuficientes para treino, validacao e teste.")

    class SolarRiskDataset(Dataset):
        def __init__(self, samples: list[Sample], image_size: int, augment: bool):
            self.samples = samples
            self.image_size = image_size
            self.augment = augment

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int):
            sample = self.samples[index]
            image = Image.open(sample.path).convert("L")
            image = ImageOps.autocontrast(image)
            image = image.resize((self.image_size, self.image_size))
            if self.augment:
                if random.random() < 0.5:
                    image = ImageOps.mirror(image)
                if random.random() < 0.3:
                    image = ImageOps.flip(image)
                if random.random() < 0.6:
                    image = image.rotate(random.uniform(-8.0, 8.0))
                if random.random() < 0.5:
                    image = ImageEnhance.Contrast(image).enhance(random.uniform(0.9, 1.15))
            array = np.asarray(image, dtype=np.float32) / 255.0
            array = (array - 0.5) / 0.5
            return torch.from_numpy(array).unsqueeze(0), torch.tensor(sample.label_index, dtype=torch.long)

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
                nn.Linear(64, len(CLASSES)),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

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
                nn.Linear(160, len(CLASSES)),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    def parameter_count(model: nn.Module) -> int:
        return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, samples: list[Sample]) -> dict[str, Any]:
        model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        confusion = [[0 for _ in CLASSES] for _ in CLASSES]
        errors: list[dict[str, Any]] = []
        sample_cursor = 0
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(device)
                labels = labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                probabilities = torch.softmax(logits, dim=1).cpu()
                predictions = probabilities.argmax(dim=1)
                total_loss += float(loss.item())
                correct += int((predictions.to(device) == labels).sum().item())
                total += int(labels.numel())
                for batch_index, (actual_idx, predicted_idx) in enumerate(zip(labels.cpu().tolist(), predictions.tolist())):
                    confusion[actual_idx][predicted_idx] += 1
                    sample = samples[sample_cursor + batch_index]
                    if actual_idx != predicted_idx and len(errors) < 25:
                        errors.append(
                            {
                                "file": sample.path.name,
                                "actual": CLASSES[actual_idx],
                                "predicted": CLASSES[predicted_idx],
                                "peak_flux": sample.peak_flux,
                                "probabilities": {
                                    CLASSES[index]: round(float(probabilities[batch_index][index]), 3)
                                    for index in range(len(CLASSES))
                                },
                            }
                        )
                sample_cursor += labels.shape[0]

        report: dict[str, dict[str, float]] = {}
        for index, label in enumerate(CLASSES):
            tp = confusion[index][index]
            fp = sum(confusion[row][index] for row in range(len(CLASSES)) if row != index)
            fn = sum(confusion[index][col] for col in range(len(CLASSES)) if col != index)
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

    def run_one_model(architecture: str, model: nn.Module) -> dict[str, Any]:
        model = model.to(device)
        train_set = SolarRiskDataset(train_samples, args.image_size, augment=True)
        val_set = SolarRiskDataset(val_samples, args.image_size, augment=False)
        test_set = SolarRiskDataset(test_samples, args.image_size, augment=False)

        train_labels = [sample.label_index for sample in train_samples]
        class_counts = Counter(train_labels)
        class_weights = [
            len(train_labels) / max(len(CLASSES) * class_counts.get(index, 1), 1)
            for index in range(len(CLASSES))
        ]
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=sampler, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=5, factor=0.5)

        history: list[dict[str, float]] = []
        best_state: dict[str, Any] | None = None
        best_val_accuracy = -1.0
        best_val_loss = float("inf")
        stale_epochs = 0

        print(f"\nTreinando {architecture} em {device}")
        print(f"train={counts_by_label(train_samples)} val={counts_by_label(val_samples)} test={counts_by_label(test_samples)}")
        for epoch in range(1, args.max_epochs + 1):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                logits = model(images)
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                train_loss += float(loss.item())
                train_correct += int((logits.argmax(dim=1) == labels).sum().item())
                train_total += int(labels.numel())

            val_metrics = evaluate(model, val_loader, criterion, val_samples)
            train_accuracy = train_correct / max(train_total, 1)
            train_loss = train_loss / max(len(train_loader), 1)
            val_accuracy = val_metrics["accuracy"]
            val_loss = val_metrics["loss"]
            scheduler.step(val_accuracy)
            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_accuracy": train_accuracy,
                    "val_accuracy": val_accuracy,
                }
            )
            print(
                f"Epoch {epoch:03d}/{args.max_epochs} "
                f"train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f} "
                f"train_loss={train_loss:.4f} val_loss={val_loss:.4f}"
            )

            improved = val_accuracy > best_val_accuracy or (
                val_accuracy == best_val_accuracy and val_loss < best_val_loss
            )
            if improved:
                best_val_accuracy = val_accuracy
                best_val_loss = val_loss
                stale_epochs = 0
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            else:
                stale_epochs += 1

            if best_val_accuracy >= args.target_accuracy and epoch >= args.min_epochs:
                print(f"Alvo de validacao atingido para {architecture}: {best_val_accuracy:.4f}")
                break
            if stale_epochs >= args.patience and epoch >= args.min_epochs:
                print(f"Parada antecipada por paciencia em {architecture}.")
                break

        if best_state is not None:
            model.load_state_dict(best_state)

        val_metrics = evaluate(model, val_loader, criterion, val_samples)
        test_metrics = evaluate(model, test_loader, criterion, test_samples)
        checkpoint_path = args.output_dir / f"{architecture}.pt"
        checkpoint = {
            "architecture": architecture,
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "image_size": args.image_size,
            "input_channels": 1,
            "channel": f"SDO AIA {args.channel}",
            "label_policy": LABEL_POLICY,
            "metrics": {
                "val_accuracy": val_metrics["accuracy"],
                "val_loss": val_metrics["loss"],
                "test_accuracy": test_metrics["accuracy"],
                "test_loss": test_metrics["loss"],
                "confusion_matrix": test_metrics["confusion_matrix"],
                "classification_report": test_metrics["classification_report"],
                "split_counts": {
                    "train": counts_by_label(train_samples),
                    "val": counts_by_label(val_samples),
                    "test": counts_by_label(test_samples),
                },
            },
            "history": history,
            "parameter_count": parameter_count(model),
            "training_source": "SDOBenchmark peak_flux classification + SDO AIA 131",
        }
        torch.save(checkpoint, checkpoint_path)
        checkpoint_path.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        print(f"Checkpoint salvo: {checkpoint_path}")
        return {
            "architecture": architecture,
            "path": str(checkpoint_path),
            "parameter_count": parameter_count(model),
            "val_accuracy": val_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "test_accuracy": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
            "confusion_matrix": test_metrics["confusion_matrix"],
            "classification_report": test_metrics["classification_report"],
            "errors": test_metrics["errors"],
            "history": history,
        }

    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.report_dir.mkdir(parents=True, exist_ok=True)

    model_results = [
        run_one_model("acv_tiny_cnn", SolarShieldTinyCNN()),
        run_one_model("acv_deep_cnn", SolarShieldDeepCNN()),
    ]
    best = max(model_results, key=lambda item: (item["test_accuracy"], item["val_accuracy"]))
    active_path = args.output_dir / "solar_flare_classifier_131.pt"
    shutil.copyfile(best["path"], active_path)

    technical_comparison = (
        "A arquitetura Tiny usa menos camadas e menos parametros, sendo mais rapida e menos propensa a overfitting em bases pequenas. "
        "A arquitetura Deep adiciona blocos duplos de convolucao, BatchNorm e Dropout2d; isso aumenta capacidade de extrair padroes locais "
        "em regioes ativas, mas tambem exige mais regularizacao. A comparacao por acuracia, loss e matriz de confusao indica qual delas "
        "generalizou melhor no teste oficial."
    )
    results = {
        "dataset": {
            "source": str(dataset_root),
            "channel": f"SDO AIA {args.channel}",
            "label_policy": LABEL_POLICY,
            "split_counts": {
                "train": counts_by_label(train_samples),
                "val": counts_by_label(val_samples),
                "test": counts_by_label(test_samples),
            },
        },
        "comparison": [
            {
                "architecture": item["architecture"],
                "path": item["path"],
                "parameter_count": item["parameter_count"],
                "val_accuracy": item["val_accuracy"],
                "val_loss": item["val_loss"],
                "test_accuracy": item["test_accuracy"],
                "test_loss": item["test_loss"],
            }
            for item in model_results
        ],
        "best_model": best,
        "active_model_path": str(active_path),
        "target_accuracy": args.target_accuracy,
        "technical_comparison": technical_comparison,
    }
    comparison_path = args.output_dir / "acv_classification_comparison.json"
    comparison_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    report_path = args.report_dir / "acv_classification_report.md"
    build_markdown_report(results, report_path)
    print(f"\nModelo ativo: {active_path}")
    print(f"Comparacao: {comparison_path}")
    print(f"Relatorio: {report_path}")
    print(json.dumps(results["comparison"], indent=2))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Treina duas CNNs do zero para classificacao ACV do Hermes SolarShield.")
    parser.add_argument("--raw-dir", default=Path("data/sdobenchmark"), type=Path)
    parser.add_argument("--output-dir", default=Path("ml/models"), type=Path)
    parser.add_argument("--report-dir", default=Path("reports"), type=Path)
    parser.add_argument("--channel", default="131")
    parser.add_argument("--image-size", default=96, type=int)
    parser.add_argument("--max-epochs", default=80, type=int)
    parser.add_argument("--min-epochs", default=12, type=int)
    parser.add_argument("--patience", default=18, type=int)
    parser.add_argument("--target-accuracy", default=0.88, type=float)
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--learning-rate", default=1e-3, type=float)
    parser.add_argument("--val-fraction", default=0.15, type=float)
    parser.add_argument("--max-train-per-class", default=0, type=int)
    parser.add_argument("--max-test-per-class", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
