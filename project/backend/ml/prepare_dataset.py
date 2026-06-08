from __future__ import annotations

import argparse
import csv
import random
import shutil
from collections import defaultdict
from pathlib import Path


def _sample_dir(split_dir: Path, sample_id: str) -> Path:
    active_region, folder = sample_id.split("_", 1)
    return split_dir / active_region / folder


def _collect_images(split_dir: Path, threshold: float, channel: str) -> dict[str, list[Path]]:
    metadata_path = split_dir / "meta_data.csv"
    if not metadata_path.exists():
        raise SystemExit(f"Metadados nao encontrados: {metadata_path}")

    grouped: dict[str, list[Path]] = {"high": [], "low": []}
    with metadata_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            label = "high" if float(row["peak_flux"]) >= threshold else "low"
            folder = _sample_dir(split_dir, row["id"])
            if not folder.exists():
                continue
            pattern = "*.jpg" if channel.lower() == "all" else f"*__{channel}.jpg"
            grouped[label].extend(sorted(folder.glob(pattern)))
    return grouped


def _copy_split(paths: list[Path], target: Path, limit: int | None) -> int:
    target.mkdir(parents=True, exist_ok=True)
    selected = paths[:limit] if limit else paths
    for index, source in enumerate(selected):
        shutil.copy2(source, target / f"{source.parent.name}_{index:05d}.jpg")
    return len(selected)


def prepare_sdobenchmark(
    raw_dir: Path,
    output_dir: Path,
    threshold: float = 1e-6,
    channel: str = "131",
    val_ratio: float = 0.2,
    max_images_per_class: int = 3000,
    seed: int = 42,
) -> None:
    dataset_root = raw_dir / "SDOBenchmark_example" if (raw_dir / "SDOBenchmark_example").exists() else raw_dir
    training_dir = dataset_root / "training"
    test_dir = dataset_root / "test"
    if not training_dir.exists():
        raise SystemExit(f"Pasta de treinamento nao encontrada: {training_dir}")

    random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_source = _collect_images(training_dir, threshold, channel)
    test_source = _collect_images(test_dir, threshold, channel) if test_dir.exists() else {"high": [], "low": []}

    summary = {}
    for label, paths in train_source.items():
        random.shuffle(paths)
        if max_images_per_class:
            paths = paths[:max_images_per_class]
        split_index = max(1, int(len(paths) * (1 - val_ratio)))
        train_paths = paths[:split_index]
        val_paths = paths[split_index:] or paths[-1:]
        summary[f"train/{label}"] = _copy_split(train_paths, output_dir / "train" / label, None)
        summary[f"val/{label}"] = _copy_split(val_paths, output_dir / "val" / label, None)

    for label, paths in test_source.items():
        random.shuffle(paths)
        limit = max(1, max_images_per_class // 4) if max_images_per_class else None
        summary[f"test/{label}"] = _copy_split(paths, output_dir / "test" / label, limit)

    print("Dataset SDOBenchmark processado")
    print(f"Origem: {dataset_root}")
    print(f"Destino: {output_dir}")
    print(f"Canal SDO: {channel}")
    print(f"Limiar high: peak_flux >= {threshold}")
    for key in sorted(summary):
        print(f"{key}: {summary[key]} imagens")

    missing = [key for key, count in summary.items() if key.startswith(("train", "val")) and count == 0]
    if missing:
        raise SystemExit(f"Split invalido, sem imagens em: {', '.join(missing)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepara o SDOBenchmark para classificacao low/high.")
    parser.add_argument("--raw-dir", default="data/sdobenchmark", type=Path)
    parser.add_argument("--output-dir", default="data/processed", type=Path)
    parser.add_argument("--threshold", default=1e-6, type=float)
    parser.add_argument("--channel", default="131")
    parser.add_argument("--val-ratio", default=0.2, type=float)
    parser.add_argument("--max-images-per-class", default=3000, type=int)
    parser.add_argument("--seed", default=42, type=int)
    args = parser.parse_args()
    prepare_sdobenchmark(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        channel=args.channel,
        val_ratio=args.val_ratio,
        max_images_per_class=args.max_images_per_class,
        seed=args.seed,
    )
