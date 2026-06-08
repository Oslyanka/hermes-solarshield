from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

CDX_URL = "https://web.archive.org/cdx"
ARCHIVE_RAW = "https://web.archive.org/web/{timestamp}id_/https://www.spaceweatherlive.com/en/solar-activity/solar-flares.html"
SPACEWEATHERLIVE_FLARES = "https://www.spaceweatherlive.com/en/solar-activity/solar-flares.html"
SDO_BROWSE = "https://sdo.gsfc.nasa.gov/assets/img/browse"


def _clean_text(html: str) -> str:
    html = re.sub(r"<script[\s\S]*?</script>|<style[\s\S]*?</style>", " ", html, flags=re.I)
    text = re.sub(r"<[^>]+>", " ", html)
    return re.sub(r"\s+", " ", text)


def _snapshot_datetime(timestamp: str) -> dt.datetime:
    return dt.datetime.strptime(timestamp, "%Y%m%d%H%M%S").replace(tzinfo=dt.timezone.utc)


def _extract_probabilities(html: str) -> dict[str, int] | None:
    text = _clean_text(html)
    values = {
        flare_class.upper(): int(value)
        for flare_class, value in re.findall(r"([CMX])-class\s+solar flare\s+(\d+)%", text, flags=re.I)
    }
    if all(key in values for key in ("C", "M", "X")):
        return values
    return None


def fetch_snapshots(start: str, end: str, limit: int) -> list[str]:
    response = requests.get(
        CDX_URL,
        params={
            "url": SPACEWEATHERLIVE_FLARES,
            "from": start,
            "to": end,
            "output": "json",
            "fl": "timestamp,statuscode,mimetype",
            "filter": "statuscode:200",
            "limit": limit,
        },
        headers={"User-Agent": "HermesSolarShield/1.0 academic training"},
        timeout=90,
    )
    response.raise_for_status()
    rows = response.json()
    return [row[0] for row in rows[1:]]


def _list_sdo_0131(day: dt.date) -> list[str]:
    base = f"{SDO_BROWSE}/{day.year:04d}/{day.month:02d}/{day.day:02d}/"
    response = requests.get(base, headers={"User-Agent": "HermesSolarShield/1.0"}, timeout=45)
    response.raise_for_status()
    names = re.findall(r'href="([^"]+_1024_0131\.jpg)"', response.text)
    return [urljoin(base, name) for name in names]


def _url_time(url: str) -> dt.datetime:
    name = url.rsplit("/", 1)[-1]
    return dt.datetime.strptime(name[:15], "%Y%m%d_%H%M%S").replace(tzinfo=dt.timezone.utc)


def _nearest_sdo_url(snapshot_time: dt.datetime) -> str | None:
    urls = _list_sdo_0131(snapshot_time.date())
    if not urls:
        return None
    return min(urls, key=lambda url: abs((_url_time(url) - snapshot_time).total_seconds()))


def build_dataset(output_dir: Path, start: str, end: str, limit: int, max_per_day: int) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    rows: list[dict] = []
    if manifest_path.exists():
        rows = json.loads(manifest_path.read_text(encoding="utf-8"))

    timestamps = fetch_snapshots(start, end, limit)
    seen_per_day: dict[str, int] = {}
    existing_images = {row["path"] for row in rows}
    for row in rows:
        day_key = row["date"].replace("-", "")
        seen_per_day[day_key] = seen_per_day.get(day_key, 0) + 1

    for timestamp in timestamps:
        snapshot_time = _snapshot_datetime(timestamp)
        day_key = snapshot_time.strftime("%Y%m%d")
        if seen_per_day.get(day_key, 0) >= max_per_day:
            continue
        try:
            page = requests.get(ARCHIVE_RAW.format(timestamp=timestamp), headers={"User-Agent": "HermesSolarShield/1.0"}, timeout=20)
            page.raise_for_status()
            probabilities = _extract_probabilities(page.text)
            if not probabilities:
                continue
            image_url = _nearest_sdo_url(snapshot_time)
            if not image_url:
                continue
            image_dir = output_dir / "images"
            image_dir.mkdir(exist_ok=True)
            image_path = image_dir / image_url.rsplit("/", 1)[-1]
            if str(image_path) in existing_images:
                seen_per_day[day_key] = seen_per_day.get(day_key, 0) + 1
                continue
            if not image_path.exists():
                image = requests.get(image_url, headers={"User-Agent": "HermesSolarShield/1.0"}, timeout=35)
                image.raise_for_status()
                image_path.write_bytes(image.content)
            row = {
                "timestamp": timestamp,
                "date": snapshot_time.date().isoformat(),
                "snapshot_url": ARCHIVE_RAW.format(timestamp=timestamp),
                "image_url": image_url,
                "path": str(image_path),
                "probabilities": probabilities,
                "targets": [probabilities["C"] / 100.0, probabilities["M"] / 100.0, probabilities["X"] / 100.0],
            }
            rows.append(row)
            existing_images.add(str(image_path))
            seen_per_day[day_key] = seen_per_day.get(day_key, 0) + 1
            print(f"{timestamp}: C={probabilities['C']} M={probabilities['M']} X={probabilities['X']} {image_path.name}", flush=True)
            manifest_path.write_text(json.dumps(sorted(rows, key=lambda item: item["date"]), indent=2), encoding="utf-8")
        except Exception as exc:
            print(f"Falha snapshot {timestamp}: {exc}", flush=True)

    rows = sorted(rows, key=lambda item: item["date"])
    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def train(
    manifest: list[dict],
    output: Path,
    max_epochs: int,
    target_mae: float,
    target_accuracy: float,
    tolerance: float,
    batch_size: int,
    protected_weight: int,
    protected_focus: float,
    use_protected_samples: bool,
    patience: int,
    seed: int,
) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, Subset, random_split

    class ForecastDataset(Dataset):
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            image = Image.open(row["path"]).convert("L").resize((128, 128))
            array = np.asarray(image, dtype=np.float32) / 255.0
            return torch.from_numpy(array).unsqueeze(0), torch.tensor(row["targets"], dtype=torch.float32)

    class ForecastCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
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

    if len(manifest) < 12:
        raise SystemExit("Poucos snapshots validos para treino.")
    torch.manual_seed(seed)
    np.random.seed(seed)
    dataset = ForecastDataset(manifest)
    protected_train = (
        [index for index, row in enumerate(manifest) if "web.archive.org" not in row.get("snapshot_url", "")]
        if use_protected_samples
        else []
    )
    remaining = [index for index in range(len(manifest)) if index not in protected_train]
    val_size = max(3, math.ceil(len(dataset) * 0.2))
    val_size = min(val_size, max(len(remaining) - 1, 1))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(remaining), generator=generator).tolist()
    val_indices = [remaining[index] for index in permutation[:val_size]]
    weighted_protected_train = [
        index
        for index in protected_train
        for _ in range(max(1, protected_weight))
    ]
    train_indices = weighted_protected_train + [remaining[index] for index in permutation[val_size:]]
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    protected_set = Subset(dataset, protected_train) if protected_train else None
    train_size = len(train_set)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    protected_loader = DataLoader(protected_set, batch_size=batch_size) if protected_set else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ForecastCNN().to(device)
    criterion = nn.SmoothL1Loss(beta=0.08)
    optimizer = torch.optim.AdamW(model.parameters(), lr=8e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", patience=5, factor=0.55)

    output.parent.mkdir(parents=True, exist_ok=True)
    best: dict = {}
    best_mae = 999.0
    best_accuracy = -1.0
    epochs_without_improvement = 0
    history = []
    print(f"Training Wayback forecast C/M/X on {device}: total={len(dataset)} train={train_size} val={val_size}", flush=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            output_raw = model(images)
            loss = criterion(torch.sigmoid(output_raw), targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        errors = []
        predictions = []
        actuals = []
        with torch.no_grad():
            for images, targets in val_loader:
                preds = torch.sigmoid(model(images.to(device))).cpu()
                errors.append(torch.abs(preds - targets))
                predictions.append(preds)
                actuals.append(targets)
        error_matrix = torch.cat(errors, dim=0)
        prediction_matrix = torch.cat(predictions, dim=0)
        target_matrix = torch.cat(actuals, dim=0)
        per_class_mae = error_matrix.mean(dim=0)
        mae = float(per_class_mae.mean().item())
        protected_mae = None
        protected_per_class_mae = None
        if protected_loader is not None:
            protected_errors = []
            with torch.no_grad():
                for images, targets in protected_loader:
                    preds = torch.sigmoid(model(images.to(device))).cpu()
                    protected_errors.append(torch.abs(preds - targets))
            protected_error_matrix = torch.cat(protected_errors, dim=0)
            protected_per_class_mae = protected_error_matrix.mean(dim=0)
            protected_mae = float(protected_per_class_mae.mean().item())
        selection_mae = mae
        if protected_mae is not None and protected_focus > 0:
            focus = max(0.0, min(protected_focus, 1.0))
            selection_mae = (mae * (1.0 - focus)) + (protected_mae * focus)
        rounded_exact_match = float(
            ((prediction_matrix * 100).round() == (target_matrix * 100).round()).float().mean().item()
        )
        strict_exact_match = float((error_matrix <= 0).float().mean().item())
        accuracy_within_tolerance = (
            rounded_exact_match if tolerance <= 0 else float((error_matrix <= tolerance).float().mean().item())
        )
        metrics = {
            "mae": mae,
            "mae_c": float(per_class_mae[0].item()),
            "mae_m": float(per_class_mae[1].item()),
            "mae_x": float(per_class_mae[2].item()),
            "accuracy_within_tolerance": accuracy_within_tolerance,
            "tolerance": tolerance,
            "rounded_exact_match": rounded_exact_match,
            "strict_exact_match": strict_exact_match,
            "train_loss": train_loss / max(len(train_loader), 1),
            "selection_mae": selection_mae,
        }
        if protected_mae is not None and protected_per_class_mae is not None:
            metrics.update(
                {
                    "protected_mae": protected_mae,
                    "protected_mae_c": float(protected_per_class_mae[0].item()),
                    "protected_mae_m": float(protected_per_class_mae[1].item()),
                    "protected_mae_x": float(protected_per_class_mae[2].item()),
                    "protected_focus": max(0.0, min(protected_focus, 1.0)),
                }
            )
        scheduler.step(mae)
        history.append({"epoch": epoch, **metrics})
        print(
            f"Epoch {epoch}/{max_epochs} mae={mae:.4f} "
            f"C={metrics['mae_c']:.4f} M={metrics['mae_m']:.4f} X={metrics['mae_x']:.4f} "
            f"protected={protected_mae if protected_mae is not None else 0:.4f} "
            f"select={selection_mae:.4f} acc_tol={metrics['accuracy_within_tolerance']:.3f} "
            f"exact_round={rounded_exact_match:.3f}",
            flush=True,
        )
        is_better = selection_mae < best_mae or (
            math.isclose(selection_mae, best_mae, rel_tol=0.0, abs_tol=1e-6)
            and metrics["accuracy_within_tolerance"] > best_accuracy
        )
        if is_better:
            best_mae = selection_mae
            best_accuracy = metrics["accuracy_within_tolerance"]
            best = metrics
            epochs_without_improvement = 0
            torch.save(
                {
                    "architecture": "forecast_cmx_cnn",
                    "model_state_dict": model.state_dict(),
                    "classes": ["C", "M", "X"],
                    "channel": "0131",
                    "metrics": metrics,
                    "training_source": "Wayback SpaceWeatherLive forecast + NASA SDO AIA 131",
                    "protected_training_samples": len(protected_train),
                    "protected_weight": max(1, protected_weight) if use_protected_samples else 1,
                    "protected_samples_enabled": use_protected_samples,
                },
                output,
            )
        else:
            epochs_without_improvement += 1
        output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
        if best_mae <= target_mae or metrics["accuracy_within_tolerance"] >= target_accuracy:
            break
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"Early stop: sem melhora por {patience} epocas.", flush=True)
            break

    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina forecast C/M/X usando Wayback SpaceWeatherLive + SDO AIA 131.")
    parser.add_argument("--start", default="20200901")
    parser.add_argument("--end", default=dt.date.today().strftime("%Y%m%d"))
    parser.add_argument("--dataset-dir", default="data/wayback_swl_0131", type=Path)
    parser.add_argument("--output", default="ml/models/solar_flare_cmx_131.pt", type=Path)
    parser.add_argument("--snapshot-limit", default=180, type=int)
    parser.add_argument("--max-per-day", default=1, type=int)
    parser.add_argument("--max-epochs", default=120, type=int)
    parser.add_argument("--target-mae", default=0.08, type=float)
    parser.add_argument("--target-accuracy", default=0.90, type=float)
    parser.add_argument("--tolerance", default=0.10, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--protected-weight", default=8, type=int)
    parser.add_argument("--protected-focus", default=0.0, type=float)
    parser.add_argument("--use-protected-samples", action="store_true")
    parser.add_argument("--patience", default=80, type=int)
    parser.add_argument("--seed", default=31, type=int)
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    args = parser.parse_args()

    manifest_path = args.dataset_dir / "manifest.json"
    if args.train_only and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = build_dataset(args.dataset_dir, args.start, args.end, args.snapshot_limit, args.max_per_day)
    if args.build_only:
        print(json.dumps({"samples": len(manifest), "manifest": str(args.dataset_dir / "manifest.json")}, indent=2))
        return
    print(
        json.dumps(
            train(
                manifest,
                args.output,
                args.max_epochs,
                args.target_mae,
                args.target_accuracy,
                args.tolerance,
                args.batch_size,
                args.protected_weight,
                args.protected_focus,
                args.use_protected_samples,
                args.patience,
                args.seed,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
