from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

THRESHOLDS = {"C": 1e-6, "M": 1e-5, "X": 1e-4}
DONKI_URL = "https://kauai.ccmc.gsfc.nasa.gov/DONKI/WS/get/FLR"
SDO_BROWSE = "https://sdo.gsfc.nasa.gov/assets/img/browse"


def _parse_flare_flux(class_type: str) -> float:
    match = re.match(r"([BCMX])\s*([0-9.]+)", class_type.strip(), flags=re.I)
    if not match:
        return 0.0
    letter, value = match.groups()
    multiplier = {"B": 1e-7, "C": 1e-6, "M": 1e-5, "X": 1e-4}[letter.upper()]
    return float(value) * multiplier


def _targets_from_class(class_type: str) -> list[float]:
    flux = _parse_flare_flux(class_type)
    return [1.0 if flux >= threshold else 0.0 for threshold in THRESHOLDS.values()]


def _class_bucket(class_type: str) -> str:
    return class_type.strip()[0].upper() if class_type else "B"


def _fetch_donki_events(start: dt.date, end: dt.date) -> list[dict]:
    response = requests.get(
        DONKI_URL,
        params={"startDate": start.isoformat(), "endDate": end.isoformat()},
        headers={"User-Agent": "HermesSolarShield/1.0 academic training"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def _list_sdo_0131(day: dt.date) -> list[str]:
    base = f"{SDO_BROWSE}/{day.year:04d}/{day.month:02d}/{day.day:02d}/"
    response = requests.get(base, headers={"User-Agent": "HermesSolarShield/1.0"}, timeout=45)
    response.raise_for_status()
    names = re.findall(r'href="([^"]+_1024_0131\.jpg)"', response.text)
    return [urljoin(base, name) for name in names]


def _url_time(url: str) -> dt.datetime:
    name = url.rsplit("/", 1)[-1]
    return dt.datetime.strptime(name[:15], "%Y%m%d_%H%M%S").replace(tzinfo=dt.timezone.utc)


def _nearest_sdo_url(peak_time: str) -> str | None:
    peak = dt.datetime.fromisoformat(peak_time.replace("Z", "+00:00"))
    urls = _list_sdo_0131(peak.date())
    if not urls:
        return None
    return min(urls, key=lambda url: abs((_url_time(url) - peak).total_seconds()))


def build_dataset(output_dir: Path, start: dt.date, end: dt.date, max_per_class: int) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    existing_rows: list[dict] = []
    folder_targets = {"c": [1.0, 0.0, 0.0], "m": [1.0, 1.0, 0.0], "x": [1.0, 1.0, 1.0]}
    for folder, targets in folder_targets.items():
        for path in sorted((output_dir / folder).glob("*_1024_0131.jpg")):
            existing_rows.append({"path": str(path), "url": "", "class_type": folder.upper(), "peak_time": "", "active_region": None, "targets": targets})
    if sum(1 for row in existing_rows if row["class_type"] == "X") >= max(8, max_per_class // 3):
        low_root = Path("sample_data/sdo_0131_examples/low")
        for path in sorted(low_root.glob("*_1024_0131.jpg")):
            existing_rows.append({"path": str(path), "url": "", "class_type": "NONE", "peak_time": "", "active_region": None, "targets": [0.0, 0.0, 0.0]})
        manifest_path.write_text(json.dumps(existing_rows, indent=2), encoding="utf-8")
        return existing_rows

    events = _fetch_donki_events(start, end)
    selected: dict[str, list[dict]] = {"C": [], "M": [], "X": []}
    for event in sorted(events, key=lambda item: item.get("peakTime", "")):
        class_type = event.get("classType") or ""
        bucket = _class_bucket(class_type)
        if bucket not in selected or len(selected[bucket]) >= max_per_class:
            continue
        if not event.get("peakTime"):
            continue
        selected[bucket].append(event)
        if all(len(items) >= max_per_class for items in selected.values()):
            break

    rows: list[dict] = []
    for bucket, bucket_events in selected.items():
        folder = output_dir / bucket.lower()
        folder.mkdir(parents=True, exist_ok=True)
        for event in bucket_events:
            try:
                url = _nearest_sdo_url(event["peakTime"])
                if not url:
                    continue
                filename = url.rsplit("/", 1)[-1]
                image_path = folder / filename
                if not image_path.exists():
                    image = requests.get(url, headers={"User-Agent": "HermesSolarShield/1.0"}, timeout=60)
                    image.raise_for_status()
                    image_path.write_bytes(image.content)
                rows.append(
                    {
                        "path": str(image_path),
                        "url": url,
                        "class_type": event["classType"],
                        "peak_time": event["peakTime"],
                        "active_region": event.get("activeRegionNum"),
                        "targets": _targets_from_class(event["classType"]),
                    }
                )
                print(f"{bucket}: {event['classType']} {filename}", flush=True)
            except Exception as exc:
                print(f"Falha em {event.get('classType')} {event.get('peakTime')}: {exc}", flush=True)

    # Add non-event/low-risk examples already curated for the UI.
    low_root = Path("sample_data/sdo_0131_examples/low")
    if low_root.exists():
        for path in sorted(low_root.glob("*_1024_0131.jpg")):
            rows.append({"path": str(path), "url": "", "class_type": "NONE", "peak_time": "", "active_region": None, "targets": [0.0, 0.0, 0.0]})

    manifest_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return rows


def train(manifest: list[dict], output: Path, max_epochs: int, target_accuracy: float, batch_size: int) -> dict:
    import numpy as np
    import torch
    import torch.nn as nn
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, random_split

    class FlareDataset(Dataset):
        def __init__(self, rows: list[dict]):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            row = self.rows[index]
            image = Image.open(row["path"]).convert("L").resize((128, 128))
            array = np.asarray(image, dtype=np.float32) / 255.0
            return torch.from_numpy(array).unsqueeze(0), torch.tensor(row["targets"], dtype=torch.float32)

    class CMXCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 24, 3, padding=1), nn.BatchNorm2d(24), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(24, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(48, 96, 3, padding=1), nn.BatchNorm2d(96), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(96, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
                nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Linear(128 * 4 * 4, 128), nn.ReLU(), nn.Dropout(0.35), nn.Linear(128, 3))

        def forward(self, x):
            return self.head(self.features(x))

    dataset = FlareDataset(manifest)
    val_size = max(1, math.ceil(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(22))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CMXCNN().to(device)

    target_matrix = torch.tensor([row["targets"] for row in manifest], dtype=torch.float32)
    positives = target_matrix.sum(dim=0)
    negatives = len(manifest) - positives
    pos_weight = (negatives / torch.clamp(positives, min=1)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)

    output.parent.mkdir(parents=True, exist_ok=True)
    best: dict = {}
    history = []
    best_macro = 0.0
    print(f"Treinando DONKI C/M/X em {device}: total={len(dataset)} train={train_size} val={val_size}", flush=True)
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())

        model.eval()
        correct = torch.zeros(3)
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                logits = model(images.to(device))
                predictions = (torch.sigmoid(logits).cpu() >= 0.5).float()
                correct += (predictions == targets).sum(dim=0)
                total += targets.shape[0]
        per_class = correct / max(total, 1)
        macro = float(per_class.mean().item())
        metrics = {"macro_accuracy": macro, "c_accuracy": float(per_class[0]), "m_accuracy": float(per_class[1]), "x_accuracy": float(per_class[2]), "train_loss": train_loss / max(len(train_loader), 1)}
        history.append({"epoch": epoch, **metrics})
        print(f"Epoch {epoch}/{max_epochs} macro={macro:.4f} C={metrics['c_accuracy']:.4f} M={metrics['m_accuracy']:.4f} X={metrics['x_accuracy']:.4f}", flush=True)
        if macro > best_macro:
            best_macro = macro
            best = metrics
            torch.save({"architecture": "cmx_cnn", "model_state_dict": model.state_dict(), "classes": ["C", "M", "X"], "thresholds": THRESHOLDS, "channel": "0131", "metrics": metrics, "training_source": "NASA DONKI FLR + SDO AIA 131"}, output)
        if best_macro >= target_accuracy:
            break
    output.with_suffix(".history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Treina C/M/X usando NASA DONKI FLR + SDO AIA 131.")
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default=dt.date.today().isoformat())
    parser.add_argument("--dataset-dir", default="data/donki_0131", type=Path)
    parser.add_argument("--output", default="ml/models/solar_flare_cmx_131.pt", type=Path)
    parser.add_argument("--max-per-class", default=45, type=int)
    parser.add_argument("--max-epochs", default=80, type=int)
    parser.add_argument("--target-accuracy", default=0.90, type=float)
    parser.add_argument("--batch-size", default=32, type=int)
    args = parser.parse_args()
    manifest = build_dataset(args.dataset_dir, dt.date.fromisoformat(args.start), dt.date.fromisoformat(args.end), args.max_per_class)
    print(json.dumps(train(manifest, args.output, args.max_epochs, args.target_accuracy, args.batch_size), indent=2))


if __name__ == "__main__":
    main()
