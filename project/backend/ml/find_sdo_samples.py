from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from urllib.parse import urljoin

import requests

from model import load_model, predict_flare_risk


def list_day_images(year: int, month: int, day: int, channels: list[str]) -> list[str]:
    base = f"https://sdo.gsfc.nasa.gov/assets/img/browse/{year:04d}/{month:02d}/{day:02d}/"
    response = requests.get(base, headers={"User-Agent": "HermesSolarShield/1.0"}, timeout=45)
    response.raise_for_status()
    links = re.findall(r'href="([^"]+_1024_(?:' + "|".join(map(re.escape, channels)) + r')\.jpg)"', response.text)
    return [urljoin(base, link) for link in links]


def main() -> None:
    parser = argparse.ArgumentParser(description="Encontra 3 imagens SDO reais para LOW, MEDIUM e HIGH.")
    parser.add_argument("--output-dir", default="sample_data/sdo_class_samples", type=Path)
    parser.add_argument("--max-per-class", default=3, type=int)
    parser.add_argument("--channels", default="0131,0171,0193,0211,0304,0335,HMIIF,HMIB,HMIIC")
    args = parser.parse_args()

    dates = [
        (2026, 5, 1), (2026, 5, 5), (2026, 5, 10), (2026, 5, 15), (2026, 5, 20),
        (2026, 5, 25), (2026, 5, 31), (2026, 6, 1), (2026, 6, 2),
        (2026, 4, 1), (2026, 4, 10), (2026, 4, 20), (2026, 4, 30),
        (2026, 3, 1), (2026, 3, 15), (2026, 3, 30),
    ]
    channels = [channel.strip() for channel in args.channels.split(",") if channel.strip()]
    found: dict[str, list[dict[str, str | float]]] = {"LOW": [], "MEDIUM": [], "HIGH": []}
    model = load_model()
    tmp_dir = args.output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for year, month, day in dates:
        try:
            urls = list_day_images(year, month, day, channels)
        except Exception as exc:
            print(f"Falha ao listar {year}-{month:02d}-{day:02d}: {exc}")
            continue

        # Sample every few frames so we cover the day without downloading thousands of files.
        step = max(len(urls) // 36, 1)
        for url in urls[::step]:
            if all(len(items) >= args.max_per_class for items in found.values()):
                break
            filename = url.rsplit("/", 1)[-1]
            tmp_path = tmp_dir / filename
            try:
                if not tmp_path.exists():
                    image_response = requests.get(url, headers={"User-Agent": "HermesSolarShield/1.0"}, timeout=45)
                    image_response.raise_for_status()
                    tmp_path.write_bytes(image_response.content)
                result = predict_flare_risk(tmp_path.read_bytes(), model)
                risk = result["risk_level"]
                if len(found[risk]) >= args.max_per_class:
                    continue
                day_key = filename[:8]
                if any(str(existing["file"]).find(day_key) >= 0 for existing in found[risk]):
                    continue
                target = args.output_dir / risk.lower()
                target.mkdir(parents=True, exist_ok=True)
                final_path = target / filename
                shutil.copy2(tmp_path, final_path)
                item = {
                    "risk": risk,
                    "score": result["score"],
                    "confidence": result["confidence"],
                    "url": url,
                    "file": str(final_path),
                }
                found[risk].append(item)
                print(f"{risk}: {result['score']} {filename}")
            except Exception as exc:
                print(f"Falha ao avaliar {url}: {exc}")
        if all(len(items) >= args.max_per_class for items in found.values()):
            break

    for risk, items in found.items():
        print(f"\n{risk} ({len(items)} imagens)")
        for item in items:
            print(f"- score={item['score']} confidence={item['confidence']} file={item['file']}")
            print(f"  url={item['url']}")

    shutil.rmtree(tmp_dir, ignore_errors=True)
    if any(len(items) < args.max_per_class for items in found.values()):
        raise SystemExit("Nao encontrei todas as classes com os parametros atuais.")


if __name__ == "__main__":
    main()
