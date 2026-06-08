from __future__ import annotations

import argparse
import re
from pathlib import Path
from urllib.parse import urljoin

import requests

from model import load_model, predict_flare_risk

SPACEWEATHERLIVE_URL = "https://www.spaceweatherlive.com/en/solar-activity.html"
NASA_SDO_FALLBACK = "https://sdo.gsfc.nasa.gov/assets/img/latest/latest_1024_0131.jpg"


def fetch_flare_probabilities(page_url: str = SPACEWEATHERLIVE_URL) -> dict[str, int]:
    headers = {"User-Agent": "HermesSolarShield/1.0 academic test"}
    response = requests.get(page_url, headers=headers, timeout=30)
    response.raise_for_status()
    text = re.sub(r"<[^>]+>", " ", response.text)
    text = re.sub(r"\s+", " ", text)
    probabilities = {}
    for flare_class, value in re.findall(r"([CMX])-class\s+solar flare\s+(\d+)%", text, flags=re.IGNORECASE):
        probabilities[flare_class.upper()] = int(value)
    return probabilities


def _candidate_images(html: str, page_url: str) -> list[str]:
    urls = re.findall(r"""(?:src|data-src|href)=["']([^"']+\.(?:jpg|jpeg|png))["']""", html, flags=re.IGNORECASE)
    absolute = [urljoin(page_url, item) for item in urls]
    preferred = [
        url for url in absolute
        if any(token in url.lower() for token in ("sdo", "aia", "solar", "sun", "latest_1024"))
    ]

    def score(url: str) -> int:
        lower = url.lower()
        points = 0
        if "131" in lower or "0131" in lower:
            points += 8
        if "aia" in lower:
            points += 4
        if "hmi" in lower:
            points -= 4
        if "sdo" in lower:
            points += 2
        return points

    return sorted(preferred or absolute, key=score, reverse=True)


def download_spaceweatherlive_image(output: Path, page_url: str = SPACEWEATHERLIVE_URL) -> Path:
    headers = {"User-Agent": "HermesSolarShield/1.0 academic test"}
    response = requests.get(page_url, headers=headers, timeout=30)
    response.raise_for_status()
    candidates = _candidate_images(response.text, page_url)
    candidates.append(NASA_SDO_FALLBACK)

    last_error: Exception | None = None
    for image_url in candidates:
        try:
            image_response = requests.get(image_url, headers=headers, timeout=30)
            image_response.raise_for_status()
            content_type = image_response.headers.get("content-type", "")
            if "image" not in content_type and len(image_response.content) < 10_000:
                continue
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(image_response.content)
            print(f"Imagem de teste baixada: {image_url}")
            return output
        except Exception as exc:
            last_error = exc

    raise RuntimeError(f"Nao foi possivel baixar uma imagem solar. Ultimo erro: {last_error}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Testa o modelo com imagem solar do SpaceWeatherLive.")
    parser.add_argument("--output-image", default="sample_data/spaceweatherlive_latest.jpg", type=Path)
    parser.add_argument("--page-url", default=SPACEWEATHERLIVE_URL)
    args = parser.parse_args()

    image_path = download_spaceweatherlive_image(args.output_image, args.page_url)
    model = load_model()
    result = predict_flare_risk(image_path.read_bytes(), model, forecast_calibration=True)
    swl = fetch_flare_probabilities(args.page_url)
    model_percent = round(float(result["score"]) * 100, 1)
    target_percent = swl.get("C")
    print("Resultado Hermes SolarShield")
    print(f"Imagem: {image_path}")
    print(f"Modo: {result['mode']}")
    print(f"Risco: {result['risk_level']}")
    print(f"Score: {result['score']}")
    print(f"Confianca: {result['confidence']}")
    print(f"Modelo flare C+ (%): {model_percent}")
    print(f"Modelo probabilidades C/M/X: {result.get('flare_probabilities')}")
    print(f"SpaceWeatherLive C-class (%): {target_percent}")
    print(f"Diferenca absoluta: {abs(model_percent - target_percent) if target_percent is not None else 'N/A'}")
    print(f"SpaceWeatherLive probabilidades: {swl}")
    print(f"Explicacao: {result['explanation_short']}")


if __name__ == "__main__":
    main()
