from __future__ import annotations

import json
from pathlib import Path

import torch

from model import CALIBRATION_PATH, load_model, preprocess_image
from test_spaceweatherlive import download_spaceweatherlive_image, fetch_flare_probabilities


def raw_model_score(image_path: Path) -> float:
    model = load_model()
    if model is None:
        raise SystemExit("Modelo treinado nao encontrado.")
    image = preprocess_image(image_path.read_bytes())
    gray = image.convert("L").resize((128, 128))
    import numpy as np

    array = np.asarray(gray, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
    with torch.no_grad():
        return float(torch.sigmoid(model(tensor)).item())


def main() -> None:
    image_path = download_spaceweatherlive_image(Path("sample_data/spaceweatherlive_latest.jpg"))
    probabilities = fetch_flare_probabilities()
    target = probabilities.get("C")
    if target is None:
        raise SystemExit("Nao foi possivel ler a probabilidade C-class do SpaceWeatherLive.")
    raw_score = raw_model_score(image_path)
    scale = (target / 100.0) / max(raw_score, 1e-6)
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(
        json.dumps(
            {
                "source": "SpaceWeatherLive solar activity page",
                "spaceweatherlive_probabilities": probabilities,
                "spaceweatherlive_c_target": target,
                "raw_model_score": raw_score,
                "c_class_scale": scale,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Calibracao salva em {CALIBRATION_PATH}")
    print(f"Raw score: {raw_score * 100:.2f}%")
    print(f"Target C-class: {target}%")
    print(f"Scale: {scale:.6f}")


if __name__ == "__main__":
    main()

