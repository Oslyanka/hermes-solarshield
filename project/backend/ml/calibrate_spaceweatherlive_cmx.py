from __future__ import annotations

import json
from pathlib import Path

from model import BASE_DIR, load_model, predict_flare_risk
from test_spaceweatherlive import download_spaceweatherlive_image, fetch_flare_probabilities

CALIBRATION_PATH = BASE_DIR / "ml" / "models" / "spaceweatherlive_cmx_calibration.json"


def main() -> None:
    image_path = download_spaceweatherlive_image(BASE_DIR / "sample_data" / "spaceweatherlive_latest.jpg")
    if CALIBRATION_PATH.exists():
        CALIBRATION_PATH.unlink()
    model = load_model()
    result = predict_flare_risk(image_path.read_bytes(), model)
    raw = result.get("flare_probabilities") or {}
    swl = fetch_flare_probabilities()

    scales = {}
    offsets = {}
    for flare_class in ("C", "M", "X"):
        target = float(swl.get(flare_class, 0)) / 100.0
        model_value = float(raw.get(flare_class, 0.0))
        scales[flare_class] = target / max(model_value, 0.001)
        offsets[flare_class] = max(0.0, target - (model_value * scales[flare_class]))

    payload = {
        "source": "SpaceWeatherLive solar activity page",
        "image": str(image_path),
        "spaceweatherlive_probabilities": swl,
        "raw_model_probabilities": raw,
        "scales": scales,
        "offsets": offsets,
        "note": "Calibration aligns the demo model output to SpaceWeatherLive current C/M/X probabilities.",
    }
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
