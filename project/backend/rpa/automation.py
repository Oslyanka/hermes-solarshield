from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import sys
from pathlib import Path
from typing import Any

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rpa.alerting import record_operational_alert

BASE_DIR = Path(__file__).resolve().parents[1]
LOG_PATH = BASE_DIR / "logs" / "rpa.log"
EVIDENCE_PATH = BASE_DIR / "reports" / "rpa_last_run.json"


def default_sample() -> Path:
    validated = sorted((BASE_DIR / "sample_data" / "acv_recommended_0131").glob("high_*.jpg"))
    if validated:
        return validated[0]
    real_high = sorted((BASE_DIR / "sample_data" / "sdo_0131_examples" / "high").glob("*_1024_0131.jpg"))
    if real_high:
        return real_high[0]
    return BASE_DIR / "sample_data" / "solar_high.png"


def configure_logger() -> logging.Logger:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hermes-rpa")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logger.addHandler(handler)
    return logger


def run_against_api(api_url: str, sample: Path) -> dict[str, Any]:
    logger = configure_logger()
    logger.info("Iniciando rotina automatica Hermes SolarShield")
    if not sample.exists():
        raise FileNotFoundError(f"Imagem de exemplo nao encontrada: {sample}")

    mime_type = mimetypes.guess_type(sample.name)[0] or "image/jpeg"
    with sample.open("rb") as image_file:
        predict_response = requests.post(
            f"{api_url.rstrip('/')}/predict",
            files={"file": (sample.name, image_file, mime_type)},
            timeout=30,
        )
    predict_response.raise_for_status()
    analysis = predict_response.json()
    logger.info("Previsao concluida: risco=%s score=%s", analysis["risk_level"], analysis["score"])

    report_response = requests.post(
        f"{api_url.rstrip('/')}/generate-report",
        json={"analysis_id": analysis["id"]},
        timeout=30,
    )
    report_response.raise_for_status()
    report = report_response.json()
    logger.info("Relatorio gerado: id=%s", report["id"])
    alert = record_operational_alert(analysis, report)
    logger.info("Alerta registrado em planilha compartilhada simulada: %s", alert["csv_path"])
    logger.info("Email operacional gerado: %s", alert["email_path"])
    evidence = {
        "status": "completed",
        "api_url": api_url.rstrip("/"),
        "sample": str(sample),
        "analysis": {
            "id": analysis["id"],
            "risk_level": analysis["risk_level"],
            "score": analysis["score"],
            "confidence": analysis["confidence"],
            "source": analysis["source"],
        },
        "report": {
            "id": report["id"],
            "title": report["title"],
            "txt_path": report.get("txt_path"),
            "pdf_path": report.get("pdf_path"),
        },
        "alert": alert,
        "log_path": str(LOG_PATH),
    }
    EVIDENCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    EVIDENCE_PATH.write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    logger.info("Evidencia JSON salva em %s", EVIDENCE_PATH)
    return evidence


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Executa a rotina RPA do Hermes SolarShield.")
    parser.add_argument("--api-url", default="http://127.0.0.1:8000")
    parser.add_argument("--sample", default=default_sample(), type=Path)
    args = parser.parse_args()
    print(run_against_api(args.api_url, args.sample))
