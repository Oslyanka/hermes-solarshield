from __future__ import annotations

from pathlib import Path
from typing import Any

import re
import requests
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from database.db import add_rpa_log, create_analysis, get_analysis, init_db, list_analyses, list_reports, list_rpa_logs
from gaie.inference import GAIE_MODEL_PATH, FEATURE_NAMES, predict_engineering_risk
from genai.copilot import answer_question, generate_sections, prompt_catalog
from ml.model import load_model, predict_flare_risk
from ml.model import MODEL_MODE, MODEL_PATH
from reporting import generate_report_files
from rpa.alerting import record_operational_alert
from sample_data.generate_samples import create_sample

BASE_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = BASE_DIR / "sample_data"
LOG_FILE = BASE_DIR / "logs" / "rpa.log"
SPACEWEATHERLIVE_URL = "https://www.spaceweatherlive.com/en/solar-activity.html"

app = FastAPI(title="Hermes SolarShield API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "https://hermessolarshield.vercel.app/"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL = None


class ReportRequest(BaseModel):
    analysis_id: int | None = None


class CopilotRequest(BaseModel):
    question: str
    analysis_id: int | None = None


class GaiePredictionRequest(BaseModel):
    features: dict[str, Any]


def ensure_samples() -> None:
    for level in ("low", "medium", "high"):
        path = SAMPLE_DIR / f"solar_{level}.png"
        if not path.exists():
            create_sample(path, level)


def automation_sample() -> Path:
    validated_high = sorted((SAMPLE_DIR / "acv_recommended_0131").glob("high_*.jpg"))
    if validated_high:
        return validated_high[0]
    real_high = sorted((SAMPLE_DIR / "sdo_0131_examples" / "high").glob("*_1024_0131.jpg"))
    if real_high:
        return real_high[0]
    return SAMPLE_DIR / "solar_high.png"


def aia_131_examples() -> list[dict[str, str]]:
    recommended_root = SAMPLE_DIR / "acv_recommended_0131"
    recommended_examples: list[dict[str, str]] = []
    if recommended_root.exists():
        for level in ("high", "medium", "low"):
            for index, path in enumerate(sorted(recommended_root.glob(f"{level}_*.jpg"))[:3], start=1):
                recommended_examples.append(
                    {
                        "id": f"acv-{level}-{index}",
                        "level": level.upper(),
                        "label": f"Template validado {level.upper()} {index}",
                        "template": "validated_aia_131",
                        "channel": "SDO AIA 131",
                        "format": "single __131.jpg",
                        "filename": path.name,
                        "url": f"/samples/acv-recommended/{path.name}",
                    }
                )
        if recommended_examples:
            return recommended_examples

    root = SAMPLE_DIR / "sdo_0131_examples"
    examples: list[dict[str, str]] = []
    preferred = {
        "high": {"20240510_065546_1024_0131.jpg", "20240514_165246_1024_0131.jpg", "20241003_120000_1024_0131.jpg"},
        "medium": {"20260501_000000_1024_0131.jpg", "20260505_000000_1024_0131.jpg", "20260510_000000_1024_0131.jpg"},
        "low": {"20260505_024600_1024_0131.jpg", "20260510_044200_1024_0131.jpg", "20260515_000000_1024_0131.jpg"},
    }
    for level in ("high", "medium", "low"):
        folder = root / level
        paths = sorted(folder.glob("*_1024_0131.jpg"))
        preferred_paths = [path for path in paths if path.name in preferred.get(level, set())]
        remaining_paths = [path for path in paths if path not in preferred_paths]
        selected = (preferred_paths + remaining_paths)[:3]
        for index, path in enumerate(selected, start=1):
            examples.append(
                {
                    "id": f"0131-{level}-{index}",
                    "level": level.upper(),
                    "label": f"Recomendado AIA 131 {level.upper()} {index}",
                    "template": "recommended_aia_131",
                    "channel": "SDO AIA 131",
                    "format": "1024_0131.jpg",
                    "filename": path.name,
                    "url": f"/samples/0131/{level}/{path.name}",
                }
            )
    return examples


@app.on_event("startup")
def startup() -> None:
    global MODEL
    import os
    print("APP DIR CONTENTS:", os.listdir("/app"))
    print("ML DIR:", os.listdir("/app/ml") if os.path.exists("/app/ml") else "NO ML DIR")
    print("MODELS DIR:", os.listdir("/app/ml/models") if os.path.exists("/app/ml/models") else "NO MODELS DIR")
    init_db()
    ensure_samples()
    MODEL = load_model()
    print(f"MODEL LOADED: {MODEL}")


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "service": "Hermes SolarShield", "mode": "demo" if MODEL is None else "trained"}


@app.get("/model/status")
def model_status() -> dict[str, Any]:
    metadata = getattr(MODEL, "hermes_metadata", {}) if MODEL is not None else {}
    return {
        "mode": "demo" if MODEL is None else "trained",
        "selected_model_mode": MODEL_MODE,
        "model_path": str(MODEL_PATH),
        "model_exists": MODEL_PATH.exists(),
        "model_metadata": metadata,
        "external_calibration_applied": False,
    }


@app.get("/gaie/status")
def gaie_status() -> dict[str, Any]:
    return {
        "feature_count": len(FEATURE_NAMES),
        "features": FEATURE_NAMES,
        "model_path": str(GAIE_MODEL_PATH),
        "model_exists": GAIE_MODEL_PATH.exists(),
        "endpoint": "/gaie/predict",
    }


@app.post("/gaie/predict")
def gaie_predict(request: GaiePredictionRequest) -> dict[str, Any]:
    return predict_engineering_risk(request.features)


@app.get("/spaceweatherlive")
def spaceweatherlive() -> dict[str, Any]:
    try:
        response = requests.get(SPACEWEATHERLIVE_URL, headers={"User-Agent": "HermesSolarShield/1.0"}, timeout=30)
        response.raise_for_status()
        text = re.sub(r"<[^>]+>", " ", response.text)
        text = re.sub(r"\s+", " ", text)
        probabilities = {
            flare_class.upper(): int(value)
            for flare_class, value in re.findall(r"([CMX])-class\s+solar flare\s+(\d+)%", text, flags=re.IGNORECASE)
        }
        return {"source": SPACEWEATHERLIVE_URL, "probabilities": probabilities}
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Erro ao consultar SpaceWeatherLive: {exc}") from exc


@app.get("/samples")
def samples() -> list[dict[str, str]]:
    ensure_samples()
    real_examples = aia_131_examples()
    if real_examples:
        return real_examples
    return [
        {"id": "low", "level": "LOW", "label": "Solar Sample LOW", "url": "/samples/solar_low.png"},
        {"id": "medium", "level": "MEDIUM", "label": "Solar Sample MEDIUM", "url": "/samples/solar_medium.png"},
        {"id": "high", "level": "HIGH", "label": "Solar Sample HIGH", "url": "/samples/solar_high.png"},
    ]


@app.get("/samples/0131/{level}/{filename}")
def aia_131_sample_file(level: str, filename: str) -> FileResponse:
    if level not in {"low", "medium", "high"} or "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Sample invalido.")
    path = SAMPLE_DIR / "sdo_0131_examples" / level / filename
    if not path.exists() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path)


@app.get("/samples/acv-recommended/{filename}")
def acv_recommended_sample_file(filename: str) -> FileResponse:
    if "/" in filename or "\\" in filename:
        raise HTTPException(status_code=400, detail="Sample invalido.")
    path = SAMPLE_DIR / "acv_recommended_0131" / filename
    if not path.exists() or path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path)


@app.get("/samples/{filename}")
def sample_file(filename: str) -> FileResponse:
    path = SAMPLE_DIR / filename
    if not path.exists() or path.suffix.lower() != ".png":
        raise HTTPException(status_code=404, detail="Sample not found")
    return FileResponse(path)


@app.get("/spaceweatherlive/image")
def spaceweatherlive_latest_image() -> FileResponse:
    path = SAMPLE_DIR / "spaceweatherlive_latest.jpg"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Imagem SpaceWeatherLive ainda nao foi baixada. Rode ml/test_spaceweatherlive.py.")
    return FileResponse(path)


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem.")
    try:
        image_bytes = await file.read()
        filename = file.filename or "upload"
        use_forecast_calibration = False
        result = predict_flare_risk(image_bytes, MODEL, forecast_calibration=use_forecast_calibration)
        metadata = {
            "content_type": file.content_type,
            "forecast_calibration": use_forecast_calibration,
            "features": result.get("features"),
            "class_probabilities": result.get("class_probabilities"),
            "flare_probabilities": result.get("flare_probabilities"),
        }
        analysis = create_analysis(result, source=filename, metadata=metadata)
        return {
            **analysis,
            "features": result["features"],
            "class_probabilities": result.get("class_probabilities"),
            "flare_probabilities": result.get("flare_probabilities"),
            "mode": result["mode"],
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao analisar imagem: {exc}") from exc


@app.post("/copilot")
def copilot(request: CopilotRequest) -> dict[str, str]:
    analysis = None
    if request.analysis_id:
        try:
            analysis = get_analysis(request.analysis_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
    elif list_analyses(1):
        analysis = list_analyses(1)[0]
    return answer_question(request.question, analysis)


@app.get("/genai/prompts")
def prompts() -> dict[str, str]:
    return prompt_catalog()


@app.post("/generate-report")
def generate_report(request: ReportRequest) -> dict[str, Any]:
    try:
        analyses = list_analyses(1)
        if request.analysis_id is not None:
            analysis = get_analysis(request.analysis_id)
        elif analyses:
            analysis = analyses[0]
        else:
            raise HTTPException(status_code=404, detail="Nenhuma analise encontrada.")
        sections = generate_sections(analysis)
        return generate_report_files(analysis, sections)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar relatorio: {exc}") from exc


@app.post("/rpa/run")
def run_rpa() -> dict[str, Any]:
    ensure_samples()
    logs = []
    try:
        def log(message: str) -> None:
            logs.append(add_rpa_log(message))
            LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
            with LOG_FILE.open("a", encoding="utf-8") as handle:
                handle.write(f"{logs[-1]['created_at']} INFO {message}\n")

        log("Rotina operacional: carregando imagem solar validada.")
        sample = automation_sample()
        image_bytes = sample.read_bytes()
        log("Rotina operacional: executando leitura de risco solar.")
        result = predict_flare_risk(image_bytes, MODEL)
        analysis = create_analysis(
            result,
            source=f"automation:{sample.name}",
            metadata={
                "automation": True,
                "sample": str(sample),
                "features": result.get("features"),
                "class_probabilities": result.get("class_probabilities"),
                "flare_probabilities": result.get("flare_probabilities"),
            },
        )
        log(f"Rotina operacional: resultado salvo no banco com risco {analysis['risk_level']}.")
        report = generate_report_files(analysis, generate_sections(analysis))
        log(f"Rotina operacional: relatorio gerado com id {report['id']}.")
        alert = record_operational_alert(analysis, report)
        log(f"Rotina operacional: alerta registrado em planilha e email draft.")
        return {"status": "completed", "sample": str(sample), "analysis": analysis, "report": report, "alert": alert, "logs": logs}
    except Exception as exc:
        add_rpa_log(f"Rotina operacional: falha - {exc}", level="ERROR")
        raise HTTPException(status_code=500, detail=f"Erro na rotina RPA: {exc}") from exc


@app.get("/analyses")
def analyses() -> list[dict[str, Any]]:
    return list_analyses()


@app.get("/reports")
def reports() -> list[dict[str, Any]]:
    return list_reports()


@app.get("/reports/{report_id}/download/{kind}")
def download_report(report_id: int, kind: str) -> FileResponse:
    report = next((item for item in list_reports(200) if item["id"] == report_id), None)
    if report is None:
        raise HTTPException(status_code=404, detail="Relatorio nao encontrado.")
    path_value = report.get("pdf_path") if kind == "pdf" else report.get("txt_path")
    if not path_value:
        raise HTTPException(status_code=404, detail="Arquivo nao encontrado.")
    path = Path(path_value)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Arquivo nao existe no disco.")
    return FileResponse(path, filename=path.name)


@app.get("/rpa/logs")
def rpa_logs() -> list[dict[str, Any]]:
    return list_rpa_logs()
