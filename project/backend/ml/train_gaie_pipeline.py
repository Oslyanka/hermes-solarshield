from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_synthetic_dataset(rows: int, seed: int):
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(seed)
    c_probability = rng.beta(2.2, 3.4, rows) * 100
    m_probability = np.clip(c_probability * rng.beta(1.2, 7.5, rows) + rng.normal(2.0, 3.0, rows), 0, 100)
    x_probability = np.clip(m_probability * rng.beta(0.8, 10.0, rows) + rng.normal(0.3, 0.9, rows), 0, 100)
    solar_wind_speed_kms = np.clip(rng.normal(470, 120, rows) + x_probability * 4.5, 260, 950)
    proton_density_pcm3 = np.clip(rng.gamma(2.4, 3.1, rows) + m_probability / 12, 0.5, 45)
    xray_flux = np.clip((c_probability * 0.000000012) + (m_probability * 0.00000008) + (x_probability * 0.00000022), 0, 0.00008)
    kp_index = np.clip(rng.normal(2.8, 1.4, rows) + m_probability / 32 + x_probability / 18, 0, 9)
    satellite_altitude_km = rng.choice([550, 700, 1200, 20200, 35786], rows, p=[0.32, 0.18, 0.18, 0.14, 0.18])
    radiation_dose_rate = np.clip(rng.normal(35, 16, rows) + m_probability * 1.15 + x_probability * 3.2 + kp_index * 4.5, 2, 260)
    payload_criticality = rng.integers(1, 6, rows)
    comms_dependency = rng.integers(1, 6, rows)
    battery_margin_pct = np.clip(rng.normal(64, 18, rows) - kp_index * 1.5, 5, 100)

    severity = (
        c_probability * 0.008
        + m_probability * 0.020
        + x_probability * 0.050
        + kp_index * 0.12
        + radiation_dose_rate * 0.006
        + payload_criticality * 0.10
        + comms_dependency * 0.06
        - battery_margin_pct * 0.006
        + rng.normal(0, 0.24, rows)
    )
    bins = [-999, 1.55, 2.85, 999]
    labels = ["LOW", "MEDIUM", "HIGH"]
    risk_level = pd.cut(severity, bins=bins, labels=labels).astype(str)

    return pd.DataFrame(
        {
            "c_probability": c_probability.round(2),
            "m_probability": m_probability.round(2),
            "x_probability": x_probability.round(2),
            "solar_wind_speed_kms": solar_wind_speed_kms.round(2),
            "proton_density_pcm3": proton_density_pcm3.round(2),
            "xray_flux": xray_flux,
            "kp_index": kp_index.round(2),
            "satellite_altitude_km": satellite_altitude_km,
            "radiation_dose_rate": radiation_dose_rate.round(2),
            "payload_criticality": payload_criticality,
            "comms_dependency": comms_dependency,
            "battery_margin_pct": battery_margin_pct.round(2),
            "risk_level": risk_level,
        }
    )


def train_pipeline(args: argparse.Namespace) -> None:
    import joblib
    import pandas as pd
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.inspection import permutation_importance
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    backend_root = Path(__file__).resolve().parents[1]
    data_dir = backend_root / "data" / "gaie"
    model_dir = backend_root / "ml" / "models"
    report_dir = backend_root / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_synthetic_dataset(args.rows, args.seed)
    dataset_path = data_dir / "solar_ops_synthetic.csv"
    dataset.to_csv(dataset_path, index=False)

    features = [column for column in dataset.columns if column != "risk_level"]
    x_train, x_test, y_train, y_test = train_test_split(
        dataset[features],
        dataset["risk_level"],
        test_size=0.2,
        random_state=args.seed,
        stratify=dataset["risk_level"],
    )

    candidates = {
        "random_forest": RandomForestClassifier(n_estimators=220, max_depth=12, class_weight="balanced", random_state=args.seed),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=180, learning_rate=0.055, max_depth=3, random_state=args.seed),
    }

    results = {}
    best_name = ""
    best_model = None
    best_accuracy = -1.0
    for name, model in candidates.items():
        model.fit(x_train, y_train)
        prediction = model.predict(x_test)
        accuracy = accuracy_score(y_test, prediction)
        results[name] = {
            "accuracy": round(float(accuracy), 4),
            "classification_report": classification_report(y_test, prediction, output_dict=True, zero_division=0),
            "confusion_matrix": confusion_matrix(y_test, prediction, labels=["LOW", "MEDIUM", "HIGH"]).tolist(),
        }
        if accuracy > best_accuracy:
            best_name = name
            best_model = model
            best_accuracy = accuracy

    assert best_model is not None
    metrics_path = model_dir / "gaie_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "dataset": str(dataset_path),
                "rows": len(dataset),
                "columns": len(dataset.columns),
                "features": features,
                "models": results,
                "best_model": best_name,
                "best_accuracy": round(float(best_accuracy), 4),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    joblib.dump(best_model, model_dir / "gaie_best_model.joblib")

    importance = permutation_importance(best_model, x_test, y_test, n_repeats=10, random_state=args.seed)
    importance_frame = pd.DataFrame(
        {
            "feature": features,
            "permutation_importance_mean": importance.importances_mean,
            "permutation_importance_std": importance.importances_std,
        }
    ).sort_values("permutation_importance_mean", ascending=False)
    importance_frame.to_csv(report_dir / "gaie_feature_importance.csv", index=False)

    shap_status = "SHAP nao executado porque a biblioteca shap nao esta instalada."
    try:
        import shap

        explainer = shap.TreeExplainer(best_model)
        shap_values = explainer.shap_values(x_test)
        if isinstance(shap_values, list):
            mean_abs = sum(abs(values).mean(axis=0) for values in shap_values) / len(shap_values)
        else:
            mean_abs = abs(shap_values).mean(axis=0)
            if getattr(mean_abs, "ndim", 1) > 1:
                mean_abs = mean_abs.mean(axis=1)
        shap_frame = pd.DataFrame({"feature": features, "mean_abs_shap": mean_abs}).sort_values("mean_abs_shap", ascending=False)
        shap_frame.to_csv(report_dir / "gaie_shap_summary.csv", index=False)
        shap_status = "SHAP executado e salvo em backend/reports/gaie_shap_summary.csv."
    except Exception as exc:
        shap_status = f"SHAP nao executado: {exc}"

    report_path = report_dir / "gaie_pipeline_report.md"
    report_path.write_text(
        f"""# Hermes SolarShield - Relatorio GAIE

## Problema

Prever o risco operacional LOW, MEDIUM ou HIGH para uma missao espacial a partir de telemetria solar, ambiente orbital e criticidade de engenharia.

## Dados

- Fonte: dataset sintetico gerado por IA/engenharia de requisitos, inspirado em SpaceWeatherLive, SDO e telemetria operacional.
- Linhas: {len(dataset)}
- Colunas: {len(dataset.columns)}
- Arquivo: `{dataset_path.relative_to(backend_root)}`

## Pipeline

1. Geracao de dados sinteticos com distribuicoes controladas.
2. Engenharia de atributos solares e operacionais.
3. Divisao treino/teste estratificada.
4. Treinamento de Random Forest e Gradient Boosting.
5. Comparacao por accuracy, classification report e matriz de confusao.
6. Interpretabilidade por SHAP quando a biblioteca esta instalada, com fallback por permutation importance.
7. Deploy via FastAPI no endpoint `/gaie/predict`.

## Resultados

- Melhor modelo: {best_name}
- Accuracy de teste: {best_accuracy:.2%}
- Metricas JSON: `{metrics_path.relative_to(backend_root)}`
- Importancia de atributos: `reports/gaie_feature_importance.csv`
- SHAP: {shap_status}

## Comparacao dos modelos

| Modelo | Accuracy |
|---|---:|
| Random Forest | {results["random_forest"]["accuracy"]:.2%} |
| Gradient Boosting | {results["gradient_boosting"]["accuracy"]:.2%} |

## Deploy

Com a API ativa, envie as 12 variaveis tabulares para `POST /gaie/predict`. Se `gaie_best_model.joblib` existir, o endpoint usa o melhor modelo treinado; caso contrario, usa regras de fallback para manter a demonstracao funcional.
""",
        encoding="utf-8",
    )
    print(f"Dataset: {dataset_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treina o pipeline GAIE do Hermes SolarShield.")
    parser.add_argument("--rows", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=42)
    train_pipeline(parser.parse_args())
