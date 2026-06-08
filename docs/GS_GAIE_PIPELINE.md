# Generative AI For Engineering - Pipeline ML

## Problema

Prever risco operacional `LOW`, `MEDIUM` ou `HIGH` para uma missao espacial a partir de variaveis solares, ambientais e de criticidade de engenharia.

## Dados

O pipeline GAIE gera um dataset sintetico com 1.200 linhas e 13 colunas, atendendo ao requisito minimo de 1.000 linhas e 10 colunas.

Arquivo gerado apos execucao:

```text
backend/data/gaie/solar_ops_synthetic.csv
```

Variaveis:

```text
c_probability
m_probability
x_probability
solar_wind_speed_kms
proton_density_pcm3
xray_flux
kp_index
satellite_altitude_km
radiation_dose_rate
payload_criticality
comms_dependency
battery_margin_pct
risk_level
```

## Pipeline

Script principal:

```text
backend/ml/train_gaie_pipeline.py
```

Etapas:

1. Geracao do dataset sintetico.
2. Engenharia de atributos solares e operacionais.
3. Divisao treino/teste estratificada.
4. Treinamento de dois modelos:
   - Random Forest.
   - Gradient Boosting.
5. Comparacao por accuracy, classification report e matriz de confusao.
6. Escolha do melhor modelo.
7. Interpretabilidade com SHAP, quando disponivel.
8. Fallback de interpretabilidade por permutation importance.
9. Deploy via FastAPI.

## Como executar

```bash
cd backend
pip install -r requirements-ml.txt
python ml/train_gaie_pipeline.py --rows 1200 --seed 42
```

## Arquivos gerados

```text
backend/data/gaie/solar_ops_synthetic.csv
backend/ml/models/gaie_best_model.joblib
backend/ml/models/gaie_metrics.json
backend/reports/gaie_pipeline_report.md
backend/reports/gaie_feature_importance.csv
backend/reports/gaie_shap_summary.csv
```

## Deploy

Endpoints:

```text
GET /gaie/status
POST /gaie/predict
```

Exemplo de payload:

```json
{
  "features": {
    "c_probability": 60,
    "m_probability": 15,
    "x_probability": 1,
    "solar_wind_speed_kms": 520,
    "proton_density_pcm3": 8,
    "xray_flux": 0.000002,
    "kp_index": 4,
    "satellite_altitude_km": 550,
    "radiation_dose_rate": 70,
    "payload_criticality": 5,
    "comms_dependency": 4,
    "battery_margin_pct": 62
  }
}
```

## Observacao importante

O Hermes Copilot tambem usa prompts estruturados em `backend/genai/prompts.py`, mas a entrega GAIE do PDF cobra um pipeline IA/ML completo. Por isso, o projeto inclui tanto o Copilot generativo quanto o pipeline tabular com dois modelos e interpretabilidade.

