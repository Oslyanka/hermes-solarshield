# Hermes SolarShield - Relatorio GAIE

## Problema

Prever o risco operacional LOW, MEDIUM ou HIGH para uma missao espacial a partir de telemetria solar, ambiente orbital e criticidade de engenharia.

## Dados

- Fonte: dataset sintetico gerado por IA/engenharia de requisitos, inspirado em SpaceWeatherLive, SDO e telemetria operacional.
- Linhas: 1200
- Colunas: 13
- Arquivo: `data\gaie\solar_ops_synthetic.csv`

## Pipeline

1. Geracao de dados sinteticos com distribuicoes controladas.
2. Engenharia de atributos solares e operacionais.
3. Divisao treino/teste estratificada.
4. Treinamento de Random Forest e Gradient Boosting.
5. Comparacao por accuracy, classification report e matriz de confusao.
6. Interpretabilidade por SHAP quando a biblioteca esta instalada, com fallback por permutation importance.
7. Deploy via FastAPI no endpoint `/gaie/predict`.

## Resultados

- Melhor modelo: gradient_boosting
- Accuracy de teste: 80.83%
- Metricas JSON: `ml\models\gaie_metrics.json`
- Importancia de atributos: `reports/gaie_feature_importance.csv`
- SHAP: SHAP nao executado: GradientBoostingClassifier is only supported for binary classification right now!

## Comparacao dos modelos

| Modelo | Accuracy |
|---|---:|
| Random Forest | 79.58% |
| Gradient Boosting | 80.83% |

## Deploy

Com a API ativa, envie as 12 variaveis tabulares para `POST /gaie/predict`. Se `gaie_best_model.joblib` existir, o endpoint usa o melhor modelo treinado; caso contrario, usa regras de fallback para manter a demonstracao funcional.
