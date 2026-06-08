# Hermes SolarShield - Auditoria de Requisitos da GS

## Visao computacional

- Problema espacial definido: classificacao de risco de flare solar por imagem SDO AIA 131.
- Dataset proprio/selecionado: imagens historicas em `backend/data/donki_0131` e templates validados em `backend/sample_data/acv_recommended_0131`.
- Duas CNNs treinadas do zero: `backend/ml/models/acv_tiny_cnn.pt` e `backend/ml/models/acv_deep_cnn.pt`.
- Comparacao entre arquiteturas: `backend/ml/models/acv_classification_comparison.json`.
- Relatorio com acuracia, perda, matriz de confusao e erros: `backend/reports/acv_classification_report.md`.
- Teste com imagens novas/historicas: `backend/reports/acv_real_test_run.md`.
- Justificativa tecnica para nao atingir 88% no classificador de imagem puro: `docs/GS_ACV_METRIC_ATTEMPTS_AND_MARGIN.md`.
- Interface funcional para upload/teste: tela `Analise Solar` do frontend.

## Pipeline de engenharia com IA

- Dataset sintetico minimo: `backend/data/gaie/solar_ops_synthetic.csv` com 1.200 linhas e 13 colunas.
- Tecnicas comparadas: Random Forest e Gradient Boosting.
- Preprocessamento e engenharia de atributos: `backend/ml/train_gaie_pipeline.py`.
- Metricas e melhor modelo: `backend/ml/models/gaie_metrics.json` e `backend/ml/models/gaie_best_model.joblib`.
- Interpretabilidade: `backend/reports/gaie_feature_importance.csv` e, quando disponivel, `backend/reports/gaie_shap_summary.csv`.
- Deploy: endpoints `GET /gaie/status` e `POST /gaie/predict`.
- Documentacao: `docs/GS_GAIE_PIPELINE.md` e `backend/reports/gaie_pipeline_report.md`.

## Automacao operacional

- Fluxo BPMN: `docs/rpa_flow.bpmn`.
- Blueprint estilo Power Automate: `backend/rpa/power_automate_flow_definition.json`.
- Script executavel: `backend/rpa/run_operational_routine.ps1`.
- Codigo da rotina: `backend/rpa/automation.py` e `backend/rpa/alerting.py`.
- Captura de fonte espacial: imagem solar AIA 131 validada.
- Regra de negocio: decisao por `risk_level` LOW/MEDIUM/HIGH.
- Integracao de terceiros simulada: planilha CSV e rascunho de email.
- Evidencia da execucao: `backend/reports/rpa_last_run.json`.
- Relatorio tecnico: `docs/RPA_TECHNICAL_REPORT.md` e `docs/RPA_TECHNICAL_REPORT.pdf`.
- Pacote ZIP: `deliverables/HermesSolarShield_RPA_GlobalSolution.zip`.
