# Integrantes
#### Aksel Viktor Caminha Rae RM: 99011 
#### Ian Xavier Kuraoka RM: 98860 
#### Lucas Laia Manentti RM: 97709 
#### Rony Ken Nagai RM: 551549 
#### Tomáz Versolato Carballo RM: 551417 

# Hermes SolarShield

Hermes SolarShield e um webapp academico baseado no Projeto Hermes para previsao e resposta a solar flares. A aplicacao simula uma plataforma de clima espacial que analisa imagens solares, estima risco nas proximas 24 horas, gera explicacoes com IA generativa e automatiza alertas e relatorios por RPA.

O projeto roda em modo demo mesmo sem o dataset real SDOBenchmark. Quando o dataset e um modelo treinado estiverem disponiveis, o backend ja possui a estrutura de preparacao, treinamento e carregamento do modelo.

## Stack

- Frontend: React + Vite
- Backend: Python + FastAPI
- Computer Vision: Python, Pillow, NumPy e estrutura para PyTorch
- Banco de dados: SQLite
- RPA: script Python e endpoint manual
- Relatorios: TXT e PDF simples

## Estrutura

```text
hermes-solarshield/
  frontend/
  backend/
  backend/ml/
  backend/genai/
  backend/rpa/
  backend/database/
  backend/reports/
  backend/logs/
  README.md
  .env.example
```

## Como instalar

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn main:app --reload
```

A API ficara em:

```text
http://127.0.0.1:8000
```

Teste rapido:

```text
http://127.0.0.1:8000/health
```

### Frontend

Em outro terminal:

```bash
cd frontend
npm install
npm run dev
```

O dashboard ficara em:

```text
http://127.0.0.1:5173
```

## Endpoints principais

- `GET /health`: status da API.
- `POST /predict`: recebe imagem solar e retorna `risk_level`, `score`, `confidence`, `flare_probabilities` com C/M/X e `explanation_short`.
- `POST /generate-report`: gera relatorio textual e PDF simples.
- `POST /rpa/run`: executa rotina automatica completa.
- `GET /analyses`: lista analises salvas.
- `GET /reports`: lista relatorios gerados.

Tambem foram incluidos:

- `POST /copilot`: resposta do Hermes Copilot.
- `GET /gaie/status`: lista atributos e status do modelo tabular GAIE.
- `POST /gaie/predict`: prediz risco operacional com telemetria tabular GAIE.
- `GET /samples`: imagens solares demo.
- `GET /rpa/logs`: logs da automacao.
- `GET /reports/{id}/download/txt` e `/pdf`: download dos relatorios.

## Modelo de Computer Vision

O projeto possui dois caminhos:

- `solar_flare_classifier_131.pt`: classificador ACV atual, treinado do zero com SDOBenchmark AIA 131 para classes `LOW`, `MEDIUM` e `HIGH`.
- `solar_flare_cnn.pt`: classificador binario legado para risco geral.
- `solar_flare_cmx_131.pt`: modelo multi-label auxiliar, treinado no canal SDO AIA 131 para estimar chances de flares C-class, M-class e X-class.

O backend prioriza `backend/ml/models/solar_flare_classifier_131.pt` quando esse arquivo existe. Esse checkpoint atende ao requisito de classificacao de imagens da materia Applied Computer Vision. A saida principal e `class_probabilities`:

```json
{
  "LOW": 0.12,
  "MEDIUM": 0.31,
  "HIGH": 0.57
}
```

Para manter a comparacao operacional com SpaceWeatherLive, o backend tambem deriva uma estimativa auxiliar `flare_probabilities`:

```json
{
  "C": 0.85,
  "M": 0.20,
  "X": 0.01
}
```

Modelo ACV atual:

```text
fonte: SDOBenchmark_full
canal: SDO AIA 131
classes: LOW (<C), MEDIUM (C-class), HIGH (M/X-class)
arquiteturas treinadas do zero: acv_tiny_cnn e acv_deep_cnn
melhor modelo: acv_tiny_cnn
val_accuracy: 57.14%
test_accuracy: 53.00%
matriz de confusao e analise de erros: backend/reports/acv_classification_report.md
comparacao JSON: backend/ml/models/acv_classification_comparison.json
```

O criterio de referencia de 88% nao foi atingido no teste honesto. A justificativa tecnica esta no relatorio: o `peak_flux` futuro do SDOBenchmark depende de informacao temporal e magnetica, enquanto esta entrega usa apenas um frame AIA 131 por amostra. O projeto continua dentro do parametro academico porque apresenta a limitacao e aponta melhorias futuras.

### Escolha do modelo ativo

Use `HERMES_MODEL_MODE` no arquivo `backend/.env` ou na variavel de ambiente antes de iniciar a API:

```text
HERMES_MODEL_MODE=classifier
```

Carrega o modelo academico `solar_flare_classifier_131.pt` e mostra `Prob. LOW`, `Prob. MEDIUM` e `Prob. HIGH`.

```text
HERMES_MODEL_MODE=forecast
```

Carrega o modelo `solar_flare_cmx_131.pt`, treinado para aproximar as porcentagens historicas C/M/X do SpaceWeatherLive. Nessa configuracao o frontend deve mostrar `Chance C-class`, `Chance M-class` e `Chance X-class`, nao probabilidades LOW/MEDIUM/HIGH.

No modo `forecast`, o backend usa a saida direta do modelo `solar_flare_cmx_131.pt`, sem calibracao externa ou ajuste posterior por arquivo JSON. O SpaceWeatherLive aparece apenas como referencia de comparacao na interface.

Exemplo no PowerShell:

```powershell
cd backend
$env:HERMES_MODEL_MODE="forecast"
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Modelo DONKI alternativo, mantido como referencia academica:

```text
macro_accuracy: 92.31%
C accuracy: 100.00%
M accuracy: 84.62%
X accuracy: 92.31%
canal: SDO AIA 131
fonte de treino: NASA DONKI FLR + SDO AIA 131
```

Modelo de forecast atual, treinado para aproximar as probabilidades historicas do SpaceWeatherLive:

```text
amostras: 114 registros historicos/validos
imagem: NASA SDO AIA 131 original, 1024_0131.jpg
rotulo: Wayback Machine da pagina SpaceWeatherLive Solar Flares
modo atual: ensemble simples dos dois melhores checkpoints por validacao
calibracao manual: nao aplicada
protected samples: desabilitado por padrao no re-treino honesto
MAE medio do melhor membro: 8.35 pontos percentuais
MAE C do melhor membro: 5.39 pontos percentuais
MAE M do melhor membro: 12.46 pontos percentuais
MAE X do melhor membro: 7.20 pontos percentuais
avaliacao externa atual: C 99.1% vs 99%, M 55.6% vs 60%, X 21.8% vs 15%
fonte de treino: Wayback SpaceWeatherLive forecast + NASA SDO AIA 131 ensemble
```

Relatorio do re-treino honesto:

```text
backend/reports/wayback_forecast_retrain_report.md
```

O checkpoint antigo de 90.48% com tolerancia de +/-25 pontos percentuais foi preservado em:

```text
backend/ml/models/solar_flare_cmx_131_tol025_backup.pt
```

## Modo demo

Se nao existir modelo treinado em `backend/ml/models/`, o sistema usa uma heuristica inteligente para demonstracao. Ela estima risco a partir de brilho, contraste, densidade de bordas e concentracao de pixels intensos na imagem solar.

As imagens de exemplo sao geradas automaticamente em `backend/sample_data/` na inicializacao da API.

## Dataset SDOBenchmark

Dataset usado como referencia:

```text
https://www.kaggle.com/datasets/fhnw-i4ds/sdobenchmark
```

Sugestao de fluxo:

1. Baixe o dataset pelo Kaggle.
2. Coloque os arquivos em `backend/data/sdobenchmark/`.
3. Rode a preparacao:

```bash
cd backend
python ml/prepare_dataset.py --raw-dir data/sdobenchmark --output-dir data/processed
```

Opcionalmente, com a Kaggle API configurada em `C:\Users\<seu_usuario>\.kaggle\kaggle.json`:

```bash
cd backend
pip install -r requirements-ml.txt
python ml/download_sdobenchmark.py --output-dir data/sdobenchmark
python ml/prepare_dataset.py --raw-dir data/sdobenchmark --output-dir data/processed
```

4. Organize o dataset processado no formato esperado por `torchvision.datasets.ImageFolder`:

```text
backend/data/processed/
  train/
    low/
    high/
  val/
    low/
    high/
```

5. Instale PyTorch e TorchVision conforme seu ambiente.
6. Treine:

```bash
python ml/train.py --dataset-dir data/processed --max-epochs 30 --target-accuracy 0.90 --output ml/models/solar_flare_cnn.pt
```

O treino salva o melhor modelo e para automaticamente quando a acuracia de validacao chega a 90%.

Para a entrega de Applied Computer Vision, treine e compare duas CNNs do zero no SDOBenchmark AIA 131:

```bash
python ml/train_acv_classifiers.py --raw-dir data/sdobenchmark --output-dir ml/models --report-dir reports --channel 131 --image-size 96 --max-epochs 60 --target-accuracy 0.88 --batch-size 64 --max-train-per-class 700
```

Esse comando gera:

```text
backend/ml/models/acv_tiny_cnn.pt
backend/ml/models/acv_deep_cnn.pt
backend/ml/models/solar_flare_classifier_131.pt
backend/ml/models/acv_classification_comparison.json
backend/reports/acv_classification_report.md
```

Para treinar o modelo multi-label C/M/X no canal AIA 131 usando o SDOBenchmark:

```bash
python ml/train_cmx.py --raw-dir data/sdobenchmark --output ml/models/solar_flare_cmx_131.pt --channel 131 --max-epochs 80 --target-accuracy 0.90 --max-images 1000 --batch-size 32
```

Para re-treinar com rotulos historicos mais precisos da NASA DONKI FLR, baixando os frames SDO AIA 131 mais proximos do pico de cada flare:

```bash
python ml/train_donki_cmx.py --start 2024-01-01 --end 2026-06-02 --dataset-dir data/donki_0131 --output ml/models/solar_flare_cmx_131.pt --max-per-class 45 --max-epochs 80 --target-accuracy 0.90 --batch-size 32
```

Esse e o caminho recomendado para melhorar as probabilidades `M` e `X`, porque usa o `classType` real dos eventos, como `C5.3`, `M1.8` e `X8.7`.

Para treinar o modelo de forecast usando snapshots do Wayback da pagina do SpaceWeatherLive e imagens originais NASA SDO AIA 131, uma por dia:

```bash
python ml/train_wayback_forecast.py --start 20200901 --end 20260602 --dataset-dir data/wayback_swl_0131 --snapshot-limit 500 --max-per-day 1 --build-only
python ml/train_wayback_forecast.py --dataset-dir data/wayback_swl_0131 --output ml/models/solar_flare_cmx_131.pt --train-only --max-epochs 220 --target-accuracy 0.95 --target-mae 0.035 --tolerance 0.05 --batch-size 16 --patience 70 --seed 7
```

Este e o caminho recomendado quando o objetivo e aproximar as porcentagens exibidas pelo SpaceWeatherLive, e nao apenas classificar eventos fisicos C/M/X.
Para evitar ajuste indevido ao valor atual da pagina, os parametros de amostras protegidas ficam desligados por padrao e nao foram usados no re-treino honesto documentado.

Os scripts de calibracao antigos foram mantidos apenas como historico experimental. A aplicacao atual nao aplica calibracao externa no endpoint `/predict`; para aproximar os valores do SpaceWeatherLive, re-treine o modelo com mais snapshots historicos em vez de ajustar a saida em pos-processamento.

Para testar com uma imagem real AIA 131 da pagina SpaceWeatherLive/NASA SDO:

```bash
python ml/test_spaceweatherlive.py
```

Comparacao dos exemplos AIA 131:

```text
backend/reports/sdo_0131_examples_comparison.md
```

## Generative AI

O modulo `backend/genai/` contem prompts estruturados para:

- resumo tecnico;
- explicacao simples;
- plano de mitigacao;
- relatorio executivo.

Por padrao, o Hermes Copilot usa templates locais sem chamar API externa. Para integrar uma API real futuramente, configure uma chave no `.env`:

```text
GENAI_API_KEY=sua_chave
```

Nenhuma chave e hardcoded no codigo.

### Generative AI For Engineering - pipeline GAIE

Para atender ao enunciado de GAIE, o projeto tambem inclui um pipeline tabular completo de IA/ML em `backend/ml/train_gaie_pipeline.py`. Ele gera um dataset sintetico de engenharia espacial com 1.200 linhas e 13 colunas, treina duas tecnicas diferentes e publica o melhor modelo no endpoint FastAPI `/gaie/predict`.

Instale as dependencias de experimento:

```bash
cd backend
pip install -r requirements-ml.txt
```

Treine o pipeline:

```bash
python ml/train_gaie_pipeline.py --rows 1200 --seed 42
```

Esse comando gera:

```text
backend/data/gaie/solar_ops_synthetic.csv
backend/ml/models/gaie_best_model.joblib
backend/ml/models/gaie_metrics.json
backend/reports/gaie_pipeline_report.md
backend/reports/gaie_feature_importance.csv
backend/reports/gaie_shap_summary.csv
```

O SHAP e executado quando a biblioteca `shap` esta instalada. Se houver incompatibilidade local, o script ainda salva `gaie_feature_importance.csv` por permutation importance para manter a interpretabilidade documentada.

Exemplo de chamada ao deploy tabular:

```json
POST /gaie/predict
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

## RPA

A rotina automatica simula:

1. carregar uma imagem solar de exemplo;
2. executar previsao;
3. salvar resultado no SQLite;
4. gerar relatorio;
5. criar alerta;
6. registrar logs.

Execucao manual pelo frontend: pagina `RPA Automations`, botao `Executar rotina automatica`.

Execucao manual por script:

```bash
cd backend
python rpa/automation.py --api-url http://127.0.0.1:8000
```

Logs:

```text
backend/logs/rpa.log
```

### Agendamento

Windows Task Scheduler:

```text
Programa: caminho\para\python.exe
Argumentos: rpa\automation.py --api-url http://127.0.0.1:8000
Iniciar em: C:\...\hermes-solarshield\backend
```

Linux/macOS cron, exemplo a cada 6 horas:

```cron
0 */6 * * * cd /caminho/hermes-solarshield/backend && python rpa/automation.py --api-url http://127.0.0.1:8000
```

## Como cada materia e atendida

### Applied Computer Vision

- Upload ou selecao de imagem solar demo.
- Preprocessamento de imagem em `backend/ml/model.py`.
- Predicao de risco `LOW`, `MEDIUM` ou `HIGH`.
- Score numerico, confianca e probabilidades por classe.
- Dataset SDOBenchmark AIA 131 selecionado e rotulado por `peak_flux`.
- Duas arquiteturas CNN proprias treinadas do zero: `acv_tiny_cnn` e `acv_deep_cnn`.
- Comparacao por accuracy, loss, matriz de confusao, precision, recall, F1 e exemplos de erros.
- Relatorio academico em `backend/reports/acv_classification_report.md`.
- Script de preparacao do SDOBenchmark em `backend/ml/prepare_dataset.py`.
- Script principal de treino ACV em `backend/ml/train_acv_classifiers.py`.

### Generative AI For Engineering

- Hermes Copilot na pagina `Generative AI`.
- Prompts estruturados em `backend/genai/prompts.py`.
- Geracao de resumo tecnico, resumo para leigos, plano de mitigacao e relatorio executivo.
- Fallback por templates locais, sem dependencia obrigatoria de API externa.
- Preparado para integracao futura via `.env`.
- Pipeline GAIE em `backend/ml/train_gaie_pipeline.py`.
- Dataset sintetico de engenharia espacial com 1.200 linhas e 13 colunas.
- Comparacao entre Random Forest e Gradient Boosting.
- Validacao por accuracy, classification report e matriz de confusao em `backend/ml/models/gaie_metrics.json`.
- Interpretabilidade com SHAP em `backend/reports/gaie_shap_summary.csv` quando `shap` esta instalado.
- Fallback de interpretabilidade por permutation importance em `backend/reports/gaie_feature_importance.csv`.
- Deploy tabular por FastAPI em `POST /gaie/predict`.

### Robotic Process Automation RPA

- Endpoint `POST /rpa/run`.
- Script `backend/rpa/automation.py`.
- Logs em SQLite e em `backend/logs/rpa.log`.
- Simulacao completa de pipeline operacional: imagem, predicao, banco, relatorio e alerta.
- Botao manual no frontend.
- Instrucoes para agendamento com cron ou Task Scheduler.

## Checklist academico

- [x] Aplicacao web unica com frontend e backend.
- [x] Dashboard dark mode com identidade Hermes.
- [x] Sidebar fixa com logo.
- [x] Paginas Dashboard, ACV, Generative AI, RPA e Reports.
- [x] FastAPI com endpoints solicitados.
- [x] Persistencia SQLite para analises e relatorios.
- [x] Computer Vision em modo demo e estrutura para modelo real.
- [x] Scripts para dataset e treinamento.
- [x] Duas CNNs treinadas do zero para classificacao de imagens.
- [x] Comparacao entre arquiteturas com accuracy, loss e matriz de confusao.
- [x] Analise de erros e justificativa tecnica quando a referencia de 88% nao e atingida.
- [x] IA Generativa com prompts e fallback sem API externa.
- [x] Pipeline GAIE com dataset sintetico 1000+ linhas e 10+ colunas.
- [x] Dois modelos tabulares comparados para GAIE.
- [x] Interpretabilidade GAIE por SHAP ou permutation importance.
- [x] Endpoint FastAPI `/gaie/predict` para deploy do modelo tabular.
- [x] RPA manual e script agendavel.
- [x] Relatorios TXT e PDF simples.
- [x] README com instalacao, execucao e relacao com as materias.

## Arquivos de entrega GS

Use estes arquivos como guia final para Applied Computer Vision e Generative AI For Engineering:

```text
docs/GS_ENTREGA_O_QUE_ENVIAR.md
docs/GS_ACV_MODEL_ARCHITECTURES.md
docs/GS_GAIE_PIPELINE.md
docs/GS_VIDEO_ROTEIRO_ACV_GAIE.md
docs/acv_gaie_traceability.md
notebooks/acv_training_demo.ipynb
notebooks/gaie_pipeline_demo.ipynb
```