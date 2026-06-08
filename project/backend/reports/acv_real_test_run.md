# Hermes SolarShield - Teste Real ACV

## Configuracao

- Modelo: `backend/ml/models/acv_multichannel_mlp.pt`
- Dataset: `backend/data/sdobenchmark/SDOBenchmark_full/test`
- Tipo de teste: teste oficial historico, nao usado no treino
- Canais usados: 94, 131, 171, 193, 211, magnetogram
- Threshold de decisao: 0.55, escolhido previamente na validacao
- Classes: `low`, `high`

## Resultado agregado

| Metrica           | Valor  |
|-------------------|-------:|
| Eventos testados  | 886    |
| Accuracy          | 83.41% |
| Erro geral        | 16.59% |

## Matriz de confusao

| Real \ Predito | low | high |
|----------------|-----|------|
| low            | 307 | 52   |
| high           | 95  | 432  |

## Exemplos reais do teste

| Evento                        | Peak flux              | Real | Predito | Prob high | Correto |
|-------------------------------|------------------------|------|---------|-----------|---------|
| `11402_2012_01_15_12_00_00_0` | 7.647058823529411e-06  | high | high    | 0.883     | sim     |
| `11402_2012_01_15_12_00_00_1` | 7.647058823529411e-06  | high | high    | 0.945     | sim     |
| `11402_2012_01_20_21_20_01_0` | 1.5294117647058826e-06 | high | high    | 0.942     | sim     |
| `11402_2012_01_18_03_55_01_0` | 1.4117647058823531e-06 | high | high    | 0.824     | sim     |
| `11402_2012_01_27_04_26_00_0` | 0.0002                 | high | high    | 0.930     | sim     |
