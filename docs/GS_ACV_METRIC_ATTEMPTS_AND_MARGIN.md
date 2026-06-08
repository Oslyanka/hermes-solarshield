# Hermes SolarShield - Evidencia de Metricas ACV

## Resultado honesto

A referencia de 88% nao foi atingida sem trapaca nos experimentos executados. O melhor resultado defensavel no teste historico oficial foi:

| Modelo                 | Dados                                                                         | Test accuracy | Margem de erro |
|------------------------|-------------------------------------------------------------------------------|--------------:|---------------:|
| `acv_multichannel_mlp` | SDOBenchmark full, amostras por evento, canais 94/131/171/193/211/magnetogram | 83.41%        | 16.59%         |

Esse resultado usa o split oficial `training/test` do SDOBenchmark full. O treino e a validacao saem apenas de `training`; o `test` oficial e usado somente para a medicao final.

## Melhor modelo atual

- Script: `backend/ml/train_acv_multichannel_features.py`
- Checkpoint: `backend/ml/models/acv_multichannel_mlp.pt`
- Comparacao JSON: `backend/ml/models/acv_multichannel_feature_comparison.json`
- Relatorio: `backend/reports/acv_multichannel_feature_report.md`
- Threshold escolhido na validacao: 0.55
- Matriz de confusao no teste:

| Real \ Predito | low | high |
|----------------|----:|-----:|
| low            | 307 | 52   |
| high           | 95  | 432  |

## Metricas por classe do melhor modelo

| Classe | Precision | Recall | F1    |
|--------|----------:|-------:|------:|
| low    | 0.764     | 0.855  | 0.807 |
| high   | 0.893     | 0.820  | 0.855 |

## Experimentos executados

| Experimento                                                      | Arquivo principal                               | Resultado observado |
|------------------------------------------------------------------|-------------------------------------------------|--------------------:|
| Checkpoint binario antigo `solar_flare_cnn.pt` em teste separado | `backend/ml/evaluate_model.py`                  | 64.08%              |
| CNN binaria tiny no SDOBenchmark full processado                 | `backend/ml/train_acv_binary_classifiers.py`    | 75.15%              |
| Features visuais por imagem no SDOBenchmark full                 | `backend/ml/train_acv_feature_models.py`        | 66.08%              |
| Features multicanal por amostra/evento                           | `backend/ml/train_acv_multichannel_features.py` | 83.41%              |
| DONKI C/M/X com duas CNNs                                        | `backend/ml/train_acv_donki_classifiers.py`     | 72.00%              |
| SDOBenchmark C/M/X com duas CNNs                                 | `backend/ml/train_acv_classifiers.py`           | 53.00%              |

## Nota contra vazamento

Nao foram usados nomes de arquivo, datas, rotulos do teste ou calibracao manual para forcar a metrica. O ajuste de threshold do melhor modelo foi feito somente no conjunto de validacao, antes da avaliacao final no teste.

## Caminhos legitimos para tentar chegar a 88%

1. Treinar CNN temporal/multicanal em GPU usando mais frames por canal.
2. Usar transfer learning com `torchvision` ou modelo solar pre-treinado, documentando a origem dos pesos.
3. Fazer validacao cruzada por regiao ativa para reduzir variancia por evento.
4. Melhorar o alvo: regressao de `peak_flux` seguida de classe low/high ou C/M/X.
5. Aumentar o dataset com mais eventos historicos GOES/SDO e manter teste final bloqueado.
