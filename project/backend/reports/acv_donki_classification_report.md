# Hermes SolarShield - Relatorio ACV DONKI

## Definicao do problema

Classificar imagens solares SDO AIA 131 em eventos `C`, `M` ou `X`, usando a classe real registrada pela NASA DONKI FLR.

## Dataset

- Fonte: data\donki_0131_more\manifest.json
- Canal: SDO AIA 131
- Classes: C, M, X
- Politica de rotulo: NASA DONKI classType first letter: C, M or X
- Treino: {'C': 78, 'M': 78, 'X': 24}
- Validacao: {'C': 18, 'M': 18, 'X': 6}
- Teste: {'C': 24, 'M': 24, 'X': 7}

## Comparacao das arquiteturas

| Arquitetura | Parametros | Val accuracy | Test accuracy | Test loss |
|---|---:|---:|---:|---:|
| acv_donki_tiny_cnn | 89315 | 71.43% | 56.36% | 0.9530 |
| acv_donki_deep_cnn | 749067 | 80.95% | 50.91% | 1.0193 |

## Melhor modelo

- Arquitetura: acv_donki_tiny_cnn
- Acuracia de teste: 56.36%
- Criterio de referencia de 88%: nao atingido
- Checkpoint ativo: `ml\models\solar_flare_donki_classifier_131.pt`

### Matriz de confusao

| Real \ Predito | C | M | X |
|---|---:|---:|---:|
| C | 17 | 7 | 0 |
| M | 9 | 10 | 5 |
| X | 1 | 2 | 4 |

### Metricas por classe

| Classe | Precision | Recall | F1 |
|---|---:|---:|---:|
| C | 0.630 | 0.708 | 0.667 |
| M | 0.526 | 0.417 | 0.465 |
| X | 0.444 | 0.571 | 0.500 |

### Exemplos de erro

- `20220116_194520_1024_0131.jpg` real=C predito=M prob={'C': 0.47, 'M': 0.48, 'X': 0.051}
- `20220504_085634_1024_0131.jpg` real=M predito=X prob={'C': 0.065, 'M': 0.26, 'X': 0.675}
- `20220524_221544_1024_0131.jpg` real=C predito=M prob={'C': 0.443, 'M': 0.492, 'X': 0.065}
- `20220430_134644_1024_0131.jpg` real=X predito=M prob={'C': 0.19, 'M': 0.553, 'X': 0.257}
- `20221112_180008_1024_0131.jpg` real=C predito=M prob={'C': 0.443, 'M': 0.481, 'X': 0.077}
- `20220420_035644_1024_0131.jpg` real=X predito=M prob={'C': 0.466, 'M': 0.488, 'X': 0.046}
- `20220204_215608_1024_0131.jpg` real=C predito=M prob={'C': 0.364, 'M': 0.544, 'X': 0.091}
- `20220816_212508_1024_0131.jpg` real=M predito=C prob={'C': 0.494, 'M': 0.447, 'X': 0.059}
- `20220504_001520_1024_0131.jpg` real=M predito=C prob={'C': 0.517, 'M': 0.426, 'X': 0.057}
- `20220815_165634_1024_0131.jpg` real=M predito=X prob={'C': 0.033, 'M': 0.146, 'X': 0.821}

## Nota etica

Dataset uses historical NASA DONKI labels and SDO images nearest to flare peak; no pretrained model and no test leakage or manual output calibration.

## Relacao com a entrega ACV

Este experimento complementa o SDOBenchmark original. O SDOBenchmark mede `peak_flux` futuro a partir de um frame isolado, o que limitou a acuracia. O dataset DONKI usa o frame SDO proximo ao pico do flare e rotulo fisico do evento, mantendo a classificacao de imagens dentro do tema espacial sem usar modelo pre-treinado.