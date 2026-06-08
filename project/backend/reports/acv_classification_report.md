# Hermes SolarShield - Relatorio ACV

## 1. Definicao do problema

O problema de Visao Computacional e classificar imagens solares SDO AIA 131 em risco LOW, MEDIUM ou HIGH. A classificacao apoia a plataforma Hermes SolarShield na triagem operacional de clima espacial para infraestrutura orbital.

## 2. Dataset utilizado

- Fonte: data\sdobenchmark\SDOBenchmark_full
- Canal visual: SDO AIA 131
- Classes: LOW, MEDIUM, HIGH
- Treino: {'LOW': 595, 'MEDIUM': 595, 'HIGH': 434}
- Validacao: {'LOW': 105, 'MEDIUM': 105, 'HIGH': 77}
- Teste: {'LOW': 356, 'MEDIUM': 351, 'HIGH': 176}
- Rotulagem:
  - LOW: peak_flux < 1e-6 W/m2: below C-class solar flare range
  - MEDIUM: 1e-6 <= peak_flux < 1e-5 W/m2: C-class solar flare range
  - HIGH: peak_flux >= 1e-5 W/m2: M/X-class operational risk range

Pre-processamento: conversao para escala de cinza, autocontraste, redimensionamento, normalizacao numerica e aumentos leves no treino.

## 3. Treinamento de CNNs do zero

Foram treinadas duas arquiteturas convolucionais proprias, sem uso de modelos pre-treinados.

| Arquitetura  | Parametros | Val accuracy | Test accuracy | Test loss |
|--------------|------------|--------------|---------------|-----------|
| acv_tiny_cnn | 89315      | 57.14%       | 53.00%        | 0.8884    |
| acv_deep_cnn | 749067     | 56.10%       | 51.76%        | 0.9094    |

## 4. Avaliacao do melhor modelo

- Melhor arquitetura: acv_tiny_cnn
- Acuracia de teste: 53.00%
- Loss de teste: 0.8884
- Criterio de referencia de 88%: nao atingiu.

Justificativa tecnica: o rotulo `peak_flux` do SDOBenchmark representa a intensidade futura do evento solar na janela do exemplo, enquanto a entrada usada nesta entrega e apenas um frame do canal AIA 131. A imagem isolada nao contem toda a informacao temporal e magnetica usada por sistemas reais de previsao de flares. Melhorias futuras incluem usar sequencias temporais, multiplos canais SDO, magnetogramas HMI, aumento do dataset balanceado e modelos hibridos CNN + series temporais.

### Matriz de confusao

| Real \ Predito | LOW | MEDIUM | HIGH |
|----------------|-----|--------|------|
| LOW            | 292 | 38     | 26   |
| MEDIUM         | 134 | 60     | 157  |
| HIGH           | 30  | 30     | 116  |

### Metricas por classe

| Classe | Precision | Recall | F1    |
|--------|-----------|--------|-------|
| LOW    | 0.640     | 0.820  | 0.719 |
| MEDIUM | 0.469     | 0.171  | 0.251 |
| HIGH   | 0.388     | 0.659  | 0.488 |

### Exemplos de erros

- 2013-11-02T172200__131.jpg: real=HIGH predito=LOW prob={'LOW': 0.408, 'MEDIUM': 0.365, 'HIGH': 0.227}
- 2014-02-27T000000__131.jpg: real=HIGH predito=LOW prob={'LOW': 0.387, 'MEDIUM': 0.361, 'HIGH': 0.252}
- 2014-02-14T175800__131.jpg: real=MEDIUM predito=LOW prob={'LOW': 0.478, 'MEDIUM': 0.292, 'HIGH': 0.23}
- 2015-01-09T145301__131.jpg: real=MEDIUM predito=LOW prob={'LOW': 0.612, 'MEDIUM': 0.267, 'HIGH': 0.121}
- 2015-12-23T141200__131.jpg: real=HIGH predito=LOW prob={'LOW': 0.367, 'MEDIUM': 0.361, 'HIGH': 0.272}
- 2014-02-05T163100__131.jpg: real=MEDIUM predito=LOW prob={'LOW': 0.749, 'MEDIUM': 0.212, 'HIGH': 0.04}
- 2013-12-17T113200__131.jpg: real=MEDIUM predito=HIGH prob={'LOW': 0.054, 'MEDIUM': 0.406, 'HIGH': 0.54}
- 2012-11-29T180700__131.jpg: real=MEDIUM predito=HIGH prob={'LOW': 0.168, 'MEDIUM': 0.37, 'HIGH': 0.461}
- 2014-06-28T092400__131.jpg: real=MEDIUM predito=LOW prob={'LOW': 0.483, 'MEDIUM': 0.384, 'HIGH': 0.133}
- 2015-01-09T212401__131.jpg: real=MEDIUM predito=LOW prob={'LOW': 0.442, 'MEDIUM': 0.31, 'HIGH': 0.248}

## 5. Comparacao tecnica

A arquitetura Tiny usa menos camadas e menos parametros, sendo mais rapida e menos propensa a overfitting em bases pequenas. A arquitetura Deep adiciona blocos duplos de convolucao, BatchNorm e Dropout2d; isso aumenta capacidade de extrair padroes locais em regioes ativas, mas tambem exige mais regularizacao. A comparacao por acuracia, loss e matriz de confusao indica qual delas generalizou melhor no teste oficial.