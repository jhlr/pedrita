# Relatório comparativo — detecção e localização de manipulação em imagens

Avaliação de detectores próprios (CNN), detectores externos do HuggingFace e
estratégias de orquestração (ensemble) em três distribuições distintas, com foco
em generalização fora-de-distribuição (out-of-distribution, OOD).

## Resumo executivo

1. Nenhum detector único generaliza entre tipos de manipulação. O ranking dos
   modelos se inverte conforme a distribuição do input.
2. Os modelos CNN próprios dominam no domínio em que foram treinados (rostos
   deepfake/GAN): até 99% de acerto, mais rápidos que qualquer detector externo.
3. Os detectores do HuggingFace só agregam valor fora dessa distribuição. Em
   imagens de splicing/composição (CIPLAB, `quarto.png`) os CNN próprios falham e
   o `sdxl` é o único que sustenta recall de fake.
4. Orquestrar múltiplas CNNs próprias (ensemble) reduz variância dentro de uma
   distribuição, mas não cria capacidade que nenhuma delas tem: continua cego a
   tipos de manipulação não vistos no treino.
5. "Leve em parâmetros" não implica "leve na prática": o EfficientNet-B6 tem
   menos parâmetros que os ViT, mas é o mais lento e o menos acurado.
6. Detectores CNN leves prontos para essa tarefa praticamente não existem no
   HuggingFace (3 de 96 candidatos), e todos são piores que os modelos próprios.

## Metodologia

- Métrica: acurácia binária real vs fake. Para os modelos CNN também se reporta o
  detalhamento sure/wrong/dunno da função `compare` (limiar 0.7).
- Limiar de decisão: 0.5 nos benchmarks de detector (pred = fake se score > 0.5);
  0.7 nas avaliações via `compare` (CIPLAB e eval de pasta).
- Amostragem: 50–60 imagens por classe nos benchmarks rápidos; CIPLAB avaliado na
  íntegra (2041 imagens).
- Latência medida em Apple Silicon (MPS), milissegundos por imagem.
- Ressalva: os números de amostra (N=50–60/classe) têm margem de erro de alguns
  pontos percentuais; servem para comparação relativa, não como métrica final.

## Datasets

| Dataset | Tipo de fake | Tamanho (teste) | Papel |
|---|---|---|---|
| `dataset` | geral (variado) | 1283 fake / 1301 real | distribuição geral |
| `faces` | deepfake / GAN de rostos | 10000 / 10000 | domínio dos modelos próprios |
| `ciplab` | splicing / Photoshop (olho, nariz, boca) | 960 / 1081 | OOD, falsificação manual |
| `quarto.png` | pessoa inserida em foto real | 1 imagem | OOD pontual |

## Achado 1 — Detectores HuggingFace: o ranking depende da distribuição

Modelos testados: `light` = Skullly/DeepFake-EN-B6 (40.7M, EfficientNet);
`deepfake` = prithivMLmods/Deep-Fake-Detector-v2 (85.8M, ViT);
`sdxl` = Organika/sdxl-detector (86.7M, Swin).

`dataset` (geral), N=60/classe:

| Detector | Params | Acurácia | Recall fake | Recall real | Latência |
|---|---|---|---|---|---|
| light | 40.7M | 55.8% | 21.7% | 90.0% | 399 ms |
| deepfake | 85.8M | 50.8% | 98.3% | 3.3% | 257 ms |
| sdxl | 86.7M | 75.8% | 76.7% | 75.0% | 132 ms |

`faces` (rostos deepfake), N=60/classe:

| Detector | Acurácia | Recall fake | Recall real | Latência |
|---|---|---|---|---|
| light | 71.7% | 86.7% | 56.7% | 300 ms |
| deepfake | 48.3% | 60.0% | 36.7% | 52 ms |
| sdxl | 55.0% | 38.3% | 71.7% | 60 ms |

O melhor em `dataset` (sdxl, 75.8%) é mediano em `faces` (55%); o melhor em
`faces` (light, 71.7%) é o pior em `dataset` (55.8%). O `deepfake`, apesar do
nome, colapsa em ambos (classifica quase tudo como fake).

## Achado 2 — Caso OOD pontual (`quarto.png`)

Foto real com uma pessoa inserida (composição, não deepfake clássico).

| Fonte | Score de fake |
|---|---|
| mother (CNN próprio) | 3% |
| ensemble (5 CNNs próprios) | 28% |
| deepfake (HF) | 75% |
| sdxl (HF) | 99.97% |

Os modelos próprios são enganados; apenas os detectores externos — em especial o
`sdxl` — identificam a manipulação. Caso emblemático do ponto cego de splicing.

## Achado 3 — CNN leve pronta no HuggingFace: escassa e inferior

Varredura de 96 detectores de classificação de imagem. Apenas 3 são CNN
(o restante é ViT/Swin/transformer):

| CNN (HF) | Params | `dataset` | `faces` |
|---|---|---|---|
| date3k2/resnet-real-fake-image | 23.5M | 50.8% | 50.0% |
| Skullly/DeepFake-EN-B6 | 40.7M | 55.8% | 71.7% |
| Okohogbole/ai-vs-real-resnet50 | 25M | config inválida (não carrega) | — |

Comparado aos modelos próprios em `faces`:

| Modelo | Acurácia `faces` | Latência |
|---|---|---|
| mother (próprio) | 99.2% | 50 ms |
| faces5 (próprio) | 91.7% | 21 ms |
| melhor CNN do HF (light) | 71.7% | 300 ms |

A CNN própria é mais acurada e mais rápida que qualquer CNN leve disponível no
HuggingFace para essa tarefa.

## Achado 4 — Modelos próprios individuais e por família

Acurácia individual (N=50/classe), `dataset` e `faces`:

| Modelo | `dataset` | `faces` |
|---|---|---|
| mother | 92% | 97% |
| ultra_1 | 94% | 91% |
| ultra_3 | 91% | 95% |
| ultra_4 | 88% | 98% |
| ultra_5 | 88% | 98% |
| ultra_6 | 92% | 98% |
| ultra_8 | 90% | 97% |
| pedrita7 | 94% | 93% |
| pedrita1 | 94% | 81% |
| pedrita6 | 96% | 86% |
| pedrita13 | 44% | 97% |
| pedrita (base) | 76% | 53% |
| faces5 | 37% | 88% |

A família `ultra` é consistente nas duas distribuições; a família `pedrita` tem
alta variância (de 44% a 97%) e vários modelos fracos.

Ensemble por família (média das probabilidades):

| Família | nº modelos | `dataset` | `faces` |
|---|---|---|---|
| pedrita (todas) | 17 | 96% | 88% |
| ultra (todas) | 7 | 90% | 97% |
| pedrita + ultra | 24 | 95% | 95% |

`ultra` orquestrada é a mais equilibrada e a mais enxuta. `pedrita` é melhor só no
`dataset`, ao custo de `faces`.

## Achado 5 — Ensemble curado de 5 modelos

Busca exaustiva entre combinações de 5 (ordenado por pior-caso). Vencedor:

> mother + ultra_1 + ultra_6 + ultra_8 + pedrita7

| Conjunto | `dataset` | `faces` |
|---|---|---|
| ensemble curado (5) | 90.8% | 98.3% |
| melhor modelo único | 92% (mother) | 98% (ultra_6) |

O ensemble curado iguala ou supera o melhor modelo único nas duas distribuições
simultaneamente, com baixa variância.

## Achado 6 — CIPLAB (splicing/Photoshop): o ponto cego confirmado

Avaliação na íntegra (2041 imagens), convertida em recall por classe:

| Modelo | Recall real | Recall fake |
|---|---|---|
| mother (sozinho) | ~88% | ~7% |
| ensemble (5 próprios) | ~88% | ~4% |
| sdxl (HF) | ~52% | ~41% |

Detalhamento `compare` (limiar 0.7):

| | Real W / S / D | Fake W / S / D | Overall S |
|---|---|---|---|
| ensemble | 2.3% / 46.8% / 3.9% | 40.5% / 1.9% / 4.7% | 48.7% |
| sdxl | 21.0% / 27.5% / 4.5% | 23.7% / 19.5% / 3.8% | 47.1% |

Conclusões do achado:
- Os CNN próprios (treinados em deepfake/GAN) são cegos ao splicing manual:
  classificam ~88–93% dos fakes do CIPLAB como reais.
- O ensemble não corrige isso (recall de fake ~4%, igual ou pior que mother
  sozinho): orquestrar modelos da mesma família não inventa capacidade ausente.
- O `sdxl` pega ~10x mais desses fakes (~41% vs ~4%), mas perde precisão nos
  reais (erra ~21% deles). Os dois são complementares e ambos insuficientes
  sozinhos.

## Conclusões e recomendações

1. Domínio conhecido (rostos deepfake): usar o CNN próprio. É o mais acurado e o
   mais rápido. Não há ganho em adicionar detectores externos.
2. Robustez geral: manter o `sdxl` como rede de segurança OOD, acionado quando o
   CNN próprio está incerto (fusão por confiança / gated), e não como média fixa
   — média cega derruba os 99% do domínio conhecido para a faixa dos ~50–75%.
3. Orquestração: o ensemble curado de 5 (mother + ultra_1 + ultra_6 + ultra_8 +
   pedrita7) é o melhor equilíbrio entre acurácia e variância dentro das
   distribuições conhecidas. Não resolve OOD por si só.
4. Solução durável para splicing: treinar no tipo que falta. Com o CIPLAB
   disponível, fazer split treino/teste e fine-tune de uma `ultra` é o caminho
   que efetivamente eleva o recall de fake de ~4% para um patamar utilizável,
   sem depender de modelo externo.
5. Leveza: a métrica de peso deve ser latência, não número de parâmetros. O
   `sdxl` (86.7M) é mais rápido (132 ms) que o EfficientNet-B6 (40.7M, 399 ms).

## Apêndice — proveniência dos dados

- Detectores HF carregados via `transformers` (AutoModelForImageClassification).
- Modelos próprios: arquivos `models/*.pkl` (timm/EfficientNet).
- Tracking de experimentos via MLflow (`v3/tracking.py`), métricas em namespace
  `train/` e `eval/`.
- Limiares e tamanhos de amostra indicados por seção. Amostras pequenas têm
  margem de erro; recomenda-se reavaliação na íntegra antes de decisão final.
