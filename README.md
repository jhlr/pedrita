# Veritas — Uso via import (Python API)

Este README explica como o seu colega de infraestrutura deve importar e usar as funcionalidades do projeto diretamente via Python (sem usar a CLI).

Visão geral
- O código principal está em `v3/`:
  - `v3/helper.py` — utilitários e transformações auxiliares (`transform()`, `best_device()`, etc.).
  - `v3/predict.py` — funções de predição: `predict`, `predict_and_heatmap`, `predict_batch`, `evaluate_folder`.
    - Observação: `predict` / `predict_batch` já retornam probabilidades
    - Também expõe `set_model()` para carregar/instanciar o modelo diretamente do módulo `predict`.
  - `v3/train.py` — função de treino: `train_head`.
    - Também expõe `set_model()` para carregar/instanciar o modelo antes do treino.

Requisitos mínimos

Instale dependências (ajuste versões conforme seu ambiente):

```bash
python -m pip install torch torchvision timm joblib opencv-python numpy pytorch-grad-cam pillow
```

Como importar e usar (exemplos)

1) Carregar/instanciar o modelo

Agora você pode carregar o modelo diretamente a partir dos módulos `predict` ou `train` — não é necessário importar `helper` explicitamente.

```python
from v3 import predict, train

# Carrega (ou cria) e define o modelo global usado pelo módulo
# use force=True para criar via timm caso o arquivo não exista
predict.set_model('model_temp', force=True)
# ou, alternativamente, se for treinar antes: train.set_model('model_temp', force=True)

# A partir daqui, chamadas em predict.* e train.train_head funcionarão

```

2) Predição de uma única imagem

```python
import cv2
from v3 import predict

img = cv2.imread('path/to/image.jpg')  # OpenCV retorna BGR
fake_prob, heatmap = predict.predict_and_heatmap(img)

print('Probabilidade fake:', fake_prob)
if heatmap is not None:
  cv2.imwrite('outputs/heatmap.jpg', heatmap)

```

Observações:
- `predict.predict(img)` e `predict.predict_batch(...)` já retornam probabilidades (softmax). `predict.predict_and_heatmap` devolve a probabilidade de `fake` e um heatmap (BGR) quando aplicável.

3) Predição em lote

```python
imgs = [cv2.imread(p) for p in ['a.jpg', 'b.jpg']]
probs_batch = predict.predict_batch(imgs)
# probs_batch já contém probabilidades por classe (softmax)

```

4) Avaliar uma pasta de teste (programaticamente)

```python
from v3 import predict

# Estrutura esperada: test_dir/real/*  e test_dir/fake/*
probs, gts = predict.evaluate_folder('dataset/test', batch_size=16, limit=None, thresh=0.6)

```

5) Treinamento programático

```python
from v3 import train, dset

# carrega/define o modelo via train.set_model
train.set_model('model_temp', force=True)

# Preparar lista de caminhos (ou passar uma instância de dset.Dataset)
filepaths = []
with open('path/to/filelist.txt','r') as fh:
  filepaths = [l.strip() for l in fh if l.strip()]

# Executa treino da cabeça/classificador
train.train_head(filepaths=filepaths, epochs=5, batch_size=16)

```

Notas importantes
- `train.train_head` aceita uma lista de caminhos OU uma instância de `dset.Dataset` (veja `v3/dset.py` para as classes disponíveis).
- Você pode usar `predict.set_model()` ou `train.set_model()` para carregar/definir o modelo global; não é necessário chamar `helper.set_model()` diretamente.
- As leituras/escritas de imagens usam OpenCV (BGR).
- `helper.transform()` fornece o pipeline de pré-processamento compatível com o modelo carregado.

**Estatísticas Atuais e Como Gerá-las**

O repositório não embute números estáticos neste README para evitar divergência com execuções locais e modelos atualizados. Para gerar as estatísticas atuais (AUC, acurácia, curva ROC, etc.) execute a avaliação de teste usando a função de avaliação ou a CLI:

- Via CLI (avalia toda a pasta `dataset/test` com subpastas `real/` e `fake/`):

```bash
python v3/predict.py --model <nome_modelo> --eval dataset/test
```

Isso imprimirá o tempo de execução e chamará internamente `model.compare()` para gerar métricas e plots (quando aplicável). Os vetores de probabilidade retornados pela função também são usados para calcular métricas programaticamente.

- Via import (programaticamente):

```python
from v3 import predict
predict.set_model('model_temp', force=True)
probs, gts = predict.evaluate_folder('dataset/test', batch_size=16, thresh=0.6)
# probs: numpy array com probabilidades por amostra
# gts: lista de rótulos ground-truth (0/1)

# Para calcular métricas customizadas use sklearn.metrics
from sklearn.metrics import roc_auc_score, accuracy_score
auc = roc_auc_score(gts, probs)
print('AUC:', auc)
```

Onde `probs` é um array de probabilidades (classe `real`) produzido por `predict_batch`.

Onde procurar resultados:
- Arquivos de saída (heatmaps, imagens anotadas) são gravados em `outputs/` pelo CLI quando aplicável.
- Modelos treinados e calibradores ficam em `models/`.

Se você quiser que eu inclua números concretos (por exemplo, AUC, Accuracy, recall) nesta seção, envie as métricas obtidas localmente ou autorize-me a rodar a avaliação aqui.

**Uso via CLI (resumo rápido)**

- Avaliar uma pasta de teste:

```bash
python v3/predict.py -m pedrita2 --eval dataset/test
```

- Predizer uma imagem e salvar heatmap (quando disponível):

```bash
python v3/predict.py -m pedrita2 -i path/to/image.jpg
```

Boas práticas para integração
- Encapsule chamadas em wrappers da infraestrutura para lidar com logs, tratamento de exceções e paths relativos.
- Use `helper.best_device()` para verificar o dispositivo e mover tensores/modelo quando necessário.

Se quiser, eu também:
- acrescento um exemplo de `filelist.txt` em `dataset/`;
- crio `requirements.txt` com versões exatas usadas no ambiente.
