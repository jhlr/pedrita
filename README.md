# Veritas / Pedrita — Classificação de Imagens (repositório)

Autores:

- Ivan Edward @ iers-bd
- Elizabete Barbosa @ elizabetealbuquerque
- Joao Rietra @ jhlr

Nome da disciplina: Machine Learning I - School Innovation.

Nome da instituição de ensino: CESAR School.

Nome da solução desenvolvida: Veritas

Breve descrição

Esta solução (Veritas, também referida como Pedrita no repositório) contém utilitários, modelos e scripts para treino e inferência de classificadores de imagem (com foco em detecção de manipulação/deepfake) usados no projeto. O pacote principal está em `v3/` e expõe helpers para treino, predição e avaliação, além de uma camada opcional de análise de contexto via Gemini. O acompanhamento de experimentos é feito com MLflow.

Links do projeto

- Google Drive com materiais: https://drive.google.com/drive/folders/1ywYBYQ1KoG0kaChqxis3LcpLXM4J5CHt?usp=sharing
- Site do projeto: https://app-veritas.netlify.app

Estrutura principal

- `v3/` — pacote principal com helpers, dataset, treino, predição, tracking (MLflow) e análise de contexto (Gemini).
- `models/` — modelos serializados (`.pkl`, `.joblib`).
- `dataset/`, `faces/`, `ciplab/`, `diffusion/`, `outputs/` — pastas de dados e saídas.
- `pipeline.py` — exemplo de pipeline de treino/avaliação usado por desenvolvedores.

OBS: `v3/localize.py` (orquestração de detectores) e `v3/video.py` (predição em vídeo) estão atualmente desativados (comentados).

Requisitos

Recomenda-se usar um ambiente virtual Python (venv/virtualenv).

Exemplo rápido:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r v3/requirements.txt
```

OBS: o `requirements.txt` dentro de `v3/` lista dependências úteis como `torch`, `timm`, `torchvision`, `joblib`, `Pillow`, `mlflow`, `scikit-learn` e (para a camada de contexto) `google-genai` e `python-dotenv`.

Como executar (instruções detalhadas)

O pacote `v3` fornece um ponto de entrada via `__main__` para uso por linha de comando. Há duas formas comuns de usar o repositório:

1) Usando o pacote como módulo Python (recomendado):

```bash
# sintaxe geral
python v3 test <model_path> -i <pasta> -l <quantidade>

# exemplo prático: avalia pastas de imagens usando um modelo salvo
python v3 test models/mother.pkl -i faces -l 10000
```

Parâmetros principais:

- `<model_path>`: caminho para o arquivo do modelo serializado (por exemplo `models/mother.pkl`).
- `test`: subcomando para avaliação (outros subcomandos: `train`, `gemini`, `merge`).
- `-i` / `--image`: pasta com imagens para avaliar (pasta com subpastas por classe) ou imagem única para heatmap.
- `-l` / `--limit`: limite do número de amostras a usar (opcional).

2) Usando o pacote dentro de um script Python ou REPL:

```python
import v3 as pedrita
pedrita.best_device()
pedrita.set_model('models/mother.pkl')

# treino com loop de validação (alimenta o scheduler e guarda o melhor epoch)
ds = pedrita.DirDataset('./dataset/train', limit=3000)
pedrita.train(ds, val='./dataset', epochs=3, lr=5e-5)

# predição e heatmap (Grad-CAM) para uma imagem
prob_real, heatmap_img = pedrita.heatmap('quarto.png', minmax=True)
print(f'prob_real={prob_real:.3f}')
heatmap_img.save('heatmap.png')

# avaliação de uma pasta (subpastas real/ e fake/)
diff, acc = pedrita.evaluate_folder('./dataset')
```

Exemplos adicionais de CLI

- Treinar com uma pasta organizada:

```bash
python v3 train models/mother.pkl -i /caminho/para/treino --epochs 3 -l 1000
```

- Análise de contexto via Gemini (JSON):

```bash
python v3 gemini -i quarto.png
```

Análise de contexto (Gemini)

A camada de contexto usa o Gemini para complementar o veredito do modelo (real/fake + heatmap) com uma leitura estruturada da imagem. O veredito técnico continua sendo do CNN; o Gemini fornece contexto, criticidade e ação recomendada.

Requisitos: `pip install google-genai python-dotenv` e uma chave em `GEMINI_API_KEY` (lida de variável de ambiente ou de um arquivo `.env` na raiz). Opcional: `GEMINI_MODEL` (default `gemini-2.5-flash`).

Uso via import:

```python
import v3 as pedrita

ctx = pedrita.gemini.context('quarto.png')           # dict (JSON abaixo)
ctx = pedrita.gemini.context('quarto.png', lang='Portuguese', model='gemini-2.5-flash')
print(ctx['manipulation_certainty'], ctx['recommended_action'])
```

Formato do JSON retornado por `context()`:

```json
{
  "scene_description": "descrição factual da cena",
  "coherence": {
    "plausible": true,
    "opinion": "opinião sobre lógica/realidade/sentido da cena"
  },
  "criticality": {
    "level": "low",
    "categories": ["celebrity", "child", "sensitive_data", "politician",
                   "public_figure", "violence", "medical", "document"]
  },
  "manipulation_certainty": 0,
  "manipulation_type": "none",
  "suspect_regions": ["ex.: rosto do homem", "porta ao fundo"],
  "evidence": ["pistas visuais concretas"],
  "recommended_action": "publish"
}
```

Campos e domínios:

- `scene_description` (str) — o que a imagem mostra.
- `coherence.plausible` (bool) e `coherence.opinion` (str) — se a cena faz sentido físico/lógico, e por quê.
- `criticality.level` — `low` | `medium` | `high`.
- `criticality.categories` — subconjunto de `celebrity`, `child`, `sensitive_data`, `politician`, `public_figure`, `violence`, `medical`, `document`.
- `manipulation_certainty` (float 0–100) — confiança (%) de manipulação, segundo o Gemini.
- `manipulation_type` — `face_swap` | `splice_composite` | `inpainting` | `fully_generated` | `edited` | `none`.
- `suspect_regions` (list[str]) — onde a manipulação parece estar, em texto.
- `evidence` (list[str]) — pistas visuais que sustentam a análise.
- `recommended_action` — `publish` | `human_review` | `block`.

Onde estão as implementações

- `v3/helper.py` — funções centrais (`set_model`, `best_device`, transforms, utilitários de I/O).
- `v3/predict.py` — rotinas de predição/avaliação e helpers para gerar heatmaps/visualizações.
- `v3/train.py` — loop de treino (com validação) e hooks.
- `v3/dset.py` — definição de `DirDataset` e outros datasets.
- `v3/tracking.py` — integração canônica com MLflow (métricas em `train/*` e `eval/*`, backend SQLite).
- `v3/gemini.py` — análise de contexto via Gemini (`context()`).

Links rápidos (arquivo no repo):
- [v3/helper.py](v3/helper.py)
- [v3/predict.py](v3/predict.py)
- [v3/train.py](v3/train.py)
- [v3/dset.py](v3/dset.py)
- [v3/tracking.py](v3/tracking.py)
- [v3/gemini.py](v3/gemini.py)

Contribuição

- Abra issues para bugs ou melhorias.
- Faça PRs pequenas e focadas; adicione testes quando possível.

Licença

Coloque aqui a licença do projeto, se aplicável.
