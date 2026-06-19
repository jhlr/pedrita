"""Núcleo agnóstico de provider para a análise de CONTEXTO de uma imagem.

Em vez de localizar manipulação (uma máscara de pixels), pedimos a um modelo de
visão (Gemini, OpenAI/GPT, ...) uma leitura estruturada em JSON do contexto da
imagem, complementando os detectores locais (veredito + heatmap) com uma avaliação
de risco legível para o cliente final.

Tudo que é COMUM a qualquer provider vive aqui — prompt, parsing, normalização do
contrato de saída, cache por hash e logging no MLflow. Cada provider (ex.:
`gemini.py`, `openai_vision.py`) só implementa a chamada à própria API e delega o
resto a `analyze()`, então o resultado é idêntico independentemente de quem gerou.

Contrato de saída (chaves de `context()`/`analyze()`):
  - scene_description     : descrição factual da cena
  - coherence             : {plausible: bool, opinion: str}
  - criticality           : {level: low|medium|high, categories: [...]}
  - manipulation_certainty: confiança (%) de manipulação, 0-100
  - manipulation_type     : como foi manipulada (ou "none")
  - suspect_regions       : onde a manipulação parece estar
  - evidence              : pistas visuais que sustentam a análise
  - recommended_action    : publish | human_review | block
  - provider / model / cached : origem do laudo (preenchidos por analyze())
"""
from __future__ import annotations

import os, io, json, re
from typing import Callable, Optional, Sequence, Dict

from PIL import Image as pil

# Carrega um .env (raiz do projeto ou CWD) uma única vez, para as chaves dos
# providers (GEMINI_API_KEY, OPENAI_API_KEY, ...) poderem viver em arquivo.
try:
	from dotenv import load_dotenv, find_dotenv
	load_dotenv(find_dotenv(usecwd=True))
except ImportError:
	pass

__all__ = [
	'CRITICALITY_CATEGORIES', 'MANIPULATION_TYPES', 'ACTIONS', 'CONTEXT_MAX_SIDE',
	'CONTEXT_PROMPT', 'resolve_key', 'load_image', 'parse_json_object',
	'normalize', 'log_context', 'analyze',
]

# --- Domínios do contrato -------------------------------------------------

# Categorias de criticidade reconhecidas (assuntos sensíveis).
CRITICALITY_CATEGORIES = (
	'celebrity', 'child', 'sensitive_data', 'politician',
	'public_figure', 'violence', 'medical', 'document',
	'alcohol', 'drugs',
)
# Como a imagem foi manipulada (melhor palpite). 'none' = parece autêntica.
MANIPULATION_TYPES = (
	'face_swap', 'splice_composite', 'inpainting',
	'fully_generated', 'edited', 'none',
)
# Decisão operacional para o cliente, derivada de criticidade x certeza.
ACTIONS = ('publish', 'human_review', 'block')

# Maior lado enviado ao provider de visão. O cap reduz tokens/custo sem perder
# qualidade para esta tarefa (0 = não redimensiona). Override por env.
CONTEXT_MAX_SIDE = int(os.environ.get('CONTEXT_MAX_SIDE', '1024'))


# --- Utilidades de entrada ------------------------------------------------

def resolve_key(env_names: Sequence[str]) -> Optional[str]:
	"""Primeira chave de API não-vazia entre `env_names`, ou None."""
	return next((os.environ[k] for k in env_names if os.environ.get(k)), None)


def load_image(img) -> pil.Image:
	"""Abre/normaliza a imagem em RGB e a reduz (cópia) antes de enviar ao provider.

	Nunca muta a imagem do chamador.
	"""
	out = img if isinstance(img, pil.Image) else pil.open(str(img))
	out = out.convert('RGB')
	if CONTEXT_MAX_SIDE and max(out.size) > CONTEXT_MAX_SIDE:
		out = out.copy()
		out.thumbnail((CONTEXT_MAX_SIDE, CONTEXT_MAX_SIDE), pil.Resampling.LANCZOS)
	return out


def parse_json_object(text: str) -> dict:
	"""Extrai, da melhor forma possível, um objeto JSON da resposta do modelo."""
	text = re.sub(r'^```(?:json)?|```$', '', text.strip(), flags=re.MULTILINE).strip()
	try:
		data = json.loads(text)
	except json.JSONDecodeError:
		m = re.search(r'\{.*\}', text, flags=re.DOTALL)
		if not m:
			return {}
		try: data = json.loads(m.group(0))
		except json.JSONDecodeError: return {}
	return data if isinstance(data, dict) else {}


# --- Prompt compartilhado -------------------------------------------------

CONTEXT_PROMPT = (
	'You are a forensic image-context analyst. Examine the image and return ONLY '
	'a JSON object (no prose) with exactly these keys:\n'
	'  "scene_description": short factual description of the scene, in {lang}.\n'
	'  "coherence": object with\n'
	'       "plausible": boolean — does the scene make physical/logical sense?\n'
	'       "opinion": one or two sentences in {lang} on its logic/reality/sense,\n'
	'                  noting anything implausible (lighting, anatomy, physics, text).\n'
	'  "criticality": object with\n'
	'       "level": one of "low", "medium", "high" — how sensitive the subject is.\n'
	'       "categories": array, any of '
	'["celebrity","child","sensitive_data","politician","public_figure",'
	'"violence","medical","document","alcohol","drugs"] that apply (empty if none).\n'
	'  "manipulation_certainty": number 0-100 — your confidence (%) that the image '
	'was AI-generated or digitally manipulated.\n'
	'  "manipulation_type": one of "face_swap","splice_composite","inpainting",'
	'"fully_generated","edited","none" — best guess of how it was manipulated.\n'
	'  "suspect_regions": array of short strings naming where manipulation seems to '
	'be (e.g. "man\'s face", "background door"); empty if none.\n'
	'  "evidence": array of short strings, each a concrete visual cue supporting '
	'your assessment.\n'
	'  "recommended_action": one of "publish","human_review","block" — based on '
	'criticality and manipulation_certainty.\n'
	'Base manipulation_certainty on visual evidence (blending, warping, lighting/'
	'shadow inconsistency, texture, impossible details). Be calibrated, not extreme.'
)


# --- Normalização do contrato ---------------------------------------------

def normalize(data: dict) -> Dict:
	"""Valida/normaliza o dict cru do modelo na estrutura canônica de contexto.

	Compartilhado entre providers, então a saída é idêntica não importa quem gerou.
	"""
	coh = data.get('coherence') or {}
	crit = data.get('criticality') or {}
	level = str(crit.get('level', 'low')).strip().lower()
	if level not in ('low', 'medium', 'high'):
		level = 'low'
	cats = [str(c).strip().lower() for c in (crit.get('categories') or [])
		if str(c).strip()]
	try: certainty = float(data.get('manipulation_certainty', 0) or 0)
	except (TypeError, ValueError): certainty = 0.0
	certainty = max(0.0, min(100.0, certainty))

	mtype = str(data.get('manipulation_type', 'none')).strip().lower()
	if mtype not in MANIPULATION_TYPES:
		mtype = 'none'
	action = str(data.get('recommended_action', 'human_review')).strip().lower()
	if action not in ACTIONS:
		action = 'human_review'
	regions = [str(s).strip() for s in (data.get('suspect_regions') or []) if str(s).strip()]
	evidence = [str(s).strip() for s in (data.get('evidence') or []) if str(s).strip()]

	return {
		'scene_description': str(data.get('scene_description', '')).strip(),
		'coherence': {
			'plausible': bool(coh.get('plausible', True)),
			'opinion': str(coh.get('opinion', '')).strip(),
		},
		'criticality': {'level': level, 'categories': cats},
		'manipulation_certainty': certainty,
		'manipulation_type': mtype,
		'suspect_regions': regions,
		'evidence': evidence,
		'recommended_action': action,
	}


# --- Persistência / cache -------------------------------------------------

def log_context(provider: str, image, result: Dict, track: bool = True) -> None:
	"""Registra imagem recebida + contexto gerado como um run do MLflow (best-effort).

	Compartilhado por todos os providers, então o armazenamento é idêntico (imagem
	por hash, context.json como artefato). Importa o tracking de forma preguiçosa
	para manter este módulo utilizável sem mlflow.
	"""
	if not track:
		return
	try:
		from . import tracking
	except Exception:
		return
	tracking.log_inference(
		provider,
		params={'manipulation_certainty': result['manipulation_certainty'],
			'criticality': result['criticality']['level']},
		metrics={'manipulation_certainty': result['manipulation_certainty']},
		images={'received.png': image},
		artifacts={'context.json': result},
		tags={'kind': provider, 'manipulation_type': result['manipulation_type']},
	)


# --- Orquestração ---------------------------------------------------------

def analyze(
	generate_fn: Callable[[pil.Image, str, Optional[str], float], str],
	provider: str,
	img,
	model: Optional[str],
	lang: str,
	track: bool,
	default_model: Optional[str] = None,
) -> Dict:
	"""Núcleo compartilhado: load → cache por hash → gera → normaliza → loga.

	Se a imagem já tiver um context.json no DB (mesma hash), retorna o cacheado e
	NÃO chama o provider — economiza cota/custo. A origem (provider + modelo) fica
	gravada no resultado, então um cache-hit também sabe quem gerou o laudo.

	`generate_fn(image, prompt, model, temperature)` é a única parte específica do
	provider; tudo o mais (prompt, parsing, normalização, cache, log) é comum.
	"""
	image = load_image(img)

	cached = _lookup_cache(image)
	if cached is not None:
		cached['cached'] = True
		return cached

	model = model or default_model
	raw = generate_fn(image, CONTEXT_PROMPT.format(lang=lang), model, 0.2)
	result = normalize(parse_json_object(raw))
	result['provider'] = provider
	result['model'] = model
	result['cached'] = False
	log_context(provider, image, result, track)
	return result


def _lookup_cache(image) -> Optional[Dict]:
	"""Laudo já armazenado para esta imagem (por hash), ou None. Best-effort."""
	try:
		from . import tracking
	except Exception:
		return None
	sha = tracking.image_sha256(image)
	return tracking.find_context(sha) if sha else None
