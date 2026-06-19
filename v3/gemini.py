"""Gemini-backed contextual analysis of an image.

Instead of localizing manipulation (a pixel mask), Gemini returns a structured
JSON read of the image's CONTEXT, to complement the local detectors (verdict +
heatmap) with a human-readable risk assessment for the end client:

  - scene_description     : what the scene is
  - coherence             : opinion on logic / reality / plausibility
  - criticality           : how sensitive the subject is (celebrities, children,
                            sensitive data, politicians, ...) + a level
  - manipulation_certainty: % confidence the image was manipulated (0-100)

Requires `google-genai` and GEMINI_API_KEY (or GOOGLE_API_KEY); loads a .env.
No torch / model file needed.
"""
from __future__ import annotations

import os, json, re
from typing import Optional, Dict

from PIL import Image as pil

# Load a .env (project root or CWD) so GEMINI_API_KEY can live in a file.
try:
	from dotenv import load_dotenv, find_dotenv
	load_dotenv(find_dotenv(usecwd=True))
except ImportError:
	pass

__all__ = ['context', 'MODEL']

# Default model: fast + cheap. Swap to '-pro' for hard cases.
MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
_API_ENV = ('GEMINI_API_KEY', 'GOOGLE_API_KEY')
# Recognised criticality categories (sensitive subjects).
CRITICALITY_CATEGORIES = (
	'celebrity', 'child', 'sensitive_data', 'politician',
	'public_figure', 'violence', 'medical', 'document',
	'alcohol', 'drugs',
)
# How the image was manipulated (best guess). 'none' = looks authentic.
MANIPULATION_TYPES = (
	'face_swap', 'splice_composite', 'inpainting',
	'fully_generated', 'edited', 'none',
)
# Operational decision for the client, from criticality x certainty.
ACTIONS = ('publish', 'human_review', 'block')

_client = None  # lazy singleton


def _get_client():
	global _client
	if _client is not None:
		return _client
	key = next((os.environ[k] for k in _API_ENV if os.environ.get(k)), None)
	if not key:
		raise RuntimeError(f'No Gemini API key found. Set one of {_API_ENV}.')
	try:
		from google import genai
	except ImportError:
		raise RuntimeError(
			'google-genai not installed. Run: pip install google-genai') from None
	_client = genai.Client(api_key=key)
	return _client


# Maior lado enviado ao provider de visão. Cap reduz tokens/custo sem perder
# qualidade para esta tarefa (0 = não redimensiona). Override por env.
CONTEXT_MAX_SIDE = int(os.environ.get('CONTEXT_MAX_SIDE', '1024'))


def _load(img) -> pil.Image:
	out = img if isinstance(img, pil.Image) else pil.open(str(img))
	out = out.convert('RGB')
	# Downscale (cópia, nunca muta a imagem do chamador) antes de enviar ao provider.
	if CONTEXT_MAX_SIDE and max(out.size) > CONTEXT_MAX_SIDE:
		out = out.copy()
		out.thumbnail((CONTEXT_MAX_SIDE, CONTEXT_MAX_SIDE), pil.Resampling.LANCZOS)
	return out


def _generate(image, prompt: str, model: Optional[str], temperature: float) -> str:
	client = _get_client()
	config = None
	try:
		from google.genai import types
		config = types.GenerateContentConfig(
			temperature=temperature,
			response_mime_type='application/json',
			thinking_config=types.ThinkingConfig(thinking_budget=-1),
		)
	except Exception:
		pass
	resp = client.models.generate_content(
		model=model or MODEL, contents=[image, prompt], config=config)
	return getattr(resp, 'text', '') or ''


def _parse_obj(text: str) -> dict:
	"""Best-effort JSON-object extraction from a model response."""
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


_CONTEXT_PROMPT = (
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
	# '  "text_in_image": legible text/watermark/logo via OCR, "" if none.\n'
	'Base manipulation_certainty on visual evidence (blending, warping, lighting/'
	'shadow inconsistency, texture, impossible details). Be calibrated, not extreme.'
)


def normalize(data: dict) -> Dict:
	"""Validate/normalize a raw model dict into the canonical context structure.

	Shared across providers (Gemini, OpenAI, ...) so the output contract is
	identical no matter who generated it.
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
		# 'text_in_image': str(data.get('text_in_image', '')).strip(),  # OCR (disabled)
	}


def log_context(run_name: str, image, result: Dict, track: bool = True) -> None:
	"""Persist received image + generated context as one MLflow run (best-effort).

	Shared by every provider so the storage is identical (image by hash,
	context.json as artifact). Imported lazily to keep this usable without mlflow.
	"""
	if not track:
		return
	try:
		from . import tracking
	except Exception:
		return
	tracking.log_inference(
		run_name,
		params={'manipulation_certainty': result['manipulation_certainty'],
			'criticality': result['criticality']['level']},
		metrics={'manipulation_certainty': result['manipulation_certainty']},
		images={'received.png': image},
		artifacts={'context.json': result},
		tags={'kind': run_name, 'manipulation_type': result['manipulation_type']},
	)


def analyze(generate_fn, run_name: str, img, model, lang: str, track: bool,
		default_model: Optional[str] = None) -> Dict:
	"""Núcleo compartilhado: load → cache por hash → gera → normaliza → loga.

	Se a imagem já tiver um context.json no DB (mesma hash), retorna o cacheado
	e NÃO chama o provider — economiza cota/custo. A origem (provider + modelo)
	fica gravada no resultado, então um cache-hit também sabe quem gerou o laudo.
	`generate_fn(image, prompt, model, temperature)` é a parte específica do provider.
	"""
	image = _load(img)
	try:
		from . import tracking
	except Exception:
		tracking = None
	if tracking is not None:
		sha = tracking.image_sha256(image)
		cached = tracking.find_context(sha) if sha else None
		if cached is not None:
			cached['cached'] = True
			return cached
	model = model or default_model
	raw = generate_fn(image, _CONTEXT_PROMPT.format(lang=lang), model, 0.2)
	result = normalize(_parse_obj(raw))
	result['provider'] = run_name
	result['model'] = model
	result['cached'] = False
	log_context(run_name, image, result, track)
	return result


def context(img, model: Optional[str] = None, lang: str = 'Portuguese', track: bool = True) -> Dict:
	"""Structured contextual analysis of the image, via Gemini (com cache por hash).

	Keys: scene_description (str), coherence ({plausible, opinion}),
	criticality ({level, categories}), manipulation_certainty (float 0-100).
	"""
	return analyze(_generate, 'gemini', img, model, lang, track, default_model=MODEL)
