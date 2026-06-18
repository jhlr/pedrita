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


def _load(img) -> pil.Image:
	out = img if isinstance(img, pil.Image) else pil.open(str(img))
	return out.convert('RGB')


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
	'"violence","medical","document"] that apply (empty if none).\n'
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


def context(img, model: Optional[str] = None, lang: str = 'Portuguese') -> Dict:
	"""Return a structured contextual analysis of the image.

	Keys: scene_description (str), coherence ({plausible, opinion}),
	criticality ({level, categories}), manipulation_certainty (float 0-100).
	"""
	image = _load(img)
	raw = _generate(image, _CONTEXT_PROMPT.format(lang=lang), model, 0.2)
	data = _parse_obj(raw)

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

	result = {
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

	# Persist received image + generated context as an MLflow run (best-effort;
	# imported lazily so this module stays usable without torch/mlflow installed).
	try:
		from . import tracking
	except Exception:
		tracking = None
	if tracking is not None:
		tracking.log_inference(
			'gemini',
			params={'manipulation_certainty': certainty, 'criticality': level},
			metrics={'manipulation_certainty': certainty},
			images={'received.png': image},
			artifacts={'context.json': result},
			tags={'kind': 'gemini', 'manipulation_type': mtype},
		)

	return result
