"""OpenAI (GPT) backed contextual analysis of an image.

Drop-in alternative to `gemini.context` with the SAME output contract — it reuses
the shared prompt, JSON parsing, field normalization and MLflow logging from
`gemini`, so only the API call itself is provider-specific. Use it when the
Gemini free-tier quota runs out, or simply prefer GPT.

Requires `openai` and OPENAI_API_KEY. No torch / model file needed.
"""
from __future__ import annotations

import os, io, base64
from typing import Optional, Dict

from . import gemini

# Carrega .env (mesma estratégia do gemini.py) para OPENAI_API_KEY viver em arquivo.
try:
	from dotenv import load_dotenv, find_dotenv
	load_dotenv(find_dotenv(usecwd=True))
except ImportError:
	pass

__all__ = ['context', 'MODEL']

# Modelo com visão, barato. Troque para 'gpt-4o' em casos difíceis.
MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
_API_ENV = ('OPENAI_API_KEY',)

_client = None  # lazy singleton


def _get_client():
	global _client
	if _client is not None:
		return _client
	key = next((os.environ[k] for k in _API_ENV if os.environ.get(k)), None)
	if not key:
		raise RuntimeError(f'No OpenAI API key found. Set one of {_API_ENV}.')
	try:
		from openai import OpenAI
	except ImportError:
		raise RuntimeError('openai not installed. Run: pip install openai') from None
	_client = OpenAI(api_key=key)
	return _client


def _data_url(image) -> str:
	buf = io.BytesIO()
	image.save(buf, format='PNG')
	return 'data:image/png;base64,' + base64.b64encode(buf.getvalue()).decode()


def _generate(image, prompt: str, model: Optional[str], temperature: float) -> str:
	client = _get_client()
	resp = client.chat.completions.create(
		model=model or MODEL,
		messages=[{
			'role': 'user',
			'content': [
				{'type': 'text', 'text': prompt},
				{'type': 'image_url', 'image_url': {'url': _data_url(image)}},
			],
		}],
		response_format={'type': 'json_object'},  # JSON garantido
		temperature=temperature,
	)
	return resp.choices[0].message.content or ''


def context(img, model: Optional[str] = None, lang: str = 'Portuguese', track: bool = True) -> Dict:
	"""Structured contextual analysis of the image, via OpenAI (GPT).

	Mesmo retorno de `gemini.context` (e mesmo cache por hash): scene_description,
	coherence, criticality, manipulation_certainty, manipulation_type,
	suspect_regions, evidence, recommended_action.
	"""
	return gemini.analyze(_generate, 'openai', img, model, lang, track, default_model=MODEL)
