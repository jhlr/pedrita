"""Provider de contexto via Gemini.

Implementa apenas a chamada à API do Gemini e delega prompt, parsing,
normalização, cache e logging ao núcleo compartilhado em `context_base`. O
contrato de saída é idêntico ao de qualquer outro provider (ver `context_base`).

Requer `google-genai` e GEMINI_API_KEY (ou GOOGLE_API_KEY); o .env é carregado
pelo `context_base`. Não precisa de torch / arquivo de modelo.
"""
from __future__ import annotations

import os
from typing import Optional, Dict

from . import context_base as base

__all__ = ['context', 'MODEL']

# Modelo default: rápido + barato. Troque para '-pro' em casos difíceis.
MODEL = os.environ.get('GEMINI_MODEL', 'gemini-2.5-flash')
_API_ENV = ('GEMINI_API_KEY', 'GOOGLE_API_KEY')

_client = None  # singleton preguiçoso


def _get_client():
	global _client
	if _client is not None:
		return _client
	key = base.resolve_key(_API_ENV)
	if not key:
		raise RuntimeError(f'No Gemini API key found. Set one of {_API_ENV}.')
	try:
		from google import genai
	except ImportError:
		raise RuntimeError(
			'google-genai not installed. Run: pip install google-genai') from None
	_client = genai.Client(api_key=key)
	return _client


def _generate(image, prompt: str, model: Optional[str], temperature: float) -> str:
	"""Parte específica do provider: uma chamada ao Gemini, retornando texto cru."""
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


def context(img, model: Optional[str] = None, lang: str = 'Portuguese', track: bool = True) -> Dict:
	"""Análise contextual estruturada da imagem, via Gemini (com cache por hash).

	Mesmo retorno de qualquer provider — ver `context_base` para o contrato.
	"""
	return base.analyze(_generate, 'gemini', img, model, lang, track, default_model=MODEL)
