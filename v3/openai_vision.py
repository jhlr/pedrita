"""Provider de contexto via OpenAI (GPT).

Alternativa drop-in ao `gemini`, com o MESMO contrato de saída: implementa apenas
a chamada à API da OpenAI e delega prompt, parsing, normalização, cache e logging
ao núcleo compartilhado em `context_base`. Use quando a cota gratuita do Gemini
esgotar, ou simplesmente por preferir o GPT.

Requer `openai` e OPENAI_API_KEY; o .env é carregado pelo `context_base`. Não
precisa de torch / arquivo de modelo.
"""
from __future__ import annotations

import os, io, base64
from typing import Optional, Dict

from . import context_base as base

__all__ = ['context', 'MODEL']

# Modelo com visão, barato. Troque para 'gpt-4o' em casos difíceis.
MODEL = os.environ.get('OPENAI_MODEL', 'gpt-4o-mini')
_API_ENV = ('OPENAI_API_KEY',)

_client = None  # singleton preguiçoso


def _get_client():
	global _client
	if _client is not None:
		return _client
	key = base.resolve_key(_API_ENV)
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
	"""Parte específica do provider: uma chamada ao GPT, retornando texto cru."""
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
	"""Análise contextual estruturada da imagem, via OpenAI/GPT (com cache por hash).

	Mesmo retorno de qualquer provider — ver `context_base` para o contrato.
	"""
	return base.analyze(_generate, 'openai', img, model, lang, track, default_model=MODEL)
