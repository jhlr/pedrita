"""Leitura de metadados (EXIF) da imagem — um sinal forense independente do
modelo de pixels e do contexto do Gemini.

Não precisa de torch. Aceita bytes, caminho ou PIL.Image. Importante: leia a
imagem ORIGINAL (não a convertida para RGB, que descarta o EXIF).

Além dos campos crus, deriva sinais úteis:
  - has_exif: a imagem tem EXIF? (ausência é comum em imagens geradas por IA,
    capturas de tela ou reenviadas por apps que removem metadados)
  - edited_software: o campo Software aponta um editor (Photoshop, GIMP, ...)?
  - flags: observações legíveis para o usuário final.
"""
from __future__ import annotations

import io
from typing import Any, Dict

from PIL import Image as pil
from PIL import ExifTags

__all__ = ['metadata']

# Softwares de edição conhecidos (lower-case) — presença sugere pós-processamento.
_EDITORS = (
	'photoshop', 'gimp', 'lightroom', 'affinity', 'pixelmator', 'paint.net',
	'snapseed', 'picsart', 'canva', 'illustrator', 'inkscape', 'capture one',
)


def _load(src) -> pil.Image:
	if isinstance(src, pil.Image):
		return src
	if isinstance(src, (bytes, bytearray)):
		return pil.open(io.BytesIO(bytes(src)))
	return pil.open(str(src))


def metadata(src) -> Dict[str, Any]:
	"""Retorna um dict com os metadados e sinais derivados da imagem."""
	img = _load(src)
	w, h = img.size
	out: Dict[str, Any] = {
		'has_exif': False,
		'format': img.format,
		'mode': img.mode,
		'width': w,
		'height': h,
		'megapixels': round((w * h) / 1e6, 2),
		'camera': {'make': None, 'model': None, 'lens': None},
		'software': None,
		'datetime': None,
		'gps': False,
		'orientation': None,
		'edited_software': False,
		'flags': [],
	}

	try:
		exif = img.getexif()
	except Exception:
		exif = None

	if not exif or len(exif) == 0:
		out['flags'].append(
			'Sem metadados EXIF (comum em imagens geradas por IA, capturas de tela '
			'ou reenviadas por apps de mensagem)')
		return out

	out['has_exif'] = True
	tags = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}

	def g(name):
		v = tags.get(name)
		v = str(v).strip() if v not in (None, '') else None
		return v or None

	out['camera']['make'] = g('Make')
	out['camera']['model'] = g('Model')
	out['camera']['lens'] = g('LensModel')
	out['software'] = g('Software')
	out['datetime'] = g('DateTimeOriginal') or g('DateTime')

	ori = tags.get('Orientation')
	out['orientation'] = int(ori) if isinstance(ori, int) else None

	try:
		out['gps'] = bool(exif.get_ifd(ExifTags.IFD.GPSInfo))
	except Exception:
		out['gps'] = False

	if out['software']:
		low = out['software'].lower()
		if any(e in low for e in _EDITORS):
			out['edited_software'] = True
			out['flags'].append(f'Editada em software: {out["software"]}')

	if not out['camera']['make'] and not out['camera']['model']:
		out['flags'].append('EXIF presente, porém sem dados de câmera (make/model)')

	return out
