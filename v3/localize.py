# """Pluggable localizers and N-source fusion.
# 
# A Localizer turns an image into a (score_fake, heatmap) pair, where:
#   - score_fake : float in [0,1], higher = more likely manipulated
#   - heatmap    : np.ndarray [H,W] in [0,1] (hot = suspect) OR None
# 
# Every detector — the local CNN (`mother`) or a HuggingFace classifier — implements
# the same contract, so `fuse()` can overlay any combination of them. Adding a new
# model is a new class + a registry entry; the rest of v3 is untouched.
# 
# HuggingFace classifiers are pure detectors (verdict only). To still get a heatmap
# from them we use occlusion sensitivity: mask a grid of regions and measure the
# drop in the "fake" score. It is architecture-agnostic (works for ViT, Swin, ...),
# so it sidesteps the model-specific reshaping that Grad-CAM needs on transformers.
# """
# from __future__ import annotations
# 
# import os
# from typing import Protocol, Sequence, Optional, List, Tuple, Dict, runtime_checkable
# 
# import numpy as np
# from PIL import Image as pil
# 
# try: from . import helper, predict, train
# except ImportError:
# 	import helper, predict, train  # type: ignore
# 
# __all__ = [
# 	'Localizer', 'MotherLocalizer', 'EnsembleLocalizer', 'HFDetectorLocalizer',
# 	'fuse', 'build',
# ]
# 
# # Curated 5-model ensemble: best balanced set across dataset/ (general) and
# # faces/ (in-domain) — ~91% / ~98% on a 60/class sample, beating any single model.
# ENSEMBLE_DEFAULT = ['mother', 'ultra_1', 'ultra_6', 'ultra_8', 'pedrita7']
# 
# # Words/codes marking the "manipulated" vs the "authentic" class in id2label.
# _FAKE_WORDS = ('fake', 'deepfake', 'artificial', 'synthetic', 'generated', 'gan', 'ai')
# _REAL_WORDS = ('real', 'human', 'realism', 'authentic', 'genuine', 'pristine', 'nature')
# 
# 
# def _fake_index(id2label: dict) -> int:
# 	"""Find which class index means 'manipulated'. Handles word labels and the
# 	terse 'f'/'r' coding; falls back to the opposite of the detected real class."""
# 	items = {i: str(l).strip().lower() for i, l in id2label.items()}
# 	for i, l in items.items():
# 		if l == 'f' or any(w in l for w in _FAKE_WORDS):
# 			return i
# 	# no explicit fake label: infer it as the non-real class (binary case)
# 	real = [i for i, l in items.items() if l == 'r' or any(w in l for w in _REAL_WORDS)]
# 	if real and len(items) == 2:
# 		return next(i for i in items if i not in real)
# 	return len(items) - 1
# 
# 
# def _load(img) -> pil.Image:
# 	out = img if isinstance(img, pil.Image) else pil.open(str(img))
# 	return out.convert('RGB')
# 
# 
# @runtime_checkable
# class Localizer(Protocol):
# 	name: str
# 	def analyze(self, img) -> Tuple[float, Optional[np.ndarray]]:
# 		"""Return (score_fake in [0,1], heatmap [H,W] in [0,1] or None).
# 
# 		A pure detector may return None for the heatmap: it then contributes only
# 		to the fused verdict, not to the localization overlay.
# 		"""
# 		...
# 
# 
# class MotherLocalizer:
# 	"""The existing CNN + Grad-CAM, wrapped behind the Localizer contract."""
# 	name = 'mother'
# 
# 	def __init__(self, model_path: Optional[str] = None):
# 		if model_path is not None:
# 			helper.set_model(model_path)
# 
# 	def analyze(self, img) -> Tuple[float, Optional[np.ndarray]]:
# 		prob_real, _overlay, grey = predict.heatmap(img, minmax=True, return_grey=True)
# 		return 1.0 - float(prob_real), np.clip(grey.astype(np.float32), 0.0, 1.0)
# 
# 
# class EnsembleLocalizer:
# 	"""Orchestrate several of your own CNNs: average their verdicts and heatmaps.
# 
# 	Each model votes with its prob_real; the mean drives the verdict and the mean
# 	Grad-CAM drives the heatmap. Diverse-but-individually-strong models cover each
# 	other's blind spots across distributions (general vs faces).
# 	"""
# 	name = 'ensemble'
# 
# 	def __init__(self, models: Optional[Sequence[str]] = None,
# 		models_dir: str = 'models', heatmap: bool = True):
# 		names = list(models) if models else list(ENSEMBLE_DEFAULT)
# 		self.paths = [m if os.path.sep in m or m.endswith('.pkl')
# 			else os.path.join(models_dir, f'{m}.pkl') for m in names]
# 		self.heatmap = heatmap
# 		self._models = None
# 
# 	def _ensure(self):
# 		if self._models is not None:
# 			return
# 		import joblib
# 		# Disable online-training side effects so eval never mutates/saves models.
# 		train.online_training(None, None)  # type: ignore
# 		self._models = [joblib.load(p) for p in self.paths]
# 
# 	def analyze(self, img) -> Tuple[float, Optional[np.ndarray]]:
# 		self._ensure()
# 		image = _load(img)
# 		probs: List[float] = []
# 		greys: List[np.ndarray] = []
# 		for mdl in self._models:
# 			helper.model = mdl
# 			if self.heatmap:
# 				pr, _overlay, grey = predict.heatmap(image, minmax=True, return_grey=True)
# 				greys.append(np.clip(grey.astype(np.float32), 0.0, 1.0))
# 			else:
# 				pr = predict.predict([image])[0]
# 			probs.append(float(pr))
# 		score_fake = 1.0 - float(np.mean(probs))
# 		heat = np.clip(np.mean(greys, axis=0), 0.0, 1.0) if greys else None
# 		return score_fake, heat
# 
# 
# class HFDetectorLocalizer:
# 	"""[DEPRECATED] A HuggingFace image classifier as a detector (+ occlusion heatmap).
# 
# 	Deprecated: the downloaded HF detectors (deepfake/sdxl/light) underperform the
# 	in-domain CNNs and don't generalize to splicing (see relatorio.md). Prefer the
# 	EnsembleLocalizer plus a ciplab fine-tune. Kept only for reference/comparison.
# 
# 	`repo` is any AutoModelForImageClassification checkpoint whose labels separate
# 	real from manipulated. The "fake" class is found by label keywords. The verdict
# 	is a single forward pass; the occlusion heatmap costs grid*grid extra forwards
# 	and is OFF by default (grid=0).
# 	"""
# 
# 	def __init__(self, repo: str, name: Optional[str] = None, grid: int = 0):
# 		import warnings
# 		warnings.warn(
# 			'HFDetectorLocalizer (deepfake/sdxl/light) is deprecated: the downloaded '
# 			'HF detectors underperform the in-domain CNNs; prefer EnsembleLocalizer '
# 			'and a ciplab fine-tune.',
# 			DeprecationWarning, stacklevel=2)
# 		self.repo = repo
# 		self.name = name or repo.split('/')[-1]
# 		self.grid = grid
# 		self._model = self._proc = self._fake_idx = None
# 
# 	def _ensure(self):
# 		if self._model is not None:
# 			return
# 		import torch
# 		from transformers import AutoModelForImageClassification, AutoImageProcessor
# 		self._torch = torch
# 		self._device = helper.best_device()
# 		self._proc = AutoImageProcessor.from_pretrained(self.repo)
# 		self._model = AutoModelForImageClassification.from_pretrained(self.repo)
# 		self._model.eval().to(self._device)
# 		self._fake_idx = _fake_index(self._model.config.id2label)
# 
# 	def _scores(self, images: Sequence[pil.Image]) -> np.ndarray:
# 		"""Return the fake-class probability for each image."""
# 		torch = self._torch
# 		inputs = self._proc(images=list(images), return_tensors='pt').to(self._device)
# 		with torch.no_grad():
# 			logits = self._model(**inputs).logits
# 			probs = logits.softmax(dim=1)[:, self._fake_idx]
# 		return probs.detach().cpu().numpy()
# 
# 	def _occlusion(self, image: pil.Image, base: float) -> Optional[np.ndarray]:
# 		g = self.grid
# 		if g <= 0:
# 			return None
# 		w, h = image.size
# 		cw, ch = w / g, h / g
# 		variants, cells = [], []
# 		for i in range(g):
# 			for j in range(g):
# 				box = (int(j * cw), int(i * ch), int((j + 1) * cw), int((i + 1) * ch))
# 				im = image.copy()
# 				im.paste((127, 127, 127), box)
# 				variants.append(im); cells.append((i, j))
# 		occ = self._scores(variants)
# 		heat = np.zeros((g, g), dtype=np.float32)
# 		for (i, j), s in zip(cells, occ):
# 			heat[i, j] = max(0.0, base - float(s))  # how much masking it lowers "fake"
# 		m = heat.max()
# 		if m <= 0:
# 			return None
# 		heat /= m
# 		import cv2
# 		return cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
# 
# 	def analyze(self, img) -> Tuple[float, Optional[np.ndarray]]:
# 		self._ensure()
# 		image = _load(img)
# 		score = float(self._scores([image])[0])
# 		heat = None
# 		try:
# 			heat = self._occlusion(image, score)
# 		except Exception:
# 			heat = None
# 		return score, heat
# 
# 
# def fuse(img, localizers: Sequence[Localizer],
# 	weights: Optional[Sequence[float]] = None, alpha: float = 1.0) -> Dict:
# 	"""Run each localizer; blend verdicts and overlay available heatmaps.
# 
# 	Score and heatmap are fused independently: a detector that returns None for
# 	its heatmap still counts toward the verdict. Sources that raise are skipped so
# 	one failing model doesn't sink the others.
# 	"""
# 	from pytorch_grad_cam.utils.image import show_cam_on_image
# 	import cv2
# 	image = _load(img)
# 	w, h = image.size
# 	if weights is None:
# 		weights = [1.0] * len(localizers)
# 
# 	heats, heat_w, scores, score_w, per = [], [], [], [], []
# 	for lz, wt in zip(localizers, weights):
# 		try:
# 			s, g = lz.analyze(img)
# 			scores.append(float(s)); score_w.append(float(wt))
# 			has_heat = g is not None
# 			if has_heat:
# 				g = cv2.resize(np.asarray(g, dtype=np.float32), (w, h))
# 				heats.append(np.clip(g, 0.0, 1.0)); heat_w.append(float(wt))
# 			per.append({'name': lz.name, 'score_fake': float(s),
# 				'heatmap': has_heat, 'ok': True})
# 		except Exception as e:
# 			per.append({'name': lz.name, 'ok': False, 'error': f'{type(e).__name__}: {e}'})
# 
# 	if not scores:
# 		return {'score_fake': None, 'heatmap': image, 'sources': per}
# 
# 	sw = np.array(score_w, dtype=np.float32); sw /= sw.sum()
# 	fused_score = float((sw * np.array(scores)).sum())
# 
# 	if heats:
# 		hw = np.array(heat_w, dtype=np.float32); hw /= hw.sum()
# 		grey = np.clip(sum(wt * g for wt, g in zip(hw, heats)), 0.0, 1.0)
# 		overlay = show_cam_on_image(np.asarray(image) / 255.0, grey, use_rgb=True)
# 		out = pil.fromarray(overlay)
# 		if alpha < 1.0:
# 			out = pil.blend(image, out, alpha)
# 	else:
# 		out = image
# 	return {'score_fake': fused_score, 'heatmap': out, 'sources': per}
# 
# 
# # name -> factory; keeps the CLI declarative ("mother,ensemble").
# # deepfake/sdxl/light are DEPRECATED (downloaded HF detectors, kept for comparison).
# _REGISTRY = {
# 	'mother': lambda **kw: MotherLocalizer(),
# 	'ensemble': lambda **kw: EnsembleLocalizer(models=kw.get('ensemble')),
# 	# --- deprecated HF detectors ---
# 	'deepfake': lambda **kw: HFDetectorLocalizer(
# 		'prithivMLmods/Deep-Fake-Detector-v2-Model', name='deepfake',
# 		grid=kw.get('grid', 0)),
# 	'sdxl': lambda **kw: HFDetectorLocalizer(
# 		'Organika/sdxl-detector', name='sdxl', grid=kw.get('grid', 0)),
# 	'light': lambda **kw: HFDetectorLocalizer(
# 		'Skullly/DeepFake-EN-B6', name='light', grid=kw.get('grid', 0)),
# }
# 
# 
# def build(names: Sequence[str], **kw) -> List[Localizer]:
# 	"""Instantiate localizers by name. Unknown names raise."""
# 	out: List[Localizer] = []
# 	for n in names:
# 		n = n.strip().lower()
# 		if not n: continue
# 		if n not in _REGISTRY:
# 			raise ValueError(f'unknown localizer {n!r}; known: {list(_REGISTRY)}')
# 		out.append(_REGISTRY[n](**kw))
# 	return out
