import joblib
from pathlib import Path
from typing import cast
from PIL import Image as pil
from datetime import datetime as dt
from functools import wraps

from collections.abc import Callable, Sequence
import torch
from torch.nn import Module
from torch import Tensor
import torch.nn.functional as F
from readerwriterlock import rwlock

from torchvision import transforms as tforms
from torchvision.transforms.functional import to_pil_image

from glob import glob
from sklearn.model_selection import train_test_split

try: from .dset import DirDataset
except ImportError:
	from dset import DirDataset

try:
	import logging
	class MLflowPickleWarningFilter(logging.Filter):
		def filter(self, record):
			# Drop the log if it contains the word "Pickle" and "CloudPickle"
			return "Pickle or CloudPickle format requires" not in record.getMessage()
	# Apply the filter to the target MLflow logger
	logging.getLogger("mlflow.pytorch").addFilter(MLflowPickleWarningFilter())
	logging.getLogger("mlflow").addFilter(MLflowPickleWarningFilter())
except Exception: pass

__all__ = [
	'LABEL', 'num_classes',
	'set_model', 'save_model',
	'freeze', 'best_device', 'timer',
	'transform', 'to_pil', 'expand',
	'compare', 'DirDataset', 'seed_all',
]

import os, random

# Default seed for reproducibility; override with PEDRITA_SEED.
SEED = int(os.environ.get('PEDRITA_SEED', 42))


def seed_all(seed: int | None = None) -> int:
	"""Seed Python/NumPy/torch RNGs for reproducible runs. Returns the seed used."""
	import numpy as np
	if seed is None:
		seed = SEED
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	return int(seed)

# Match training mapping: 0 = fake, 1 = real
LABEL_NAMES = ['fake', 'real', 'other']
# map names to class indices used by the model: fake=0, real=1, other=2
LABEL: dict[str, int] = {n: i for i, n in enumerate(LABEL_NAMES)}
# for name, idx in LABEL.items(): setattr(LABEL, name, idx)

model: Module = None # type: ignore
device: torch.device = None # type: ignore
retrained: bool = False
force_cpu: bool = False

marker = rwlock.RWLockFair()
rlock = marker.gen_rlock()
wlock = marker.gen_wlock()

def num_classes() -> int:
	global model
	"""Return the number of classes the current model is configured for."""
	if model is None:
		raise RuntimeError('Model not set. Call set_model() first.')
	return int(model.num_classes) # type: ignore

def set_model(
	model_name=None, 
	/, force: bool = False,
	nclasses = num_classes, 
) -> Module:
	global model, device
	if model_name is None: return model
	best_device()
	if isinstance(model_name, Module):
		model = model_name
		model.to(device)
		model.eval()
		return model
	loaded = None
	try:
		loaded = joblib.load(model_name)
	except Exception as e:
		print(f'failed to load {model_name}: {e}')
	# If file contains a module instance
	if isinstance(loaded, Module):
		model = loaded
		print(model_name)
		model.to(device)
		model.eval()
		return model
	if not force:
		raise FileNotFoundError(f"Modelo '{model_name}' não encontrado em {model_name}") from None
	import timm
	model = timm.create_model(
		model_name, pretrained=True, 
		num_classes=nclasses)
	joblib.dump(model, f'{model_name}_c{nclasses}.pkl')
	return model

def freeze(lim: int=0):
	global model
	if not lim: 
		for p in model.parameters():
			p.requires_grad = lim != None
		return model.parameters()
	neg = lim < 0
	if neg: lim = -lim
	children = list(model.children())
	children = list(reversed(children))
	for c, child in enumerate(children):
		trainable = (c < lim) if neg else (c >= lim)
		for p in child.parameters():
			p.requires_grad = trainable
	return [ p for p in model.parameters() if p.requires_grad ]

def best_device(cpu=False) -> torch.device:
	global device, force_cpu
	# honor explicit env or programmatic override
	if cpu or force_cpu:
		device = torch.device('cpu')
		print("CPU forçado por configuração.")
		force_cpu = True
		return device
	elif device is not None:
		return device
	# 1. NVIDIA (Padrão ouro para Deep Learning)
	if torch.cuda.is_available():
		device_name = torch.cuda.get_device_name(0)
		print(f"NVIDIA Detectada: {device_name}")
		device = torch.device("cuda")
		return device
	# 2. Apple Silicon (M1, M2, M3 - Seu cenário atual)
	# 2. Apple Silicon (M1, M2, M3 - Seu cenário atual)
	# Guard against environments where `torch.backends.mps` may not exist
	try:
		mps_backend = getattr(torch.backends, 'mps', None)
		if mps_backend is not None and getattr(mps_backend, 'is_available', lambda: False)():
			print("Apple Silicon (MPS) Detectada.")
			device = torch.device("mps")
			return device
	except Exception:
		# Any unexpected issue checking MPS should fall through to other backends
		pass

	# 3. AMD / Intel via DirectML (Comum em Windows/Laptops sem NVIDIA)
	# Requer: pip install torch-directml
	try:
		import torch_directml # type: ignore
		if torch_directml.is_available():
			print("AMD/Intel (DirectML) Detectada.")
			device = torch_directml.device()
			return device
	except ImportError: pass

	# 4. Intel XPU (Específico para placas Intel Arc / Data Centers)
	if hasattr(torch, 'xpu') and torch.xpu.is_available():
		print("Intel XPU Detectada.")
		device = torch.device("xpu")
		return device
	device = torch.device("cpu")
	return device

# # kaggle_download('train/fake', 401, 500)
# # ou 'train/real', 'test/fake', 'test/real'
# def kaggle_download(folder:str, first:int, last:int, fext=['jpg', 'png', 'jpeg']):
# 	import kagglehub as kag
# 	if isinstance(fext, str):
# 		fext = fext.lstrip('.')
# 		fext = [fext]
# 	if isinstance(fext, list | tuple):
# 		fext = [x.lstrip('.') for x in fext]
# 	for i in range(first, last+1):
# 		for x in range(len(fext)):
# 			try:
# 				fname = f'{folder}/{i:04d}.{fext[x]}'
# 				if os.path.exists(f'./dataset/{fname}'): break
# 				fpath = kag.dataset_download(DATASET, fname)
# 				os.rename(fpath, f'./dataset/{fname}')
# 				break
# 			except Exception: pass

def to_pil(img) -> pil.Image | Tensor:
	if isinstance(img, Tensor):
		return img
	if isinstance(img, str):
		img = pil.open(img)
	if not isinstance(img, pil.Image):
		img = cast(pil.Image, to_pil_image(img))
	# Handle palette images with transparency (P mode with tRNS) and other
	# alpha-bearing formats. Convert such images to RGBA first to avoid the
	# PIL user warning, then composite onto a white background and return RGB.
	is_p = img.mode == 'P' and 'transparency' in getattr(img, 'info', {})
	if img.mode in ['RGBA', 'LA'] or is_p:
		# Composite alpha over white background
		bg = pil.new('RGBA', img.size, (255, 255, 255, 255))
		img = pil.alpha_composite(bg, img.convert('RGBA'))
	if img.mode != 'RGB':
		img = img.convert('RGB')
	return img

def transform(train: bool = False, norm: bool = True) -> Callable[..., Tensor]:
	"""Return a torchvision transform.

	If `train` is True, include common augmentations useful for transfer
	learning. When False, use a deterministic evaluation transform.
	"""
	global model
	cfg = getattr(model, 'default_cfg', dict(
		input_size=(224, 224),
		mean=(0.485, 0.456, 0.406),
		std=(0.229, 0.224, 0.225),
	))
	orig = cfg['input_size']
	isize = orig[:]
	for i, v in enumerate(orig):
		if v < 10:
			isize = isize[:i] + isize[i+1:]

	# `isize` should now be a tuple like (H, W) or a single int tuple
	size = isize if isinstance(isize, (tuple, list)) else (isize, isize)
	compose = [
		tforms.Lambda(to_pil),
		tforms.Resize(size),
		tforms.ToTensor(),
	]
	if norm: compose.append(tforms.Normalize(
		mean=cfg['mean'], std=cfg['std'],
	)) # type: ignore
	if train: compose[3:3] = [ # type: ignore
		tforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
		tforms.RandomRotation(degrees=15),
		tforms.RandomHorizontalFlip(p=0.5),
		tforms.RandomVerticalFlip(p=0.5),
		tforms.ColorJitter(
			brightness=0.5, contrast=0.5,
			saturation=0.5, hue=0.2 ),
	]
	return tforms.Compose(compose)

def expand(logits: Tensor) -> Tensor:
	if logits.shape[1] == 1:
		return F.pad(logits, (1, 0), value=0)
	return logits

def compare(
	p_final: Tensor | Sequence[float],
	y_test: Tensor | Sequence[float],
	thresh=0.7,
) -> tuple[Tensor, float, dict]:
	'''Print evaluation statistics and return (residuals, accuracy, stats).

	Printed/returned metrics are fractions of wrong/sure/dunno relative to the
	total number of samples. `stats` is a flat dict suitable for MLflow logging.
	'''
	if thresh > 0.5: thresh = 1 - thresh
	p_final = Tensor(p_final)
	y_test = Tensor(y_test)

	is_real = y_test >= 0.999
	is_fake = y_test <= 0.001
	is_high = p_final > (1 - thresh)
	is_low = p_final < thresh

	# WRONG: final verdict strongly on wrong side
	w_f = torch.sum(is_fake & is_high)
	w_r = torch.sum(is_real & is_low)
	# SURE: strong correct verdicts
	s_r = torch.sum(is_real & is_high)
	s_f = torch.sum(is_fake & is_low)
	# DUNNO: remaining in gray zone
	dunno = ~(is_high | is_low)
	d_r = torch.sum(is_real & dunno)
	d_f = torch.sum(is_fake & dunno)
	total = max(len(y_test), 1)
	print(f'Total {total}:')
	print(f'  Label\t', f'W', f'S', f'D{thresh*100:0.0f}%', sep='\t')
	print('  ------------------------------------')
	print(f'  Real\t', f'{w_r/total:2.1%}', f'{s_r/total:2.1%}', f'{d_r/total:2.1%}', sep='\t')
	print(f'  Fake\t', f'{w_f/total:2.1%}', f'{s_f/total:2.1%}', f'{d_f/total:2.1%}', sep='\t')
	s_total = (s_f + s_r) / total
	print('  ------------------------------------')
	print(f'  Overall', f'{(w_f+w_r)/total:2.1%}', f'{s_total:2.1%}', f'{(d_f+d_r)/total:2.1%}', sep='\t')
	stats = {
		'accuracy': s_total.item(),
		'wrong': ((w_f + w_r) / total).item(),
		'dunno': ((d_f + d_r) / total).item(),
		'sure_real': (s_r / total).item(), 'wrong_real': (w_r / total).item(),
		'dunno_real': (d_r / total).item(),
		'sure_fake': (s_f / total).item(), 'wrong_fake': (w_f / total).item(),
		'dunno_fake': (d_f / total).item(),
	}
	return p_final - y_test, s_total.item(), stats

def save_model(mpath: str | Path) -> Module:
	"""Save the given model to disk with the specified name."""
	global model, device
	cpu = torch.device('cpu')
	m = model.to(cpu)
	joblib.dump(m, mpath)
	model.to(device)
	print(mpath)
	return model

import atexit
@atexit.register
def save_on_exit():
	path = Path.cwd() / 'models' 
	path.mkdir(parents=True, exist_ok=True)
	path /= 'model_temp.pkl'
	if model is not None and retrained:
		save_model(path)

# decorator for all functions to not spam error message on keyboard interrupt
def timer(func):
	def wrapper(*args, **kwargs):
		now = dt.now()
		try: return func(*args, **kwargs)
		finally: print(dt.now() - now)
	return wraps(func)(wrapper)

def split_folder(folder: Path):
	real = glob(str(folder / 'real' / '*'))
	fake = glob(str(folder / 'fake' / '*'))
	real_train, real_test = train_test_split(real, test_size=0.2)
	fake_train, fake_test = train_test_split(fake, test_size=0.2)
	for f in real_train:
		sub = folder / 'train' / 'real'
		sub.mkdir(parents=True, exist_ok=True)
		f = Path(f)
		f.rename(sub / f.name)
	for f in real_test:
		sub = folder / 'test' / 'real'
		sub.mkdir(parents=True, exist_ok=True)
		f = Path(f)
		f.rename(sub / f.name)
	for f in fake_train:
		sub = folder / 'train' / 'fake'
		sub.mkdir(parents=True, exist_ok=True)
		f = Path(f)
		f.rename(sub / f.name)
	for f in fake_test:
		sub = folder / 'test' / 'fake'
		sub.mkdir(parents=True, exist_ok=True)
		f = Path(f)
		f.rename(sub / f.name)