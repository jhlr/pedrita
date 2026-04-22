import os, kagglehub as kag
from typing import Sequence, cast, Callable
from glob import glob
import pathlib
import numpy as np
import torch, timm, joblib
from numpy.typing import NDArray

from PIL import Image as image
from PIL.Image import Image

import torch.nn as nn
from torch import Tensor
from torchvision import transforms as tforms
from torchvision.transforms.functional import to_pil_image
# type hint for joblib
from timm.models import EfficientNet 

DATASET = 'tristanzhang32/ai-generated-images-vs-real-images'

num_classes = 2
model: nn.Module = None # type: ignore
device: torch.device = None # type: ignore
retrained = False
BASE_DIR = os.getcwd()

def set_model(model_name: str, /, *, force=False, prefix: str = '') -> torch.nn.Module:
	global model, num_classes
	# `num_classes` is a global set to 3; do not accept an override here.
	prefix = prefix or 'models/'
	prefix = os.path.join(BASE_DIR, prefix)
	fnames = [ model_name,
		os.path.join(prefix, f'{model_name}_c{num_classes}'),
		os.path.join(prefix, f'{model_name}_c{num_classes}.pkl'),
		os.path.join(prefix, f'{model_name}_c{num_classes}.joblib'),
		os.path.join(prefix, f'{model_name}'),
		os.path.join(prefix, f'{model_name}.pkl'), 
		os.path.join(prefix, f'{model_name}.joblib'),
	]
	for name in fnames:
		try:
			model = joblib.load(name)
			print(name)
			model.to(best_device())
			return model
		except Exception: continue
	if not force:
		raise FileNotFoundError(f"Modelo '{model_name}' não encontrado em {fnames}")
	model = timm.create_model(
		model_name, pretrained=True, 
		num_classes=num_classes)
	joblib.dump(model, f'{prefix}{model_name}_c{num_classes}.pkl')
	print(f"Modelo pré-treinado {model_name} salvo como {prefix}{model_name}_c{num_classes}.pkl")
	return model

def transform(train: bool = False) -> Callable[[Image], Tensor]:
	"""Return a torchvision transform.

	If `train` is True, include common augmentations useful for transfer
	learning. When False, use a deterministic evaluation transform.
	"""
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

	return tforms.Compose([
		tforms.Lambda(to_pil),
		tforms.ToTensor(),
		tforms.Resize(size),
		tforms.RandomPerspective(distortion_scale=0.2, p=0.5),
		tforms.RandomHorizontalFlip(p=0.5),
		tforms.RandomVerticalFlip(p=0.5),
		tforms.RandomAffine(degrees=10, translate=(0.05,0.05), scale=(0.95,1.05), shear=5),
		tforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
		tforms.Normalize(mean=cfg['mean'], std=cfg['std'])
	]) if train else tforms.Compose([
		tforms.Lambda(to_pil),
		tforms.ToTensor(),
		tforms.Resize(size),
		tforms.Normalize(mean=cfg['mean'], std=cfg['std'])
	])

def best_device() -> torch.device:
	global device
	if device is not None:
		return device
	# 1. NVIDIA (Padrão ouro para Deep Learning)
	if torch.cuda.is_available():
		device_name = torch.cuda.get_device_name(0)
		print(f"NVIDIA Detectada: {device_name}")
		device = torch.device("cuda")
		return device
	# 2. Apple Silicon (M1, M2, M3 - Seu cenário atual)
	elif torch.backends.mps.is_available():
		print("Apple Silicon (MPS) Detectada.")
		device = torch.device("mps")
		return device

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

def to_pil(img) -> Image:
	if isinstance(img, str):
		img = image.open(img)
	if not isinstance(img, Image):
		return cast(Image, to_pil_image(img))
	# Handle palette images with transparency (P mode with tRNS) and other
	# alpha-bearing formats. Convert such images to RGBA first to avoid the
	# PIL user warning, then composite onto a white background and return RGB.
	if img.mode == 'P':
		# PIL uses 'transparency' info for palette-based alpha
		if 'transparency' in getattr(img, 'info', {}):
			img = img.convert('RGBA')
			bg = image.new('RGBA', img.size, (255, 255, 255, 255))
			img = image.alpha_composite(bg, img).convert('RGB')
		else:
			img = img.convert('RGB')
	elif img.mode in ('RGBA', 'LA'):
		# Composite alpha over white background
		bg = image.new('RGBA', img.size, (255, 255, 255, 255))
		img = image.alpha_composite(bg, img.convert('RGBA')).convert('RGB')
	elif img.mode != 'RGB':
		img = img.convert('RGB')
	return img

pixel = np.uint8

# kaggle_download('train/fake', 401, 500)
# ou 'train/real', 'test/fake', 'test/real'
def kaggle_download(folder:str, first:int, last:int, fext=['jpg', 'png', 'jpeg']):
	if isinstance(fext, str):
		fext = fext.lstrip('.')
		fext = [fext]
	if isinstance(fext, list | tuple):
		fext = [x.lstrip('.') for x in fext]
	for i in range(first, last+1):
		for x in range(len(fext)):
			try:
				fname = f'{folder}/{i:04d}.{fext[x]}'
				if os.path.exists(f'./dataset/{fname}'): break
				fpath = kag.dataset_download(DATASET, fname)
				os.rename(fpath, f'./dataset/{fname}')
				break
			except Exception: pass
	
def compare(
	p_final: NDArray[np.float64] | Sequence[float], 
	y_test: NDArray[np.float64] | Sequence[float], thresh=0.6
) -> NDArray[np.float64]:
	'''Print simple evaluation statistics and return the probabilities.

	Metrics printed are percentages of wrong/sure/dunno relative to the
	total number of samples.
	'''
	p_final = np.asarray(p_final)
	y_test = np.array(y_test)

	is_real = y_test >= 0.999
	is_fake = y_test <= 0.001
	is_high = p_final > thresh
	is_low = p_final < (1 - thresh)

	# WRONG: final verdict strongly on wrong side
	w_total = np.sum(is_fake & is_high) + np.sum(is_real & is_low)
	# SURE: strong correct verdicts
	s_total = np.sum(is_real & is_high) + np.sum(is_fake & is_low)
	# DUNNO: remaining in gray zone
	d_total = np.sum(~(is_high | is_low))
	total = len(y_test)

	pct = lambda x: x * 100.0 / total if total else 0.0

	print(f'Total: {total}')
	print(f'Wrong: {pct(w_total):2.1f}%')
	print(f'Sure:  {pct(s_total):2.1f}%')
	print(f'Dunno: {pct(d_total):2.1f}%')
	# return original probabilities array for compatibility
	return p_final


import atexit
@atexit.register
def save_model_on_exit():
	global model, retrained
	if model is None or not retrained:
		return
	os.makedirs('models', exist_ok=True)
	mpath = 'models/model_temp.pkl'
	joblib.dump(model, mpath)
	print(mpath)



