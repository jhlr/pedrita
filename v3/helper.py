import os, kagglehub as kag
from typing import Sequence, cast, Callable
from glob import glob

import numpy as np
import torch, timm, joblib
from numpy.typing import NDArray

from PIL import Image as image
from PIL.Image import Image

import torch.nn as nn
from torch import Tensor
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
# type hint for joblib
from timm.models import EfficientNet 


DATASET = 'tristanzhang32/ai-generated-images-vs-real-images'
# DATASET_FOLDER = f'/kaggle/input/datasets/{DATASET}'
# TRAIN_DIR = DATASET_FOLDER + '/train'
# TEST_DIR  = DATASET_FOLDER + '/test'

num_classes = 2
model: nn.Module = None # type: ignore
device: torch.device = None # type: ignore

def set_model(model_name: str, /, *, force=False, prefix: str = '') -> torch.nn.Module:
	global model
	global num_classes
	# `num_classes` is a global set to 3; do not accept an override here.
	prefix = prefix or 'models/'
	fnames = [ model_name,
		f'{prefix}{model_name}_c{num_classes}',
		f'{prefix}{model_name}_c{num_classes}.pkl',
		f'{prefix}{model_name}_c{num_classes}.joblib',
		f'{prefix}{model_name}',
		f'{prefix}{model_name}.pkl', 
		f'{prefix}{model_name}.joblib',
	]
	for name in fnames:
		try:
			model = joblib.load(name)
			print(name, 'carregado com sucesso')
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

def transform() -> Callable[[Image], Tensor]:
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
	# Accept PIL Images or tensors/ndarrays. Do not force conversion to PIL first,
	# because calling code may already pass a PIL Image.
	# Ensure PIL grayscale images are converted to RGB so normalization
	# broadcasts correctly across 3 channels.
	return transforms.Compose([
		transforms.Lambda(to_pil),
		transforms.Resize(isize),
		transforms.ToTensor(),
		transforms.Normalize(mean=cfg['mean'], std=cfg['std'])
	])

def best_device() -> torch.device:
	global device
	if device is not None:
		return device
	# 1. NVIDIA (Padrão ouro para Deep Learning)
	if torch.cuda.is_available():
		device_name = torch.cuda.get_device_name(0)
		print(f"🚀 GPU NVIDIA Detectada: {device_name}")
		device = torch.device("cuda")
		return device
	# 2. Apple Silicon (M1, M2, M3 - Seu cenário atual)
	elif torch.backends.mps.is_available():
		print("🍏 GPU Apple Silicon (MPS) Detectada.")
		device = torch.device("mps")
		return device

	# 3. AMD / Intel via DirectML (Comum em Windows/Laptops sem NVIDIA)
	# Requer: pip install torch-directml
	try:
		import torch_directml # type: ignore
		if torch_directml.is_available():
			print("📦 GPU AMD/Intel (DirectML) Detectada.")
			device = torch_directml.device()
			return device
	except ImportError: pass

	# 4. Intel XPU (Específico para placas Intel Arc / Data Centers)
	if hasattr(torch, 'xpu') and torch.xpu.is_available():
		print("🔵 GPU Intel XPU Detectada.")
		device = torch.device("xpu")
		return device
	device = torch.device("cpu")
	return device

def to_tensor(img) -> Tensor:
	"""Convert images to CHW (C,H,W) tensors. For lists, return NCHW."""
	if isinstance(img, list):
		return torch.stack([to_tensor(i) for i in img])
	if isinstance(img, str):
		img = image.open(img)
	if isinstance(img, Image) and img.mode != 'RGB':
		img = img.convert('RGB')
	if not isinstance(img, Tensor):
		img = torch.as_tensor(np.array(img))
	# enforce 4D
	if img.ndim > 4: img.squeeze_()
	while img.ndim < 4: img.unsqueeze_(0)

	if img.shape[-1] in (1, 3):
		img = img.permute(0, 3, 1, 2)
	if img.shape[1] == 1:
		img = img.expand(-1, 3, -1, -1)
	return img

def enforce_uint8(img, max=False) -> Tensor:
	img = to_tensor(img)
	if img.is_floating_point() and (max or img.max() <= 1.0):
		img = img * 255.0
	return img.to(torch.uint8)

def enforce_frac(img, max=False) -> Tensor:
	img = to_tensor(img)
	hi_flot = img.is_floating_point() and (max or img.max() > 1.0)
	if img.dtype == torch.uint8 or hi_flot:
		img = img.float() / 255.0
	return img.clamp(0.0, 1.0)

def resize(img, h = 224, w = None) -> Tensor:
	if w is None: w = h
	if isinstance(img, list | tuple):
		tensors = [resize(i, h, w) for i in img]
		# inner resize returns a batched tensor of shape (1, C, H, W) for single images;
		# remove that extra leading dim before stacking to produce (N, C, H, W)
		normalized = [t.squeeze(0) if (t.ndim == 4 and t.shape[0] == 1) else t for t in tensors]
		return torch.stack(normalized)
	img = to_tensor(img)
	return torch.nn.functional.interpolate(img.float(), 
		size=(h, w), mode='bilinear', align_corners=False)

def to_pil(img) -> Image:
	if isinstance(img, str):
		img = image.open(img)
	if not isinstance(img, Image):
		return cast(Image, to_pil_image(img))
	if img.mode != 'RGB':
		img = img.convert('RGB')
	return img

pixel = np.uint8

def blend(a: Tensor, b: Tensor, alpha: float | Tensor) -> Tensor:
	# alpha pode ser escalar ou tensor broadcastable
	return a * (1.0 - alpha) + b * alpha

# kaggle_download('train/fake', 401, 500)
# ou 'train/real', 'test/fake', 'test/real'
def kaggle_download(folder:str, first:int, last:int):
	for i in range(first, last+1):
		fname = f'{folder}/{i:04d}.jpg'
		fpath = kag.dataset_download(DATASET, fname)
		os.rename(fpath, f'./dataset/{fname}')
	

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

	# Only consider definite ground-truth labels for reporting:
	# treat values >=0.999 as Real, <=0.001 as Fake. Any intermediate
	# values (the 'maybe' ground truth) are excluded from metrics so
	# they are neither punished nor rewarded.
	is_real_gt = y_test >= 0.999
	is_fake_gt = y_test <= 0.001
	definite_mask = is_real_gt | is_fake_gt

	if definite_mask.sum() == 0:
		print('No definite Real/Fake ground-truth samples to report on.')
		return p_final

	# filter arrays to only definite samples for metric computation
	pf = p_final[definite_mask]
	yf = y_test[definite_mask]

	is_real = yf >= 0.999
	is_fake = yf <= 0.001
	is_high = pf > thresh
	is_low = pf < (1 - thresh)

	# WRONG: final verdict strongly on wrong side
	w_total = np.sum(is_fake & is_high) + np.sum(is_real & is_low)
	# SURE: strong correct verdicts
	s_total = np.sum(is_real & is_high) + np.sum(is_fake & is_low)
	# DUNNO: remaining in gray zone
	d_total = np.sum(~(is_high | is_low))
	total = len(yf)

	pct = lambda x: x * 100.0 / total if total else 0.0

	print(f'Total (definite samples): {total}')
	print(f'Wrong: {pct(w_total):2.1f}%')
	print(f'Sure:  {pct(s_total):2.1f}%')
	print(f'Dunno: {pct(d_total):2.1f}%')
	# return original probabilities array for compatibility
	return p_final


import atexit
@atexit.register
def save_model_on_exit():
	"""Best-effort save: write a torch checkpoint with model and optional optimizer/epoch."""
	global model
	if model is None:
		return
	os.makedirs('models', exist_ok=True)
	mpath = 'models/model_temp.pkl'
	joblib.dump(model, mpath)
	print(mpath)

if __name__ == '__main__':
	kaggle_download("train/real", 1501, 2000)
	kaggle_download("train/fake", 1001, 1500)
	kaggle_download("train/fake", 1501, 2000)
