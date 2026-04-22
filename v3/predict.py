import torch, cv2
import numpy as np
from PIL.Image import Image

from pathlib import Path
from typing import Sequence

from torch import Tensor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

try: from . import dset, helper
except ImportError:
	import helper, dset

# 1) Build model (EfficientNet backbone)

# if reset_classifier not available:
# model.classifier = nn.Linear(in_features, 1)

# binary output -> BCEWithLogitsLoss

# Match training mapping: 0 = fake, 1 = real
LABEL_NAMES = ['fake', 'real']
LABEL = {n: i for i, n in enumerate(LABEL_NAMES)}

set_model = helper.set_model

def predict_and_heatmap(img_bgr) -> tuple[float, np.ndarray]:
	"""Return `real` probability and heatmap(s).

	- If `both=False` (default) returns: `(real_prob, heatmap_fake_or_None)` (backwards-compatible).
	- If `both=True` returns: `(real_prob, heatmap_fake_or_None, heatmap_real_or_None)`.

	Gate: if `real_prob >= thresh` returns heatmaps as None (same behavior as before).
	"""
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
	tr = helper.transform()
	device = helper.best_device()
	tensor = tr(img_rgb).unsqueeze(0).to(device)

	helper.model.eval()
	with torch.no_grad():
		logits = helper.model(tensor)

	probs = torch.softmax(logits.detach().squeeze(0), dim=0) if isinstance(logits, torch.Tensor) else torch.tensor(np.array(logits)).float()
	real_prob = float(probs[LABEL['real']].item())

	# find last Conv2d as target layer
	layer = getattr(helper, 'conv_head', None)
	if layer is None:
		for m in reversed(list(helper.model.modules())):
			if isinstance(m, torch.nn.Conv2d):
				layer = m
				break
		if layer is None:
			raise RuntimeError('target layer for Grad-CAM not found')

	def cam_target(label: str, cam: GradCAM):
		tgt = ClassifierOutputTarget(LABEL[label])
		greyscale = cam(input_tensor=tensor, targets=[tgt])[0] # type: ignore
		return cv2.resize(greyscale, (img_rgb.shape[1], img_rgb.shape[0]))
	
	with GradCAM(model=helper.model, target_layers=[layer]) as cam:
		fake_cam_img = cam_target('fake', cam)
		real_cam_img = cam_target('real', cam)
		greyscale = fake_cam_img - real_cam_img
		min, max = greyscale.min(), greyscale.max()
		greyscale = (greyscale - min) / (max - min + 1e-8)
		cam_img = show_cam_on_image(img_rgb.astype(np.float32) / 255.0, greyscale, use_rgb=True)
		cam_img = cv2.cvtColor(cam_img, cv2.COLOR_RGB2BGR)
	return real_prob, cam_img

def predict(img_bgr) -> float:
	tr = helper.transform(); device = helper.best_device()
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
	with torch.no_grad():
		probs = helper.model(tr(img_rgb).unsqueeze(0).to(device))
	p = torch.softmax(probs, dim=1)[:, LABEL['real']].squeeze().item()
	return p


def predict_batch(imgs_bgr: Sequence | Tensor) -> np.ndarray:
	"""Predict a batch. Delegate loading/conversion to `model.transform()`.

	Inputs accepted:
	- `Tensor` (N,C,H,W) or (C,H,W)
	- sequence of `Tensor` (C,H,W)
	- sequence of numpy arrays, PIL Images, or path-like objects
	"""
	if not imgs_bgr or not len(imgs_bgr):
		return np.array([])

	# single tensor batch
	if isinstance(imgs_bgr, Tensor) and imgs_bgr.ndim == 3:
		imgs_bgr.unsqueeze_(0).unsqueeze_(0)
	
	# sequence of tensors -> assume already transformed
	tr = helper.transform()
	tensors = []
	for t in imgs_bgr:
		if not isinstance(t, Tensor):
			t = tr(t)
		if t.ndim == 4 and t.shape[0] == 1:
			t = t.squeeze(0)
		tensors.append(t)
	batch = torch.stack(tensors)
	device = helper.best_device()
	if batch.device.type != device.type:
		batch = batch.to(device)
	with torch.no_grad():
		logits = helper.model(batch)
	return torch.softmax(logits, dim=1)[:, LABEL['real']].cpu().numpy()

def evaluate_folder(test_dir: str, batch_size: int = 16, thresh: float = 0.6):
	"""Evaluate a test folder using dset.DirDataset, predict_batch and helper.compare.

	Expects `test_dir` to contain subfolders `real/` and `fake/`.
	This function uses the dataset discovery from `dset.DirDataset` (samples list),
	reads images with OpenCV (BGR), runs `predict_batch` and then maps labels to
	the ground-truth encoding expected by `helper.compare` (real=0.999, fake=0.001).
	"""
	td = Path(test_dir)
	ds = dset.DirDataset(str(td / 'real'), str(td / 'fake'), shuffle=True)
	
	probs, ylabels = [], []
	for i in range(0, len(ds), batch_size):
		batch = [ds[j] for j in range(i, min(i + batch_size, len(ds)))]
		imgs, labels = zip(*batch)
		if not imgs: continue
		batch_probs = predict_batch(imgs)
		probs.extend(list(batch_probs))
		ylabels.extend([1 if int(l)==1 else 0 for l in labels[:len(batch_probs)]])

	p_arr = np.array(probs)
	helper.compare(p_arr, ylabels, thresh=thresh)
	return p_arr, ylabels

# Example usage
if __name__ == '__main__':
	import argparse
	from datetime import datetime as dt
	# get cli arguments

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', '-m', type=str, default='model_temp', help='model name (e.g. efficientnet_b0, etc.)')
	parser.add_argument('--image', '-i', type=str, default='frame.jpg')
	parser.add_argument('--eval', '-e', type=str, default=None, help='path to test folder (subfolders per label) to compute accuracy')
	parser.add_argument('--cpu', action='store_true', help='force CPU usage even if GPU available')
	args = parser.parse_args()
	
	helper.best_device(args.cpu)
	helper.set_model(args.model)
	# expose model as module-level name so predict_batch/evaluate use it

	# If evaluation requested, run and exit
	if args.eval is not None:
		now = dt.now()
		evaluate_folder(args.eval)
		print(dt.now() - now)
		exit(0)

	img = cv2.imread(args.image)
	if img is None:
		raise SystemExit('frame.jpg not found or unreadable')
	
	now = dt.now()
	prob, cam_img = predict_and_heatmap(img)
	print(f'Proba real: {prob*100:.2f} %')
	print(dt.now() - now)

	if cam_img is not None:
		fname = 'heatmap.jpg'
		cv2.imwrite(fname, cam_img)
		print(fname)
