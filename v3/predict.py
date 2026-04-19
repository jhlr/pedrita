import torch, cv2
import numpy as np
from pathlib import Path
from typing import Sequence
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import helper, dset

# 1) Build model (EfficientNet backbone)

# if reset_classifier not available:
# model.classifier = nn.Linear(in_features, 1)

# binary output -> BCEWithLogitsLoss

# Match training mapping: 0 = fake, 1 = real
LABEL_NAMES = ['fake', 'real']
LABEL = {n: i for i, n in enumerate(LABEL_NAMES)}

set_model = helper.set_model

def predict_and_heatmap(img_bgr):
	"""Return `fake` probability and Grad-CAM heatmap (BGR) or None.

	Skip heatmap when model is confident the image is `real` (>=0.6).
	"""
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
	tr = helper.transform()
	device = helper.best_device()
	tensor = tr(img_rgb).unsqueeze(0).to(device)

	model.eval()
	with torch.no_grad():
		logits = model(tensor)

	probs = torch.softmax(logits.detach().squeeze(0), dim=0) if isinstance(logits, torch.Tensor) else torch.tensor(np.array(logits)).float()
	fake_prob = float(probs[LABEL['fake']].item())
	real_prob = float(probs[LABEL['real']].item())

	# skip heatmap when real is confident
	if (probs.argmax().item() == LABEL['real']) and (real_prob >= 0.6):
		return fake_prob, None

	# find last Conv2d as target layer
	layer = getattr(model, 'conv_head', None)
	if layer is None:
		for m in reversed(list(model.modules())):
			if isinstance(m, torch.nn.Conv2d):
				layer = m
				break
		if layer is None:
			raise RuntimeError('target layer for Grad-CAM not found')

	with GradCAM(model=model, target_layers=[layer]) as cam:
		tgt = ClassifierOutputTarget(LABEL['fake'])
		grayscale_cam = cam(input_tensor=tensor, targets=[tgt])[0]  # type: ignore

	grayscale_cam = cv2.resize(grayscale_cam, (img_rgb.shape[1], img_rgb.shape[0]))
	cam_image = show_cam_on_image(img_rgb.astype(np.float32) / 255.0, grayscale_cam, use_rgb=True)
	return fake_prob, cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

def predict(img_bgr):
	tr = helper.transform(); device = helper.best_device()
	img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.uint8)
	with torch.no_grad():
		probs = model(tr(img_rgb).unsqueeze(0).to(device))
	return torch.softmax(probs, dim=1)[:, LABEL['real']].cpu().numpy()


def predict_batch(imgs_bgr: Sequence):
	if len(imgs_bgr) == 0:
		return np.zeros((0, helper.num_classes))
	tr = helper.transform(); device = helper.best_device()
	batch = torch.stack([tr(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.uint8)) for im in imgs_bgr]).to(device)
	with torch.no_grad():
		probs = model(batch)
	return torch.softmax(probs, dim=1)[:, LABEL['real']].cpu().numpy()

def evaluate_folder(test_dir: str, batch_size: int = 16, limit: int | None = None, thresh: float = 0.6):
	"""Evaluate a test folder using dset.DirDataset, predict_batch and helper.compare.

	Expects `test_dir` to contain subfolders `real/` and `fake/`.
	This function uses the dataset discovery from `dset.DirDataset` (samples list),
	reads images with OpenCV (BGR), runs `predict_batch` and then maps labels to
	the ground-truth encoding expected by `helper.compare` (real=0.999, fake=0.001).
	"""
	td = Path(test_dir)
	ds = dset.DirDataset(str(td / 'real'), str(td / 'fake'), transform=None, shuffle=False)
	items = ds.samples if not limit else ds.samples[:limit]
	if not items:
		print('No test images found in', test_dir); return

	probs, gts = [], []
	for i in range(0, len(items), batch_size):
		batch = items[i:i+batch_size]
		paths, labels = zip(*batch)
		imgs = [im for im in (cv2.imread(p) for p in paths) if im is not None]
		if not imgs: continue
		probs = predict_batch(imgs)
		gts.extend([1 if int(l)==1 else 0 for l in labels[:len(probs)]])

	p_arr = np.array(probs); y_arr = np.array([0.999 if g==1 else 0.001 for g in gts])
	helper.compare(p_arr, y_arr, thresh=thresh)
	return p_arr, y_arr

# Example usage
if __name__ == '__main__':
	import argparse
	# get cli arguments

	parser = argparse.ArgumentParser()
	parser.add_argument('--model', '-m', type=str, default='model_temp', help='model name (e.g. efficientnet_b0, etc.)')
	parser.add_argument('--image', '-i', type=str, default='frame.jpg')
	parser.add_argument('--nomap', '-n', action='store_true', help='only predict probability, do not save heatmap')
	parser.add_argument('--eval', '-e', type=str, default=None, help='path to test folder (subfolders per label) to compute accuracy')
	parser.add_argument('--limit', type=int, default=None, help='limit number of test images when evaluating')
	args = parser.parse_args()
	
	model = set_model(args.model, force=True)
	# expose model as module-level name so predict_batch/evaluate use it
	globals()['model'] = model

	# If evaluation requested, run and exit
	if args.eval is not None:
		evaluate_folder(args.eval, limit=args.limit)
		raise SystemExit(0)

	img = cv2.imread(args.image)
	if img is None:
		raise SystemExit('frame.jpg not found or unreadable')
	
	if args.nomap:
		# return a small list/dict of probabilities for readability
		logits = predict(img)
		probs = torch.softmax(logits, dim=1).squeeze(0).cpu().tolist()
		probs_dict = dict(zip(LABEL_NAMES, probs))
		print('Proba:', probs_dict)
	else:
		prob, heatmap = predict_and_heatmap(img)
		print('Proba:', prob)
		if heatmap is not None:
			fname = args.image.rsplit('.', 1)[0]
			fname = fname.rsplit("/", 1)[-1] + '.jpg'
			fname = f'outputs/{args.model}_{fname}'
			cv2.imwrite(fname, heatmap)
			print(f'Heatmap saved to {fname}')
		else:
			print('Heatmap skipped due to gating (high-confidence real)')
