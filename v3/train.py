import os, torch
import torch.nn as nn
import helper, dset

LABEL_NAMES = ['fake', 'real']
LABEL_TO_IDX = {n: i for i, n in enumerate(LABEL_NAMES)}

set_model = helper.set_model

def train_head(filepaths: dset.Dataset | list[str], epochs: int = 3, batch_size: int = 16):
	"""Train head for two-class detection: 0=fake, 1=real.

	This version uses a standard CrossEntropy loss over definite labels.
	"""
	if filepaths is None:
		raise ValueError("filepaths must be provided (use --filelist to pass a newline-separated file of paths)")
	tr = helper.transform()
	device = helper.best_device()

	ds = filepaths if isinstance(filepaths, dset.Dataset) \
	else dset.SimpleFileDataset(filepaths, transform=tr)

	dloader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

	model = helper.model
	for name, p in model.named_parameters():
		p.requires_grad = 'classifier' in name or 'head' in name or 'fc' in name

	model.to(device)
	opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-3)

	# standard classification loss (two classes)
	criterion = nn.CrossEntropyLoss(reduction='sum')

	model.train()
	# canonical class count
	n_classes = getattr(helper, 'num_classes', 2)

	for epoch in range(epochs):
		total_loss = 0.0
		total = 0
		in_allowed = 0
		for batch in dloader:
			# dataset returns (tensor, label)
			xb, yb = batch

			xb = xb.to(device)
			yb = yb.to(device)

			logits = model(xb)
			loss = criterion(logits, yb)

			# single backward/step per batch
			opt.zero_grad()
			loss.backward()
			opt.step()

			# accumulate stats over all samples in the batch
			batch_size_actual = int(yb.size(0))
			loss_value = float(loss.detach().cpu().item())
			total_loss += loss_value
			preds = logits.detach().argmax(dim=1)
			correct = int((preds == yb).sum().item())
			in_allowed += correct
			total += batch_size_actual

		avg_loss = total_loss / total if total > 0 else float('nan')
		in_allowed_rate = in_allowed / total if total > 0 else 0.0
		print(f'Epoch {epoch+1}/{epochs} loss={avg_loss:.4f} acc={in_allowed_rate:.3f}')

if __name__ == '__main__':
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('--filelist', '-l', required=False, 
		default='', 
		help='newline-separated file with image paths')
	p.add_argument('--model', '-m', 
		default='model_temp', 
		help='model name (e.g. efficientnet_b0, etc.)')
	p.add_argument('--epochs', '-e', type=int, default=2)
	args = p.parse_args()
	filepaths = []
	if args.filelist:
		with open(args.filelist, 'r') as fh:
			filepaths = [l.strip() for l in fh.readlines() if l.strip()]
	else:
		filepaths = dset.DirDataset('dataset/train/real', 'dataset/train/fake')
	set_model(args.model, force=True) # model always uses global num_classes (2)
	train_head(filepaths=filepaths, epochs=args.epochs)
