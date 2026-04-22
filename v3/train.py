import os, torch
import torch.nn as nn

try: from . import dset, helper
except ImportError:
	import helper, dset

set_model = helper.set_model

def train(filepaths: dset.DirDataset | list[str], epochs: int = 3, batch_size: int = 32):
	# Train head for two-class detection: 0=fake, 1=real.
	# This version uses a standard CrossEntropy loss over definite labels.
	if filepaths is None:
		raise ValueError("filepaths must be provided (use --filelist to pass a newline-separated file of paths)")

	# training transform (with augmentations)
	tr = helper.transform(train=True)
	ds = filepaths if isinstance(filepaths, dset.DirDataset) \
	else dset.SimpleFileDataset(filepaths, transform=tr)
	ds.transform = tr
	device = helper.best_device()

	train_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)

	# For simplicity: fully fine-tune the model (unfreeze all params)
	for p in helper.model.parameters():
		p.requires_grad = True

	helper.model.to(device)
	opt = torch.optim.AdamW(helper.model.parameters(), 
		lr=1e-4, weight_decay=1e-5)
	# Reduce LR when a monitored metric has stopped improving.
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
		opt, mode='min', factor=0.5, patience=1)
	criterion = nn.CrossEntropyLoss(reduction='mean')
	for epoch in range(epochs):
		helper.model.train()
		total_loss = 0.0
		total = 0
		correct_total = 0
		for batch in train_loader:
			xb, yb = batch
			xb = xb.to(device)
			yb = yb.to(device)

			logits = helper.model(xb)
			loss = criterion(logits, yb)

			opt.zero_grad()
			loss.backward()
			opt.step()

			batch_size_actual = int(yb.size(0))
			total += batch_size_actual
			total_loss += float(loss.detach().cpu().item())
			preds = logits.detach().argmax(dim=1)
			correct_total += int((preds == yb).sum().item())

		train_acc = correct_total / total if total > 0 else 0.0
		scheduler.step(train_acc)
		current_lr = opt.param_groups[0]['lr']
		print(f'Epoch {epoch+1}/{epochs}', 
			f'loss={total_loss:.4f}',
			f'acc={train_acc:.3f}',
			f'lrate={current_lr:.2e}')
	helper.retrained = True

if __name__ == '__main__':
	import argparse
	p = argparse.ArgumentParser()
	p.add_argument('--model', '-m', 
		default='model_temp', 
		help='model name (e.g. efficientnet_b0, etc.)')
	p.add_argument('--epochs', '-e', type=int, default=2)
	p.add_argument('--folder', '-f', type=str, default=None, help='optional folder to read from (overrides --filelist)')
	args = p.parse_args()
	filepaths = []
	if not args.folder:
		raise ValueError("Please provide a folder with --folder containing 'real/' and 'fake/' subfolders with training images.")
	filepaths = dset.DirDataset(
		os.path.join(args.folder, 'real'), 
		os.path.join(args.folder, 'fake'), 
	)
	helper.set_model(args.model) # model always uses global num_classes (2)
	train(filepaths=filepaths, epochs=args.epochs)
