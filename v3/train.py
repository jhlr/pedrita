import torch, copy
from datetime import datetime as dt
from pathlib import Path
from collections.abc import Sequence, Callable
import torch.nn as nn
import numpy as np
from torch import Tensor
import PIL.Image as pil

try: from . import helper, tracking
except ImportError:
	import helper, tracking

__all__ = ['train', 'merge', 'online_training']


def _seed_worker(worker_id):
	# Make DataLoader workers deterministic (PyTorch reproducibility recipe).
	import random
	worker_seed = torch.initial_seed() % 2**32
	np.random.seed(worker_seed)
	random.seed(worker_seed)


def with_mlflow(f):
	def wrapper(*args, **kwgs):
		with tracking.run('training', 'train', params=kwgs):
			try: return f(*args, **kwgs)
			finally: tracking.log_model(helper.model, name='final_model')
	return wrapper

@helper.timer
@with_mlflow
def train(
	filepaths: helper.DirDataset | str | Path, 
	/, *, epochs: int = 3, 	
	batch_size: int = 64,
	ohkeep: float = 0.5, 
	ohwarmup: int = 1,
	ohalpha: float = 0.5,
	clip: float = 1.0,
	freeze: int = 0,
	limit: int | None = None,
	lr: float = 5e-4,
	wd: float = 1e-5,
	val: 'helper.DirDataset | str | Path | None' = None,
):
	# Train for two-class detection: 0=fake, 1=real.
	# Supports optional Online Hard Example Mining (OHEM).
	# training transform (with augmentations)
	filepaths = helper.DirDataset(filepaths, 'train',
		limit=limit, shuffle=True,
		transform=helper.transform(train=True),
	)
	ohalpha = float(ohalpha)
	if not (0.0 <= ohalpha <= 1.0):
		raise ValueError('ohalpha must be between 0 and 1')

	device = helper.best_device()
	seed = helper.seed_all()  # reproducible: seeds python/numpy/torch (PEDRITA_SEED)
	gen = torch.Generator()
	gen.manual_seed(seed)
	train_loader = torch.utils.data.DataLoader(
		filepaths, batch_size=batch_size, shuffle=True,
		num_workers=4, persistent_workers=True,
		generator=gen, worker_init_fn=_seed_worker)

	# Optional validation set: deterministic transform (no augmentations), used to
	# drive the scheduler and keep the best epoch. None -> behaviour unchanged.
	val_loader = None
	if val is not None:
		val_ds = val if isinstance(val, helper.DirDataset) else helper.DirDataset(
			val, 'test', shuffle=False, transform=helper.transform(train=False))
		# Guarantee a deterministic eval transform even if a bare DirDataset was passed.
		val_ds.transform = helper.transform(train=False)
		val_loader = torch.utils.data.DataLoader(
			val_ds, batch_size=batch_size, shuffle=False, num_workers=4)
	best_val, best_state = -1.0, None

	treinable = helper.freeze(freeze)
	opt = torch.optim.AdamW(treinable, lr=lr, weight_decay=wd)

	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( opt, 
		mode='max', factor=0.7, patience=0, 
		threshold=0.005, min_lr=5e-6, cooldown=0, )
	helper.model.to(device)
	centropy = torch.nn.CrossEntropyLoss(reduction='none')

	def _val_acc():
		"""Validation accuracy (eval mode, no grad). None if no val set."""
		if val_loader is None:
			return None
		helper.model.eval()
		vc = vt = 0
		with torch.no_grad():
			for xb, yb in val_loader:
				xb = xb.to(device, non_blocking=True)
				yb = yb.long().to(device, non_blocking=True)
				preds = helper.expand(helper.model(xb)).argmax(dim=1)
				vc += int((preds == yb).sum()); vt += int(yb.size(0))
		return vc / vt if vt else 0.0

	# Baseline BEFORE training: gives the chart a starting point (step 0) and a
	# fallback — best_val is seeded with the pre-training score, so a training run
	# that beats nothing leaves the original weights untouched.
	base_val = _val_acc()
	if base_val is not None:
		best_val = base_val
		best_state = copy.deepcopy(helper.model.state_dict())
		print(f'Baseline val_acc={base_val:.3f}')
	tracking.log_metrics({'val/acc': base_val, 'train/lr': lr}, step=0)
	print('Starting training...')

	for epoch in range(epochs):

		with helper.wlock:
			helper.model.train()
			now = dt.now()
			total_loss = torch.tensor(0.0, device=device)
			total = 0
			correct_total = torch.tensor(0, device=device)
			selected_total = 0
			for batch in train_loader:
				xb, yb = batch
				xb = xb.to(device, non_blocking=True)
				yb = yb.long().to(device, non_blocking=True)

				logits = helper.expand(helper.model(xb))
				losses = centropy(logits, yb)

				# Apply OHEM: pick top-k hardest samples per batch (after warmup)
				loss = losses.mean()
				k = n = int(losses.numel())
				if epoch >= ohwarmup:
					k = min(n, int(ohkeep)) if ohkeep >= 1.0 else max(1, int(ohkeep * n))
					hard_losses, _ = torch.topk(losses, k)
					# Weighted mix between OHEM mean and full-batch mean
					ohmean = hard_losses.mean()
					loss = ohalpha * ohmean + (1.0 - ohalpha) * loss

				selected_total += k
				opt.zero_grad()
				loss.backward()
				torch.nn.utils.clip_grad_norm_(helper.model.parameters(), clip)
				opt.step()

				batch_size_actual = int(yb.size(0))
				total += batch_size_actual
				total_loss += loss.detach() * k
				probs = logits.softmax(dim=1)
				preds = probs.argmax(dim=1)
				correct_total += (preds == yb).sum()
		
			helper.retrained = True
			train_acc = correct_total.item() / total if total > 0 else 0.0
			avg_loss = total_loss.item() / (selected_total if selected_total > 0 else 1)

			# Validation pass. Falls back to train_acc for scheduling only when no
			# val set was provided. Best epoch (by val) is kept; seeded with baseline.
			val_acc = _val_acc()
			if val_acc is not None and val_acc > best_val:
				best_val = val_acc
				best_state = copy.deepcopy(helper.model.state_dict())

			scheduler.step(val_acc if val_acc is not None else train_acc)
			current_lr = opt.param_groups[0]['lr']

		tracking.log_metrics({
			'train/lr': current_lr,
			'train/loss': total_loss.item() / (selected_total if selected_total > 0 else 1),
			'train/acc': correct_total.item() / total if total > 0 else 0.0,
			'val/acc': val_acc,
		}, step=epoch + 1)

		print(dt.now() - now)
		print(f'Epoch {epoch+1}/{epochs}',
			f'loss={avg_loss:.4f}',
			f'acc={train_acc:.3f}',
			f'val={val_acc:.3f}' if val_acc is not None else 'val=-',
			f'lr={current_lr:.2e}',
			f'wd={wd:.1e}',
		)

	# Keep the best epoch by validation accuracy (instead of the last one).
	if best_state is not None:
		helper.model.load_state_dict(best_state)
		print(f'Best model restored: val_acc={best_val:.3f}')
	
def merge(*models: nn.Module) -> nn.Module:
	with helper.wlock:
		if not all(isinstance(m, nn.Module) for m in models):
			raise ValueError("merge expects nn.Module instances")
		out_model = copy.deepcopy(models[0])
		merged = out_model.state_dict()
		for m in models[1:]:
			state = m.state_dict()
			for k in merged.keys():
				merged[k] += state[k] / (len(models) - 1)
		out_model.load_state_dict(merged)
		return out_model

# desativado por enquanto
TRAIN_BUFFER: list[tuple[Tensor, int]] = None # type: ignore
def online_training(
	samples: Sequence[pil.Image] | Tensor, 
	preds: Sequence[float] | Tensor,
	/, *, 
	lr: float = 5e-6,
	wd: float = 2e-2,
	clip: float = 0.1,
	thresh: float = 0.98,
	batch_size: int = 32,
):	
	global TRAIN_BUFFER
	if (samples is None and preds is None) or TRAIN_BUFFER is None:
		TRAIN_BUFFER = None
		return
	"""Ajuste fino global e ultra-estabilizado com toda a rede aberta."""
	thresh = np.clip(thresh, 0.0, 1.0)
	if thresh < 0.5: thresh = 1 - thresh
	if isinstance(preds, (float, int)): preds = [preds]

	tobuffer = []
	for img, p in zip(samples, preds):
		if p > thresh or p < (1 - thresh):
			tobuffer.append((img, int(p > 0.5)))

	with helper.wlock:
		TRAIN_BUFFER.extend(tobuffer)
		if len(TRAIN_BUFFER) < batch_size: return
		device = helper.best_device()
		samples, labels = zip(*TRAIN_BUFFER)
		TRAIN_BUFFER.clear()

	tr = helper.transform()

	labels = torch.tensor(labels, dtype=torch.long, device=device)
	# Ensure `samples` is a single batched Tensor. `samples` may be a tuple/list
	# of PIL images or a tuple/list of Tensors (from TRAIN_BUFFER). Normalize
	# both cases into a Tensor of shape [B,C,H,W].
	if not isinstance(samples, Tensor):
		processed = []
		for s in samples:
			s = s if isinstance(s, Tensor) else tr(s)
			processed.append(s)
		samples = torch.stack(processed).to(device)
	
	# 1. Colocamos o modelo em treino completo
	with helper.wlock:
		helper.model.train()
		helper.freeze(0)  # Descongelamos toda a rede para ajuste fino global
		
		# 2. Passamos todos os parâmetros (sem helper.freeze)
		# Usamos os hiperparâmetros propostos de forma cirúrgica
		optimizer = torch.optim.AdamW(
			helper.model.parameters(), # Toda a rede computa gradiente
			lr=lr,                   # LR extremamente baixa (Âncora de estabilidade)
			weight_decay=wd          # WD Alto (Regularização agressiva contra viés)
		)
		criterion = torch.nn.CrossEntropyLoss()
		
		# 3. Passo de otimização
		optimizer.zero_grad()
		logits = helper.expand(helper.model(samples))
		loss = criterion(logits, labels)
		loss.backward()
		
		# 4. CLIP Mínimo (O disjuntor de segurança contra pseudo-labels ruidosos)
		torch.nn.utils.clip_grad_norm_(helper.model.parameters(), max_norm=clip)
		optimizer.step()
		# Sinaliza o salvamento automático da Pedrita
		helper.retrained = True
		helper.model.eval()
		