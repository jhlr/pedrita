import os, random
from glob import glob
from typing import List, Optional, Callable

from PIL.Image import Image
from torch.utils.data import Dataset

def _pics_from_dir(d: Optional[str]) -> List[str]:
	if not d:
		return []
	if not os.path.isdir(d):
		return []
	return [p for p in glob(os.path.join(d, '*')) if p.lower().endswith(('.jpg', '.png', '.jpeg'))]


class DirDataset(Dataset):
	def __init__(self, 
		real_dir: Optional[str] = None, 
		fake_dir: Optional[str] = None, 
		transform: Optional[Callable] = None, 
		shuffle: bool = True,
	):

		self.transform = transform
		self.samples: List[tuple[str, int]] = []

		reals = _pics_from_dir(real_dir)
		fakes = _pics_from_dir(fake_dir)

		self.samples += [(p, 1) for p in reals]
		self.samples += [(p, 0) for p in fakes]

		if shuffle:
			random.shuffle(self.samples)

		if len(self.samples) == 0:
			print('Warning: DirDataset created with 0 samples')

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int):
		path, label = self.samples[idx]
		img = self.transform(path) if self.transform else path
		return img, label

class SimpleFileDataset(Dataset):
	def __init__(self, 
		filepaths: list[str], 
		transform: Optional[Callable] = None, 
		targets: Optional[list[int]] = None,
	):
		# filter out missing files so DataLoader workers won't error
		existing = [p for p in filepaths if os.path.exists(p)]
		missing = len(filepaths) - len(existing)
		if missing > 0:
			print(f'Warning: {missing} paths in filelist were missing and will be skipped')
		self.filepaths = existing
		self.transform = transform
		self.targets = targets

	def __len__(self):
		return len(self.filepaths)

	def __getitem__(self, idx):
		p = self.filepaths[idx]
		# Delegate loading to the transform to avoid loading the same image twice.
		img = self.transform(p) if self.transform else p
		# placeholder label: when using simple filelists you should provide
		# labels separately; default to `fake` (0) as a safe placeholder.
		if self.targets is not None:
			label = self.targets[idx]
		else:
			label = 0
		return img, label
