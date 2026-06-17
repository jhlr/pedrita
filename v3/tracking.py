"""Canonical MLflow tracking for v3.

One setup point, consistent param/metric names, and NO multi-GB artifact bloat:
model weights are not logged by default (that is what filled mlruns with 9.5G of
per-epoch checkpoints). Opt in with MLFLOW_LOG_MODELS=1 if you really want them.

Conventions that make the MLflow UI chart things automatically:
  - one experiment per task ('training', 'evaluation')
  - per-epoch metrics logged with `step=epoch` -> line charts over epochs
  - eval metrics share a flat namespace ('eval/accuracy', 'eval/sure', ...) ->
    comparable across runs in the runs table / parallel-coords plot
  - a confusion figure is logged as an artifact for a quick visual
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from collections.abc import Mapping

import mlflow

# SQLite backend (the file store './mlruns' is deprecated as of Feb 2026).
# Metrics/params/runs live in the DB; artifacts (figures) go under ./mlartifacts.
TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', 'sqlite:///mlflow.db')
LOG_MODELS = os.environ.get('MLFLOW_LOG_MODELS', '0').lower() in ('1', 'true', 'yes')

_ready = False


def setup(experiment: str) -> None:
	"""Point MLflow at the SQLite store (once) and select the experiment.

	New experiments get their artifacts under ./mlartifacts (the DB only holds
	metrics/params/runs)."""
	global _ready
	if not _ready:
		mlflow.set_tracking_uri(TRACKING_URI)
		_ready = True
	if mlflow.get_experiment_by_name(experiment) is None:
		mlflow.create_experiment(experiment, artifact_location='./mlartifacts')
	mlflow.set_experiment(experiment)


@contextmanager
def run(experiment: str, name: str,
	params: Mapping | None = None, tags: Mapping | None = None):
	"""Open a run, logging params/tags up front. Nests if a run is already active."""
	setup(experiment)
	nested = mlflow.active_run() is not None
	with mlflow.start_run(run_name=name, nested=nested) as active:
		if params:
			mlflow.log_params({k: v for k, v in params.items() if v is not None})
		if tags:
			mlflow.set_tags(dict(tags))
		yield active


def log_metrics(metrics: Mapping, step: int | None = None) -> None:
	clean = {k: float(v) for k, v in metrics.items() if v is not None}
	if clean:
		mlflow.log_metrics(clean, step=step)


def log_model(model, name: str = 'model') -> None:
	"""Log model weights — only when MLFLOW_LOG_MODELS is set (off by default)."""
	if not LOG_MODELS:
		return
	import mlflow.pytorch as mlpt
	mlpt.log_model(model, name=name)


def log_confusion_figure(stats: Mapping, name: str = 'confusion.png') -> None:
	"""Render the real/fake x wrong/sure/dunno breakdown as a heatmap artifact.

	`stats` keys expected (fractions in 0..1):
	  wrong_real, sure_real, dunno_real, wrong_fake, sure_fake, dunno_fake
	Silently no-ops if matplotlib isn't available.
	"""
	try:
		import matplotlib
		matplotlib.use('Agg')
		import matplotlib.pyplot as plt
		import numpy as np
	except Exception:
		return
	grid = np.array([
		[stats.get('wrong_real', 0), stats.get('sure_real', 0), stats.get('dunno_real', 0)],
		[stats.get('wrong_fake', 0), stats.get('sure_fake', 0), stats.get('dunno_fake', 0)],
	], dtype=float)
	fig, ax = plt.subplots(figsize=(4.5, 3))
	im = ax.imshow(grid, cmap='RdYlGn_r', vmin=0, vmax=1, aspect='auto')
	ax.set_xticks(range(3), ['wrong', 'sure', 'dunno'])
	ax.set_yticks(range(2), ['real', 'fake'])
	for i in range(2):
		for j in range(3):
			ax.text(j, i, f'{grid[i, j]:.0%}', ha='center', va='center', color='black')
	fig.colorbar(im, ax=ax, fraction=0.046)
	ax.set_title('Eval breakdown')
	fig.tight_layout()
	mlflow.log_figure(fig, name)
	plt.close(fig)
