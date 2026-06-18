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
import logging
from pathlib import Path
from contextlib import contextmanager
from collections.abc import Mapping

import mlflow

logger = logging.getLogger(__name__)

# Anchor the store to the package dir (api/pedrita) so the DB and artifacts land
# in the same place no matter the process CWD (e.g. uvicorn started from backend/).
_PKG_DIR = Path(__file__).resolve().parents[1]

# SQLite backend (the file store './mlruns' is deprecated as of Feb 2026).
# Metrics/params/runs live in the DB; artifacts (images, figures) go under mlartifacts.
TRACKING_URI = os.environ.get('MLFLOW_TRACKING_URI', f"sqlite:///{_PKG_DIR / 'mlflow.db'}")
ARTIFACT_ROOT = os.environ.get('MLFLOW_ARTIFACT_ROOT', str(_PKG_DIR / 'mlartifacts'))
LOG_MODELS = os.environ.get('MLFLOW_LOG_MODELS', '0').lower() in ('1', 'true', 'yes')
# Log every inference (received + generated artifacts) by default; opt out with =0.
LOG_INFERENCE = os.environ.get('MLFLOW_LOG_INFERENCE', '1').lower() in ('1', 'true', 'yes')

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
		mlflow.create_experiment(experiment, artifact_location=ARTIFACT_ROOT)
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


def _sqlite_path() -> str | None:
	"""Filesystem path of the SQLite store, or None if the backend isn't SQLite."""
	prefix = 'sqlite:///'
	return TRACKING_URI[len(prefix):] if TRACKING_URI.startswith(prefix) else None


def _png_bytes(img) -> bytes | None:
	"""Encode a PIL.Image (or array-like) as PNG bytes."""
	try:
		from PIL import Image
		if not isinstance(img, Image.Image):
			import numpy as np
			img = Image.fromarray(np.asarray(img))
		import io
		buf = io.BytesIO()
		img.save(buf, format='PNG')
		return buf.getvalue()
	except Exception as exc:
		logger.debug('falha ao codificar imagem (ignorado): %s', exc)
		return None


def _store_blobs(run_uuid: str, blobs: list) -> None:
	"""Persist artifact bytes *inside* the SQLite mlflow.db, content-addressed.

	Bytes live once in `artifact_blobs` keyed by their sha256 (INSERT OR IGNORE),
	and `inference_artifacts` links each run to a blob by that hash. Identical
	images — re-uploads, or the same picture seen by separate endpoints — are thus
	stored a single time, with every run still pointing at it. Everything stays in
	one self-contained file (no mlartifacts/ folder)."""
	path = _sqlite_path()
	if not path or not blobs:
		return
	import sqlite3, time, hashlib
	con = sqlite3.connect(path, timeout=30)
	try:
		con.execute(
			'CREATE TABLE IF NOT EXISTS artifact_blobs ('
			' sha256 TEXT PRIMARY KEY, mime TEXT, size INTEGER, data BLOB, created REAL)')
		con.execute(
			'CREATE TABLE IF NOT EXISTS inference_artifacts ('
			' id INTEGER PRIMARY KEY AUTOINCREMENT,'
			' run_uuid TEXT, name TEXT, kind TEXT, sha256 TEXT, created REAL)')
		con.execute('CREATE INDEX IF NOT EXISTS ix_infart_run ON inference_artifacts(run_uuid)')
		now = time.time()
		for (name, kind, mime, data) in blobs:
			sha = hashlib.sha256(data).hexdigest()
			con.execute(
				'INSERT OR IGNORE INTO artifact_blobs (sha256,mime,size,data,created)'
				' VALUES (?,?,?,?,?)', (sha, mime, len(data), data, now))
			con.execute(
				'INSERT INTO inference_artifacts (run_uuid,name,kind,sha256,created)'
				' VALUES (?,?,?,?,?)', (run_uuid, name, kind, sha, now))
		con.commit()
	finally:
		con.close()


def log_inference(
	name: str,
	*,
	params: Mapping | None = None,
	metrics: Mapping | None = None,
	images: Mapping | None = None,
	artifacts: Mapping | None = None,
	tags: Mapping | None = None,
	experiment: str = 'inference',
) -> None:
	"""Persist one inference as an MLflow run with its received + generated artifacts.

	The run (params/metrics/tags) goes into MLflow's tables as usual; the binary
	artifacts are stored *inside the same SQLite DB* as BLOBs:
	  - images:    {filename: PIL.Image | np.ndarray} — received image + heatmap, PNG.
	  - artifacts: {filename: json-serialisable} — e.g. the Gemini context dict.

	Best-effort and non-fatal: tracking is auxiliary, so any failure (locked DB,
	missing mlflow, etc.) is swallowed and never breaks the prediction itself.
	Disable globally with MLFLOW_LOG_INFERENCE=0. If TRACKING_URI is not SQLite,
	images/json fall back to MLflow's regular (filesystem) artifact store.
	"""
	if not LOG_INFERENCE:
		return
	try:
		# Build the blobs up front so the SQLite write window stays tiny.
		blobs = []
		for fname, img in (images or {}).items():
			data = _png_bytes(img) if img is not None else None
			if data:
				blobs.append((fname, 'image', 'image/png', data))
		for fname, obj in (artifacts or {}).items():
			if obj is not None:
				import json
				blobs.append((fname, 'json', 'application/json',
					json.dumps(dict(obj), ensure_ascii=False).encode('utf-8')))

		# Record the received image's sha256 as a queryable run tag, so every record
		# in the DB is tied to — and dedup-able by — its source image.
		tags = dict(tags or {})
		recv = next((d for (n, k, m, d) in blobs if n == 'received.png'), None)
		if recv is not None:
			import hashlib
			tags.setdefault('image_sha256', hashlib.sha256(recv).hexdigest())

		sqlite_store = _sqlite_path() is not None
		run_uuid = None
		with run(experiment, name, params=params, tags=tags) as active:
			if metrics:
				log_metrics(metrics)
			run_uuid = active.info.run_id
			if not sqlite_store:
				# Non-SQLite backend: fall back to filesystem artifact logging.
				for fname, img in (images or {}).items():
					if img is not None:
						mlflow.log_image(img, fname)
				for fname, obj in (artifacts or {}).items():
					if obj is not None:
						mlflow.log_dict(dict(obj), fname)

		# Write BLOBs after the run is committed to minimise lock contention.
		if sqlite_store and run_uuid:
			_store_blobs(run_uuid, blobs)
	except Exception as exc:  # pragma: no cover - tracking must not break inference
		logger.debug('log_inference falhou (ignorado): %s', exc)
