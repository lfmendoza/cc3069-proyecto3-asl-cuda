"""Corrida mínima de punta a punta en CPU con datos sintéticos en memoria."""

from dataclasses import replace

import numpy as np
import torch

from asl_cuda.config import load_train_config
from asl_cuda.evaluate import evaluate_split
from asl_cuda.feeders import CpuFeeder
from asl_cuda.prep import CachedSplit
from asl_cuda.train import fit


def _split(n_per_class=3, size=48, seed=0):
    rng = np.random.default_rng(seed)
    labels = np.repeat(np.arange(29), n_per_class)
    images = rng.integers(0, 256, (len(labels), size, size, 3), dtype=np.uint8)
    return CachedSplit(images, labels, {})


def test_cpu_feeder_shapes_and_labels():
    cfg = load_train_config("configs/smoke.yaml")
    data = _split()
    feeder = CpuFeeder(data, batch_size=16, out_size=32, augment=cfg.augment, train=True, seed=1)
    seen = []
    for x, y in feeder:
        assert x.shape[1:] == (3, 32, 32) and x.dtype == torch.float32
        seen.extend(y.tolist())
    assert sorted(seen) == sorted(data.labels.tolist())


def test_fit_and_evaluate_on_cpu(tmp_path):
    cfg = replace(load_train_config("configs/smoke.yaml"), image_size=32, batch_size=16, eval_batch_size=32)
    summary = fit(cfg, _split(), _split(seed=1), torch.device("cpu"), tmp_path, log=lambda *_: None)
    assert (tmp_path / "model.pt").exists() and (tmp_path / "history.csv").exists()
    assert summary["epochs_run"] == 2
    result = evaluate_split(tmp_path / "model.pt", _split(seed=2), cfg, torch.device("cpu"), tmp_path, "test")
    assert 0.0 <= result["accuracy"] <= 1.0 and result["images"] == 87
