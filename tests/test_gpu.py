"""Pruebas que solo corren con GPU real:  ASL_REAL_GPU=1 uv run pytest tests/test_gpu.py -v"""

import os

import numpy as np
import pytest
import torch

pytestmark = pytest.mark.skipif(not (os.environ.get("ASL_REAL_GPU") and torch.cuda.is_available()),
                                reason="requiere GPU NVIDIA y ASL_REAL_GPU=1")


def _data(n=70, size=200, seed=0):
    from asl_cuda.prep import CachedSplit

    rng = np.random.default_rng(seed)
    return CachedSplit(rng.integers(0, 256, (n, size, size, 3), dtype=np.uint8), rng.integers(0, 29, n), {})


def test_real_kernel_matches_numpy():
    from numba import cuda

    from asl_cuda.augment import IMAGENET_MEAN, IMAGENET_STD, cuda_kernel, grid_for, sample_params, warp_numpy
    from asl_cuda.config import load_train_config

    cfg = load_train_config()
    src = _data(8).images
    params = sample_params(np.random.default_rng(3), 8, 200, 224, cfg.augment, True).packed()
    out = cuda.device_array((8, 3, 224, 224), np.float32)
    grid, block = grid_for(8, 224, 224, (16, 16))
    cuda_kernel()[grid, block](cuda.to_device(src), cuda.to_device(params), cuda.to_device(IMAGENET_MEAN),
                               cuda.to_device(IMAGENET_STD), out)
    # fastmath puede cambiar el último bit de algunas divisiones
    np.testing.assert_allclose(out.copy_to_host(), warp_numpy(src, params, 224), atol=1e-3)


@pytest.mark.parametrize("train", [False, True])
def test_cuda_feeder_matches_cpu_feeder(train):
    from asl_cuda.config import load_train_config
    from asl_cuda.feeders import CpuFeeder, CudaFeeder

    cfg = load_train_config()
    data = _data()
    common = dict(batch_size=32, out_size=224, augment=cfg.augment, train=train, seed=5)
    gpu = list(CudaFeeder(data, **common, device=torch.device("cuda"), channels_last=True))
    cpu = [(x.clone(), y.clone()) for x, y in CpuFeeder(data, **common)]
    assert len(gpu) == len(cpu) == 3
    for (xg, yg), (xc, yc) in zip(gpu, cpu, strict=True):
        assert xg.is_contiguous(memory_format=torch.channels_last)
        torch.testing.assert_close(xg.cpu().contiguous(), xc.contiguous(), atol=1e-3, rtol=0)
        assert torch.equal(yg.cpu(), yc)


def test_amp_training_step_runs(tmp_path):
    from dataclasses import replace

    from asl_cuda.config import load_train_config
    from asl_cuda.train import fit

    cfg = replace(load_train_config("configs/smoke.yaml"), max_per_class=None)
    summary = fit(cfg, _data(64, 120), _data(32, 120, 1), torch.device("cuda"), tmp_path, log=lambda *_: None)
    assert summary["amp"] is True and summary["epochs_run"] == 2
