"""El kernel debe producir lo mismo en los tres backends (NumPy, Numba CPU, CUDA)."""

from dataclasses import replace

import numpy as np
import pytest

from asl_cuda.augment import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    cuda_kernel,
    grid_for,
    sample_params,
    warp_cpu,
    warp_numpy,
)
from asl_cuda.config import AugmentConfig

CFG = AugmentConfig()


def _src(batch=3, size=12, seed=0):
    return np.random.default_rng(seed).integers(0, 256, (batch, size, size, 3), dtype=np.uint8)


@pytest.mark.parametrize("train", [False, True])
def test_numba_cpu_matches_numpy(train):
    src = _src()
    params = sample_params(np.random.default_rng(1), 3, 12, 10, CFG, train).packed()
    out = np.zeros((3, 3, 10, 10), np.float32)
    warp_cpu(src, params, out)
    np.testing.assert_allclose(out, warp_numpy(src, params, 10), atol=1e-4)


@pytest.mark.parametrize("train", [False, True])
@pytest.mark.parametrize("block", [(4, 4), (8, 2), (3, 5)])
def test_cuda_kernel_matches_numpy(train, block):
    # bloques que no dividen exactamente la salida prueban la condición de borde del kernel
    src = _src()
    params = sample_params(np.random.default_rng(2), 3, 12, 10, CFG, train).packed()
    out = np.full((3, 3, 10, 10), np.nan, np.float32)
    grid, threads = grid_for(3, 10, 10, block)
    cuda_kernel()[grid, threads](src, params, IMAGENET_MEAN, IMAGENET_STD, out)
    assert not np.isnan(out).any(), "algún píxel de salida no fue escrito"
    np.testing.assert_allclose(out, warp_numpy(src, params, 10), atol=1e-4)


def test_identity_transform_recovers_pixels():
    src = _src()
    params = sample_params(None, 3, 12, 12, replace(CFG, eval_crop=1.0), False).packed()
    out = warp_numpy(src, params, 12)
    restored = (out.transpose(0, 2, 3, 1) * IMAGENET_STD + IMAGENET_MEAN) * 255
    np.testing.assert_allclose(restored, src, atol=0.01)


def test_eval_params_are_deterministic():
    a = sample_params(None, 4, 200, 224, CFG, False).packed()
    b = sample_params(None, 4, 200, 224, CFG, False).packed()
    np.testing.assert_array_equal(a, b)


def test_train_params_change_with_seed_but_are_reproducible():
    a = sample_params(np.random.default_rng(7), 8, 200, 224, CFG, True).packed()
    b = sample_params(np.random.default_rng(7), 8, 200, 224, CFG, True).packed()
    c = sample_params(np.random.default_rng(8), 8, 200, 224, CFG, True).packed()
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_erase_region_is_zero_after_normalization():
    src = _src(1)
    params = sample_params(None, 1, 12, 10, CFG, False).packed()
    params[0, 12:16] = [2, 3, 6, 8]
    out = warp_numpy(src, params, 10)
    assert np.all(out[0, :, 3:8, 2:6] == 0)
    assert np.any(out[0, :, 0, 0] != 0)


def test_launch_grid_covers_output():
    grid, threads = grid_for(64, 224, 224, (16, 16))
    assert grid == (14, 14, 64) and threads == (16, 16, 1)
    grid, _ = grid_for(5, 224, 224, (32, 8))
    assert grid[0] * 32 >= 224 and grid[1] * 8 >= 224 and grid[2] == 5
