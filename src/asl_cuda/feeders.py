"""Alimentadores de lotes a partir de la caché uint8.

`CudaFeeder` es el camino propuesto:
    1. CPU: gather de los índices del lote desde el memmap a un buffer pinned (uint8).
    2. Copia H2D asíncrona del buffer (4 veces menos bytes que enviar float32).
    3. GPU: kernel Numba que remuestrea, aplica color, normaliza y escribe el
       tensor de entrada directamente en el layout que usa la red.
Se usan dos buffers pinned alternados con un evento CUDA por buffer, así el CPU
prepara el lote k+1 mientras la GPU todavía procesa el lote k.

`CpuFeeder` hace lo mismo pero el kernel corre en todos los núcleos del CPU
(Numba prange). Sirve para la variante "todo en CPU" y para la ablación
"aumentación en CPU + red en GPU".
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import torch

from .augment import IMAGENET_MEAN, IMAGENET_STD, launch_warp, sample_params, warp_cpu
from .config import AugmentConfig
from .prep import CachedSplit


class _Base:
    def __init__(self, data: CachedSplit, batch_size: int, out_size: int, augment: AugmentConfig,
                 train: bool, seed: int, limit: int | None = None) -> None:
        self.data = data
        self.batch_size = batch_size
        self.out_size = out_size
        self.augment = augment
        self.train = train
        self.seed = seed
        self.epoch = 0
        self.count = min(len(data), limit) if limit else len(data)
        self.src_size = int(data.images.shape[1])

    def __len__(self) -> int:
        return (self.count + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _plan(self) -> tuple[np.ndarray, np.random.Generator | None]:
        if not self.train:
            return np.arange(self.count), None
        rng = np.random.default_rng([self.seed, self.epoch])
        return rng.permutation(self.count), rng

    def _gather(self, indices: np.ndarray, out: np.ndarray) -> None:
        # Ordenar mejora la localidad al leer del memmap; el orden dentro del lote no importa.
        order = np.sort(indices)
        np.take(self.data.images, order, axis=0, out=out[: len(order)])
        indices[:] = order


class CudaFeeder(_Base):
    def __init__(self, *args, device: torch.device, channels_last: bool = True,
                 block: tuple[int, int] = (16, 16), **kwargs) -> None:
        super().__init__(*args, **kwargs)
        from numba import cuda

        self.cuda = cuda
        self.device = device
        self.channels_last = channels_last
        self.block = tuple(block)
        shape = (self.batch_size, self.src_size, self.src_size, 3)
        self.host = [torch.empty(shape, dtype=torch.uint8).pin_memory() for _ in range(2)]
        self.host_params = [torch.empty((self.batch_size, 16), dtype=torch.float32).pin_memory() for _ in range(2)]
        self.ready = [torch.cuda.Event(), torch.cuda.Event()]
        self.mean = torch.from_numpy(IMAGENET_MEAN).to(device)
        self.std = torch.from_numpy(IMAGENET_STD).to(device)
        self.labels = torch.from_numpy(np.array(self.data.labels, dtype=np.int64)).to(device)  # copia: el memmap es de solo lectura
        self.data_wait = 0.0  # segundos que el CPU pasó armando lotes (para el reporte)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        import time

        order, rng = self._plan()
        stream = torch.cuda.current_stream(self.device)
        nb_stream = self.cuda.external_stream(stream.cuda_stream)
        # sync=False: el kernel se encola en el mismo stream que usa torch, así que el
        # orden ya está garantizado. Con sync=True Numba bloquearía el CPU en cada
        # lote y se perdería el traslape entre armar el lote k+1 y procesar el k.
        view = lambda t: self.cuda.as_cuda_array(t, sync=False)  # noqa: E731
        d_mean, d_std = view(self.mean), view(self.std)
        self.data_wait = 0.0
        for k, start in enumerate(range(0, self.count, self.batch_size)):
            slot = k % 2
            began = time.perf_counter()
            self.ready[slot].synchronize()  # el buffer ya se copió en la iteración k-2
            indices = order[start:start + self.batch_size].copy()
            n = len(indices)
            host_np = self.host[slot].numpy()
            self._gather(indices, host_np)
            params = sample_params(rng, n, self.src_size, self.out_size, self.augment, self.train).packed()
            self.host_params[slot].numpy()[:n] = params
            self.data_wait += time.perf_counter() - began

            src = self.host[slot][:n].to(self.device, non_blocking=True)
            prm = self.host_params[slot][:n].to(self.device, non_blocking=True)
            self.ready[slot].record(stream)
            out = torch.empty((n, 3, self.out_size, self.out_size), device=self.device,
                              memory_format=torch.channels_last if self.channels_last else torch.contiguous_format)
            launch_warp(view(src), view(prm), d_mean, d_std, view(out), self.block, nb_stream)
            labels = self.labels[torch.from_numpy(indices).to(self.device, non_blocking=True)]
            yield out, labels


class CpuFeeder(_Base):
    """Kernel en CPU multinúcleo. Si `device` es cuda, el lote float32 se copia a la GPU al final."""

    def __init__(self, *args, device: torch.device | None = None, channels_last: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.device = device or torch.device("cpu")
        self.channels_last = channels_last
        self.to_gpu = self.device.type == "cuda"
        self.src = np.empty((self.batch_size, self.src_size, self.src_size, 3), dtype=np.uint8)
        self.out = [torch.empty((self.batch_size, 3, self.out_size, self.out_size), dtype=torch.float32)
                    for _ in range(2 if self.to_gpu else 1)]
        if self.to_gpu:
            self.out = [t.pin_memory() for t in self.out]
            self.copied = [torch.cuda.Event(), torch.cuda.Event()]
        self.labels = np.asarray(self.data.labels, dtype=np.int64)
        self.data_wait = 0.0

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        import time

        order, rng = self._plan()
        self.data_wait = 0.0
        for k, start in enumerate(range(0, self.count, self.batch_size)):
            began = time.perf_counter()
            slot = k % len(self.out)
            if self.to_gpu:
                self.copied[slot].synchronize()
            indices = order[start:start + self.batch_size].copy()
            n = len(indices)
            self._gather(indices, self.src)
            params = sample_params(rng, n, self.src_size, self.out_size, self.augment, self.train).packed()
            out = self.out[slot][:n]
            warp_cpu(self.src[:n], params, out.numpy())
            labels = torch.from_numpy(self.labels[indices])
            self.data_wait += time.perf_counter() - began
            if not self.to_gpu:
                yield out, labels
                continue
            x = out.to(self.device, non_blocking=True)
            self.copied[slot].record(torch.cuda.current_stream(self.device))
            if self.channels_last:
                x = x.contiguous(memory_format=torch.channels_last)
            yield x, labels.to(self.device, non_blocking=True)
