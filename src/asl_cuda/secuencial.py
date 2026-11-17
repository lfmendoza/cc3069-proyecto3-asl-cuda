"""Versión original del Lab 3 (robust-v3), sin cambios de lógica.

Se conserva para medir la línea base con las mismas imágenes y el mismo modelo.
Su flujo por cada lote, en cada época, es:

    abrir JPEG -> decodificar -> EXIF -> RGB -> transformaciones de torchvision
    (PIL, una imagen a la vez) -> ToTensor -> Normalize -> apilar lote ->
    forward/backward en CPU

Es decir, cada época vuelve a leer y decodificar los 43,500 archivos de train
y los 11,600 de validación. Con `num_workers=0` y `threads=1` se obtiene la
ejecución estrictamente secuencial; con la configuración del Lab 3
(`num_workers=4`, `threads=16`) se obtiene la versión "CPU tal como se entregó".
"""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .augment import IMAGENET_MEAN, IMAGENET_STD


def _exif_rgb(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def lab3_transform(size: int, augment: bool) -> transforms.Compose:
    """Misma secuencia de `robust_transform` del Lab 3.

    Único cambio: las dos lambdas del original se reemplazaron por una función
    con nombre, porque en Windows los procesos del DataLoader se crean con
    spawn y una lambda no se puede serializar.
    """
    prefix: list[Any] = [transforms.Lambda(_exif_rgb)]
    if augment:
        visual: list[Any] = [
            transforms.RandomResizedCrop(size, scale=(0.70, 1.0), ratio=(0.90, 1.10)),
            transforms.RandomAffine(10, translate=(0.07, 0.07), scale=(0.93, 1.07)),
            transforms.ColorJitter(0.20, 0.20, 0.12, 0.02),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(0.1, 1.0))], p=0.10),
            transforms.RandomPerspective(distortion_scale=0.12, p=0.12),
        ]
    else:
        visual = [transforms.Resize(size + 32), transforms.CenterCrop(size)]
    suffix: list[Any] = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN.tolist(), IMAGENET_STD.tolist())]
    if augment:
        suffix.append(transforms.RandomErasing(p=0.10, scale=(0.01, 0.05), ratio=(0.5, 2.0), value="random"))
    return transforms.Compose(prefix + visual + suffix)


class FileDataset(Dataset[tuple[torch.Tensor, int]]):
    """Lee el archivo del disco en cada acceso, como el ASLDataset del Lab 3."""

    def __init__(self, paths: list[str], labels: list[int], transform: Callable[[Image.Image], torch.Tensor]) -> None:
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        with Image.open(self.paths[index]) as image:
            tensor = self.transform(image.copy())
        return tensor, self.labels[index]


def make_loader(frame: pd.DataFrame, paths: list[str], size: int, batch_size: int, augment: bool,
                workers: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    dataset = FileDataset(paths, frame["label"].astype(int).tolist(), lab3_transform(size, augment))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      persistent_workers=False, generator=generator)


def run_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
              clip: float, max_batches: int | None = None) -> dict[str, float]:
    """Cuerpo del ciclo de entrenamiento del Lab 3 con cronómetros por etapa."""
    model.train()
    data_seconds = compute_seconds = 0.0
    seen = correct = 0
    loss_sum = 0.0
    tick = time.perf_counter()
    for step, (images, labels) in enumerate(loader):
        loaded = time.perf_counter()
        data_seconds += loaded - tick
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        loss_sum += float(loss.detach()) * len(labels)
        correct += int((logits.argmax(1) == labels).sum())
        seen += len(labels)
        tick = time.perf_counter()
        compute_seconds += tick - loaded
        if max_batches is not None and step + 1 >= max_batches:
            break
    return {"images": seen, "data_seconds": data_seconds, "compute_seconds": compute_seconds,
            "seconds": data_seconds + compute_seconds, "train_loss": loss_sum / max(seen, 1),
            "train_accuracy": correct / max(seen, 1)}
