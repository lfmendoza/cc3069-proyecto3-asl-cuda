from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

from .constants import ROOT


def _read(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _build(cls: type, raw: dict[str, Any]) -> Any:
    names = {f.name for f in fields(cls)}
    unknown = set(raw) - names
    if unknown:
        raise ValueError(f"{cls.__name__}: claves desconocidas {sorted(unknown)}")
    return cls(**raw)


@dataclass(frozen=True)
class DataConfig:
    seed: int
    raw_dir: str
    cache_dir: str
    external_dir: str
    cache_size: int
    group_window: int
    train_windows: list[int]
    validation_windows: list[int]
    test_windows: list[int]

    def split_of_window(self, window: int) -> str | None:
        hits = [name for name, ws in (("train", self.train_windows), ("validation", self.validation_windows),
                                      ("test", self.test_windows)) if window in ws]
        if len(hits) > 1:
            raise ValueError(f"La ventana {window} está asignada a más de una partición")
        return hits[0] if hits else None

    def path(self, attr: str) -> Path:
        value = Path(getattr(self, attr))
        return value if value.is_absolute() else ROOT / value


@dataclass(frozen=True)
class AugmentConfig:
    scale: tuple[float, float] = (0.70, 1.0)
    ratio: tuple[float, float] = (0.90, 1.10)
    rotation_deg: float = 10.0
    translate: float = 0.07
    zoom: tuple[float, float] = (0.93, 1.07)
    perspective: float = 0.12
    perspective_p: float = 0.12
    brightness: float = 0.20
    contrast: float = 0.20
    saturation: float = 0.12
    erase_p: float = 0.10
    erase_area: tuple[float, float] = (0.01, 0.05)
    eval_crop: float = 0.875


@dataclass(frozen=True)
class StageConfig:
    name: str
    epochs: int
    unfrozen_blocks: int
    backbone_lr: float
    head_lr: float


@dataclass(frozen=True)
class CudaConfig:
    amp: bool = True
    channels_last: bool = True
    cudnn_benchmark: bool = True
    threads_per_block: tuple[int, int] = (16, 16)
    augment_device: str = "gpu"


@dataclass(frozen=True)
class CpuConfig:
    threads: int = 0


@dataclass(frozen=True)
class TrainConfig:
    seed: int
    architecture: str
    pretrained: bool
    image_size: int
    batch_size: int
    eval_batch_size: int
    weight_decay: float
    label_smoothing: float
    gradient_clip_norm: float
    early_stopping_patience: int
    stages: list[StageConfig]
    augment: AugmentConfig
    cuda: CudaConfig = field(default_factory=CudaConfig)
    cpu: CpuConfig = field(default_factory=CpuConfig)
    max_per_class: int | None = None


@dataclass(frozen=True)
class BenchConfig:
    images_per_class: int
    repeats: int
    warmup: int
    prep_workers: list[int]
    train_batches: int
    batch_size: int
    inference_batches: int
    latency_samples: int
    threads_per_block: list[list[int]]
    gpu_batch_sweep: list[int]
    pipeline_repeats: int = 3
    prep_images_per_class: int = 200


def load_data_config(path: str | Path = "configs/data.yaml") -> DataConfig:
    return _build(DataConfig, _read(path))


def load_train_config(path: str | Path = "configs/train.yaml") -> TrainConfig:
    raw = _read(path)
    aug = raw.pop("augment", {})
    for key in ("scale", "ratio", "zoom", "erase_area"):
        if key in aug:
            aug[key] = tuple(aug[key])
    cuda_raw = raw.pop("cuda", {})
    if "threads_per_block" in cuda_raw:
        cuda_raw["threads_per_block"] = tuple(cuda_raw["threads_per_block"])
    if cuda_raw.get("augment_device", "gpu") not in {"gpu", "cpu"}:
        raise ValueError("cuda.augment_device debe ser 'gpu' o 'cpu'")
    cpu_raw = raw.pop("cpu", {})
    stages = [_build(StageConfig, s) for s in raw.pop("stages")]
    return TrainConfig(
        **raw,
        stages=stages,
        augment=_build(AugmentConfig, aug),
        cuda=_build(CudaConfig, cuda_raw),
        cpu=_build(CpuConfig, cpu_raw),
    )


def load_bench_config(path: str | Path = "configs/bench.yaml") -> BenchConfig:
    return _build(BenchConfig, _read(path))
