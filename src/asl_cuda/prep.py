"""Fase 1 (CPU): lectura de archivos, decodificación JPEG y armado de la caché.

Todo lo que toca el sistema de archivos se queda aquí. El resultado es un
arreglo uint8 (N, S, S, 3) guardado como .npy y abierto con memmap, de modo que
el entrenamiento ya no vuelve a abrir ni decodificar un solo JPEG.

El trabajo se reparte en bloques de rutas entre procesos (no hilos: la
decodificación con Pillow mantiene el GIL en buena parte del trabajo). Cada
proceso escribe directamente en su rango del memmap, así que no hay que
serializar imágenes de regreso al proceso principal.

Este módulo no importa torch a propósito: en Windows cada proceso hijo vuelve a
importar el módulo y cargar torch ahí costaría segundos por proceso.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

_TARGET: np.ndarray | None = None
_SIZE: int = 0


def decode_bytes(data: bytes, size: int) -> np.ndarray:
    """JPEG/PNG -> RGB uint8 (size, size, 3). Recorta al cuadrado central si hace falta."""
    with Image.open(io.BytesIO(data)) as raw:
        raw.draft("RGB", (size, size))  # en JPEG grandes decodifica directo a una escala menor
        image = ImageOps.exif_transpose(raw).convert("RGB")
    if image.size != (size, size):
        image = ImageOps.fit(image, (size, size), Image.Resampling.BILINEAR)
    return np.asarray(image, dtype=np.uint8)


def decode_file(path: str | Path, size: int) -> tuple[np.ndarray, str]:
    data = Path(path).read_bytes()
    return decode_bytes(data, size), hashlib.sha256(data).hexdigest()


def _init_worker(target_path: str, size: int) -> None:
    global _TARGET, _SIZE
    _TARGET = np.load(target_path, mmap_mode="r+")
    _SIZE = size


def _decode_chunk(task: tuple[int, Sequence[str]]) -> dict:
    start, paths = task
    assert _TARGET is not None
    began = time.perf_counter()
    shas, errors = [], []
    for offset, path in enumerate(paths):
        try:
            image, sha = decode_file(path, _SIZE)
            _TARGET[start + offset] = image
            shas.append(sha)
        except (OSError, ValueError) as exc:
            _TARGET[start + offset] = 0
            shas.append("")
            errors.append(f"{path}: {type(exc).__name__}")
    return {"start": start, "count": len(paths), "shas": shas, "errors": errors,
            "pid": os.getpid(), "seconds": time.perf_counter() - began}


@dataclass
class DecodeStats:
    images: int
    workers: int
    chunk: int
    seconds: float
    images_per_second: float
    bytes_written: int
    worker_busy_seconds: dict[int, float] = field(default_factory=dict)
    load_imbalance: float = 1.0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        data = asdict(self)
        data["worker_busy_seconds"] = {str(k): v for k, v in self.worker_busy_seconds.items()}
        return data


def resolve_workers(workers: int) -> int:
    return (os.cpu_count() or 1) if workers <= 0 else workers


def decode_to_npy(paths: Sequence[str], target: Path, size: int, workers: int, chunk: int | None = None) -> tuple[DecodeStats, list[str]]:
    """Decodifica `paths` dentro de `target` (.npy uint8). workers=1 corre en el proceso actual."""
    global _TARGET
    workers = resolve_workers(workers)
    if chunk is None:
        # ~4 bloques por proceso: suficiente para balancear sin pagar demasiado en comunicación
        chunk = max(8, min(256, len(paths) // (workers * 4) or 1))
    target.parent.mkdir(parents=True, exist_ok=True)
    array = np.lib.format.open_memmap(target, mode="w+", dtype=np.uint8, shape=(len(paths), size, size, 3))
    del array  # el header ya quedó escrito; cada proceso abre su propia vista
    tasks = [(i, list(paths[i:i + chunk])) for i in range(0, len(paths), chunk)]
    started = time.perf_counter()
    if workers == 1:
        _init_worker(str(target), size)
        results = [_decode_chunk(task) for task in tasks]
        assert _TARGET is not None
        _TARGET.flush()
        _TARGET = None
    else:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker, initargs=(str(target), size)) as pool:
            results = list(pool.map(_decode_chunk, tasks))
    elapsed = time.perf_counter() - started

    shas = [""] * len(paths)
    busy: dict[int, float] = {}
    errors: list[str] = []
    for result in results:
        shas[result["start"]:result["start"] + result["count"]] = result["shas"]
        busy[result["pid"]] = busy.get(result["pid"], 0.0) + result["seconds"]
        errors.extend(result["errors"])
    mean_busy = sum(busy.values()) / len(busy)
    stats = DecodeStats(
        images=len(paths), workers=workers, chunk=chunk, seconds=elapsed,
        images_per_second=len(paths) / elapsed if elapsed else 0.0,
        bytes_written=len(paths) * size * size * 3, worker_busy_seconds=busy,
        load_imbalance=max(busy.values()) / mean_busy if mean_busy else 1.0, errors=errors,
    )
    return stats, shas


def decode_with_threads(paths: Sequence[str], size: int, threads: int) -> float:
    """Misma decodificación con hilos en vez de procesos. Solo se usa en el
    benchmark para mostrar el efecto del GIL; devuelve los segundos."""
    from concurrent.futures import ThreadPoolExecutor

    out = np.empty((len(paths), size, size, 3), dtype=np.uint8)

    def work(i: int) -> None:
        out[i] = decode_file(paths[i], size)[0]

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(work, range(len(paths)), chunksize=32))
    return time.perf_counter() - started


@dataclass
class CachedSplit:
    images: np.ndarray
    labels: np.ndarray
    meta: dict

    def __len__(self) -> int:
        return len(self.labels)


def cache_files(cache_dir: Path, split: str, size: int) -> tuple[Path, Path, Path, Path]:
    stem = cache_dir / f"{split}_{size}"
    return (stem.with_suffix(".images.npy"), stem.with_suffix(".labels.npy"),
            stem.with_suffix(".sha256.npy"), stem.with_suffix(".meta.json"))


def build_split_cache(paths: Sequence[str], labels: np.ndarray, cache_dir: Path, split: str, size: int,
                      workers: int, digest: str, force: bool = False) -> dict:
    images_path, labels_path, sha_path, meta_path = cache_files(cache_dir, split, size)
    if meta_path.exists() and not force:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("digest") == digest and images_path.exists():
            meta["reused"] = True
            return meta
    stats, shas = decode_to_npy(paths, images_path, size, workers)
    if stats.errors:
        raise RuntimeError(f"{len(stats.errors)} archivos no se pudieron decodificar: {stats.errors[:5]}")
    np.save(labels_path, np.asarray(labels, dtype=np.int64))
    np.save(sha_path, np.asarray(shas, dtype="S64"))
    meta = {"split": split, "size": size, "digest": digest, **stats.as_dict(), "reused": False}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def load_split(cache_dir: Path, split: str, size: int, in_memory: bool = False) -> CachedSplit:
    images_path, labels_path, _, meta_path = cache_files(cache_dir, split, size)
    if not images_path.exists():
        raise FileNotFoundError(f"No existe la caché {images_path}. Corra primero `asl-cuda prep`.")
    images = np.load(images_path, mmap_mode=None if in_memory else "r")
    return CachedSplit(images, np.load(labels_path), json.loads(meta_path.read_text(encoding="utf-8")))


def load_shas(cache_dir: Path, split: str, size: int) -> np.ndarray:
    return np.load(cache_files(cache_dir, split, size)[2]).astype(str)
