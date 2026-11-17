"""Inventario del dataset y partición agrupada train/validation/test.

El dataset de Kaggle se generó extrayendo frames de un video por letra, así que
las imágenes con índices cercanos son casi idénticas. Si se reparten al azar,
validation y test terminan midiendo memoria y no generalización. Por eso se
asignan ventanas completas de 100 frames y se dejan ventanas de separación
entre particiones (misma política del Lab 3, versión robust-v3).
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DataConfig
from .constants import CLASS_TO_INDEX, CLASSES, FOLDER_ALIASES, ROOT

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def class_from_folder(name: str) -> str:
    return FOLDER_ALIASES.get(name.lower(), name.upper())


def locate_class_root(root: Path) -> Path:
    """Devuelve la carpeta que contiene exactamente las 29 subcarpetas de clase."""
    queue = [root]
    while queue:
        current = queue.pop(0)
        subdirs = [p for p in current.iterdir() if p.is_dir()] if current.is_dir() else []
        names = {class_from_folder(p.name) for p in subdirs}
        if set(CLASSES) <= names:
            return current
        queue.extend(sorted(subdirs))
    raise FileNotFoundError(f"No se encontró una carpeta con las 29 clases debajo de {root}")


def frame_index(path: Path) -> int:
    digits = re.findall(r"\d+", path.stem)
    return int(digits[-1]) if digits else -1


def scan_folder(class_root: Path) -> pd.DataFrame:
    rows = []
    for folder in sorted(p for p in class_root.iterdir() if p.is_dir()):
        label = class_from_folder(folder.name)
        if label not in CLASS_TO_INDEX:
            continue
        for path in sorted(folder.iterdir()):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                rows.append({"path": path, "class": label, "label": CLASS_TO_INDEX[label], "index": frame_index(path)})
    if not rows:
        raise FileNotFoundError(f"No hay imágenes en {class_root}")
    return pd.DataFrame(rows)


def label_for_file(path: Path) -> str | None:
    """Clase de una imagen suelta: carpeta contenedora (A/, del/...) o prefijo del nombre (A_1.jpg)."""
    folder = class_from_folder(path.parent.name)
    if folder in CLASS_TO_INDEX:
        return folder
    prefix = class_from_folder(re.split(r"[_\-\s]", path.stem)[0])
    return prefix if prefix in CLASS_TO_INDEX else None


def scan_any(root: Path) -> pd.DataFrame:
    """Busca imágenes recursivamente (fotos propias, datasets externos con clases incompletas)."""
    rows = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in IMAGE_EXTENSIONS and path.is_file():
            label = label_for_file(path)
            if label is not None:
                rows.append({"path": path, "class": label, "label": CLASS_TO_INDEX[label], "index": frame_index(path)})
    if not rows:
        raise FileNotFoundError(f"No se encontraron imágenes etiquetables en {root}")
    return pd.DataFrame(rows)


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def build_manifest(config: DataConfig) -> pd.DataFrame:
    class_root = locate_class_root(config.path("raw_dir"))
    frame = scan_folder(class_root)
    frame["window"] = (frame["index"] - 1) // config.group_window
    frame["split"] = frame["window"].map(config.split_of_window)
    frame = frame.dropna(subset=["split"]).copy()
    frame["path"] = frame["path"].map(relative)
    frame = frame.sort_values(["split", "label", "index"]).reset_index(drop=True)
    validate_manifest(frame)
    return frame


def validate_manifest(frame: pd.DataFrame) -> None:
    windows = frame.groupby(["class", "window"])["split"].nunique()
    if (windows > 1).any():
        raise ValueError("Una ventana de frames quedó repartida entre dos particiones")
    if frame["path"].duplicated().any():
        raise ValueError("Rutas duplicadas en el manifest")
    for split, group in frame.groupby("split"):
        missing = set(CLASSES) - set(group["class"])
        if missing:
            raise ValueError(f"La partición {split} no tiene las clases {sorted(missing)}")


def check_cross_split_duplicates(frame: pd.DataFrame, sha256: np.ndarray) -> int:
    """Cuenta archivos byte-idénticos que aparecen en más de una partición."""
    probe = pd.DataFrame({"sha": sha256, "split": frame["split"].to_numpy()})
    return int((probe.groupby("sha")["split"].nunique() > 1).sum())


def manifest_digest(frame: pd.DataFrame) -> str:
    stable = frame[["path", "class", "split"]].sort_values("path").to_csv(index=False)
    return hashlib.sha256(stable.encode()).hexdigest()


def subsample(frame: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """Toma `per_class` imágenes equiespaciadas por clase (determinista, cubre todo el video)."""
    parts = []
    for _, group in frame.groupby("label", sort=True):
        group = group.sort_values("index")
        if len(group) > per_class:
            group = group.iloc[np.linspace(0, len(group) - 1, per_class).astype(int)]
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def absolute_paths(frame: pd.DataFrame) -> list[str]:
    return [str(p if Path(p).is_absolute() else ROOT / p) for p in frame["path"]]
