"""Evaluación del modelo entrenado: test interno, dataset externo y fotos nuevas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import TrainConfig
from .constants import CLASSES
from .manifest import relative, scan_any
from .metrics import classification_metrics, confusion, per_class_table, top_confusions
from .model import load_checkpoint
from .prep import CachedSplit, decode_file
from .train import make_feeder, predict


def load_images(paths: list[str], size: int) -> np.ndarray:
    return np.stack([decode_file(p, size)[0] for p in paths]) if paths else np.empty((0, size, size, 3), np.uint8)


def folder_split(folder: Path, size: int) -> tuple[CachedSplit, pd.DataFrame]:
    """Subcarpetas por clase (A/, del/, ...) o archivos con la letra como prefijo (A_1.jpg)."""
    frame = scan_any(folder)
    frame["path"] = frame["path"].map(str)
    images = load_images(frame["path"].tolist(), size)
    return CachedSplit(images, frame["label"].to_numpy(), {"source": str(folder)}), frame


def evaluate_split(checkpoint: Path, data: CachedSplit, cfg: TrainConfig, device: torch.device,
                   out_dir: Path, name: str, paths: list[str] | None = None) -> dict:
    model, meta = load_checkpoint(checkpoint, device)
    if device.type == "cuda" and cfg.cuda.channels_last:
        model = model.to(memory_format=torch.channels_last)
    feeder = make_feeder(data, cfg, device, train=False)
    truth, probs, seconds = predict(model, feeder, device, amp=cfg.cuda.amp)
    present = sorted(set(truth.tolist()))
    metrics = classification_metrics(truth, probs, labels=present)
    matrix = confusion(truth, probs)
    result = {"name": name, "checkpoint": relative(checkpoint), "device": str(device), **metrics,
              "seconds": seconds, "images_per_second": len(truth) / seconds if seconds else None,
              "classes_present": [CLASSES[i] for i in present], "top_confusions": top_confusions(matrix)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{name}_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    pd.DataFrame(per_class_table(truth, probs)).to_csv(out_dir / f"{name}_per_class.csv", index=False)
    np.save(out_dir / f"{name}_confusion.npy", matrix)
    pred = probs.argmax(1)
    table = pd.DataFrame({"true": [CLASSES[i] for i in truth], "pred": [CLASSES[i] for i in pred],
                          "confidence": probs.max(1)})
    if paths is not None:
        table.insert(0, "path", [relative(Path(p)) for p in paths])
    table.to_csv(out_dir / f"{name}_predictions.csv", index=False)
    if paths is not None and len(truth) <= 90:  # hasta 3 personas x 29 clases
        contact_sheet(data.images, truth, probs, out_dir / f"{name}_predicciones.png", name)
    return result


def contact_sheet(images: np.ndarray, truth: np.ndarray, probs: np.ndarray, path: Path, title: str) -> None:
    """Miniaturas con la clase real, la predicha y la confianza (para fotos propias)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(truth)
    cols = min(10, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(1.45 * cols, 1.75 * rows), squeeze=False)
    pred = probs.argmax(1)
    for ax in axes.flat:
        ax.axis("off")
    for i, ax in enumerate(axes.flat[:n]):
        ok = pred[i] == truth[i]
        ax.imshow(images[i])
        ax.set_title(f"{CLASSES[truth[i]]} → {CLASSES[pred[i]]}\n{probs[i, pred[i]]:.2f}", fontsize=7,
                     color="#0b0b0b" if ok else "#e34948", fontweight="normal" if ok else "bold")
    hits = int((pred == truth).sum())
    fig.suptitle(f"{title}: {hits}/{n} correctas (rojo = error)", fontsize=9, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.35 / fig.get_figheight()))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def predict_files(checkpoint: Path, files: list[str], cfg: TrainConfig, device: torch.device, top: int = 3) -> list[dict]:
    model, meta = load_checkpoint(checkpoint, device)
    size = int(meta.get("cache_size", 200))
    data = CachedSplit(load_images(files, size), np.zeros(len(files), dtype=np.int64), {})
    truth, probs, _ = predict(model, make_feeder(data, cfg, device, train=False), device, amp=cfg.cuda.amp)
    rows = []
    for path, p in zip(files, probs, strict=True):
        best = np.argsort(p)[::-1][:top]
        rows.append({"path": path, "prediction": CLASSES[best[0]],
                     "top": [(CLASSES[i], round(float(p[i]), 4)) for i in best]})
    return rows
