from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from .constants import CLASSES


def classification_metrics(truth: np.ndarray, probs: np.ndarray, labels: list[int] | None = None) -> dict:
    labels = labels if labels is not None else list(range(len(CLASSES)))
    pred = probs.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(truth, pred, labels=labels, average="macro", zero_division=0)
    top3 = np.argsort(probs, axis=1)[:, -3:]
    picked = np.clip(probs[np.arange(len(truth)), truth], 1e-12, 1.0)
    return {
        "images": int(len(truth)),
        "accuracy": float(accuracy_score(truth, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "top3_accuracy": float(np.mean([t in row for t, row in zip(truth, top3, strict=True)])),
        "log_loss": float(-np.log(picked).mean()),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }


def per_class_table(truth: np.ndarray, probs: np.ndarray) -> list[dict]:
    pred = probs.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    precision, recall, f1, support = precision_recall_fscore_support(truth, pred, labels=labels, zero_division=0)
    return [{"class": CLASSES[i], "precision": float(precision[i]), "recall": float(recall[i]),
             "f1": float(f1[i]), "support": int(support[i])} for i in labels]


def confusion(truth: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return confusion_matrix(truth, probs.argmax(axis=1), labels=list(range(len(CLASSES))))


def top_confusions(matrix: np.ndarray, k: int = 8) -> list[dict]:
    off = matrix.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off, axis=None)[::-1][:k]
    rows, cols = np.unravel_index(flat, off.shape)
    return [{"actual": CLASSES[r], "predicted": CLASSES[c], "count": int(off[r, c])}
            for r, c in zip(rows, cols, strict=True) if off[r, c] > 0]
