"""Simple confusion-matrix utilities for val2 (3-class) evaluation."""
from __future__ import annotations

from typing import Sequence
import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] | None = None) -> np.ndarray:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if labels is None:
        labels = np.unique(np.concatenate([y_true, y_pred])).tolist()
    else:
        labels = list(labels)

    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    cm = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1
    return cm


def print_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] | None = None, normalize: bool = False) -> None:
    """Print a readable confusion matrix.

    Args:
        y_true: ground-truth class labels (integers)
        y_pred: predicted class labels (integers)
        labels: explicit label list (display order). If None, inferred.
        normalize: whether to row-normalize counts.
    """
    cm = confusion_matrix(y_true, y_pred, labels)
    if labels is None:
        labels = list(np.unique(np.concatenate([y_true, y_pred])))
    else:
        labels = list(labels)

    if normalize:
        with np.errstate(all='ignore'):
            row_sums = cm.sum(axis=1, keepdims=True).astype(float)
            cm_display = np.divide(cm, row_sums, where=row_sums != 0)
    else:
        cm_display = cm

    # print header
    header = "\t" + "\t".join(str(l) for l in labels)
    print("Confusion matrix (rows=true, cols=pred):")
    print(header)
    for i, lab in enumerate(labels):
        row = "\t".join(f"{v:.4f}" if normalize else str(int(v)) for v in cm_display[i])
        print(f"{lab}\t{row}")
