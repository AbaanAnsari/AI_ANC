"""
evaluation/metrics.py — Aggregated Evaluation Metrics
======================================================
Classification metrics (confusion matrix, per-class accuracy, accuracy)
computed from accumulated predictions and ground-truth labels.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

NOISE_CLASS_NAMES = {0: "stationary", 1: "non-stationary", 2: "impulsive"}
NUM_CLASSES = 3


def compute_confusion_matrix(
    true_labels: List[int],
    pred_labels: List[int],
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """
    Compute a confusion matrix.

    Returns
    -------
    np.ndarray, shape (num_classes, num_classes)
        confusion[i, j] = number of samples where true=i and predicted=j.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(true_labels, pred_labels):
        cm[int(t), int(p)] += 1
    return cm


def compute_classification_metrics(
    true_labels: List[int],
    pred_labels: List[int],
    num_classes: int = NUM_CLASSES,
) -> Dict:
    """
    Compute classification metrics.

    Returns
    -------
    dict with:
        accuracy          : float  (overall)
        per_class_accuracy: dict[int → float]
        confusion_matrix  : np.ndarray (num_classes, num_classes)
        macro_precision   : float or None
        macro_recall      : float or None
        macro_f1          : float or None
        class_counts      : dict[int → int]  (true label distribution)
    """
    true_arr = np.array(true_labels, dtype=np.int64)
    pred_arr = np.array(pred_labels, dtype=np.int64)

    total = len(true_arr)
    if total == 0:
        raise ValueError("Cannot compute metrics on empty label lists.")

    overall_acc = float(np.mean(true_arr == pred_arr))

    cm = compute_confusion_matrix(true_labels, pred_labels, num_classes)

    per_class_acc = {}
    per_class_precision = {}
    per_class_recall = {}
    per_class_f1 = {}

    for c in range(num_classes):
        # Per-class accuracy = TP+TN / total
        tp = cm[c, c]
        class_total = cm[c, :].sum()
        per_class_acc[c] = float(tp / class_total) if class_total > 0 else float("nan")

        # Precision = TP / (TP + FP)
        tp_fp = cm[:, c].sum()
        prec = float(tp / tp_fp) if tp_fp > 0 else 0.0
        per_class_precision[c] = prec

        # Recall = TP / (TP + FN)
        tp_fn = cm[c, :].sum()
        rec = float(tp / tp_fn) if tp_fn > 0 else 0.0
        per_class_recall[c] = rec

        # F1
        if prec + rec > 0:
            per_class_f1[c] = 2 * prec * rec / (prec + rec)
        else:
            per_class_f1[c] = 0.0

    macro_precision = float(np.mean(list(per_class_precision.values())))
    macro_recall = float(np.mean(list(per_class_recall.values())))
    macro_f1 = float(np.mean(list(per_class_f1.values())))

    class_counts = {c: int((true_arr == c).sum()) for c in range(num_classes)}

    return {
        "accuracy": overall_acc,
        "per_class_accuracy": per_class_acc,
        "per_class_precision": per_class_precision,
        "per_class_recall": per_class_recall,
        "per_class_f1": per_class_f1,
        "confusion_matrix": cm,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "class_counts": class_counts,
        "total_samples": total,
    }


def validate_labels(labels: List[int], valid_set: set = {0, 1, 2}) -> bool:
    """Return True if all labels are in the valid set."""
    return all(int(l) in valid_set for l in labels)
