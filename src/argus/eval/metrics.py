"""Evaluation metrics. Accuracy is never reported: at ~0.1% laundering a model that
flags nothing scores 99.9%. We report PR-AUC, minority-class F1/precision/recall,
Ring Recall, and the robustness deltas D and R."""

import numpy as np
from sklearn.metrics import (average_precision_score, f1_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)


def best_f1_threshold(y_true, scores) -> float:
    """Threshold that maximises F1 — choose it on validation, apply it to test."""
    p, r, thr = precision_recall_curve(y_true, scores)
    f1 = 2 * p * r / np.clip(p + r, 1e-12, None)
    return float(thr[np.argmax(f1[:-1])]) if len(thr) else 0.5


def ring_recall(ring_ids, y_pred, min_share: float = 1.0) -> float:
    """Share of known rings with at least `min_share` of their transfers flagged.

    ring_ids: ring id per transaction (-1 = not in a known ring).
    """
    ring_ids = np.asarray(ring_ids)
    y_pred = np.asarray(y_pred).astype(bool)
    mask = ring_ids >= 0
    if not mask.any():
        return float("nan")
    ids, inv = np.unique(ring_ids[mask], return_inverse=True)
    share = np.bincount(inv, weights=y_pred[mask]) / np.bincount(inv)
    return float((share >= min_share - 1e-12).mean())


def evaluate(y_true, scores, threshold, ring_ids=None, ring_thresholds=(1.0, 0.5)) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(scores) >= threshold).astype(int)
    out = {
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if 0 < y_true.sum() < len(y_true) else float("nan"),
        "threshold": float(threshold),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    if ring_ids is not None:
        for s in ring_thresholds:
            out[f"ring_recall@{s:g}"] = ring_recall(ring_ids, y_pred, s)
    return out


def degradation(f1_baseline: float, f1_attack: float) -> float:
    """D = F1_baseline - F1_attack: what an adaptive adversary costs."""
    return f1_baseline - f1_attack


def recovery(f1_retrained: float, f1_attack: float) -> float:
    """R = F1_retrained - F1_attack: what retraining wins back."""
    return f1_retrained - f1_attack
