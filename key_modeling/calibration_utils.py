from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def safe_logit(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    return np.log(p) - np.log1p(-p)


def softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    z = x - np.max(x, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def probabilities_from_candidate_scores(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    logits = safe_logit(scores)
    return softmax(logits / float(temperature), axis=1)


def multiclass_nll(probs: np.ndarray, y_true: np.ndarray, eps: float = 1e-12) -> float:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    chosen = np.clip(probs[np.arange(len(y_true)), y_true], eps, 1.0)
    return float(-np.mean(np.log(chosen)))


def multiclass_brier(probs: np.ndarray, y_true: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    onehot = np.zeros_like(probs)
    onehot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probs - onehot) ** 2, axis=1)))


def top1_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> tuple[float, list[dict]]:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    correct = (pred == y_true).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: list[dict] = []
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i == n_bins - 1:
            m = (conf >= lo) & (conf <= hi)
        else:
            m = (conf >= lo) & (conf < hi)
        count = int(np.sum(m))
        if count == 0:
            continue
        mean_conf = float(np.mean(conf[m]))
        acc = float(np.mean(correct[m]))
        ece += (count / n) * abs(acc - mean_conf)
        rows.append({
            "lower": lo,
            "upper": hi,
            "count": count,
            "mean_confidence": mean_conf,
            "accuracy": acc,
            "gap": acc - mean_conf,
        })
    return float(ece), rows


def probability_metrics(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> dict:
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    acc = float(np.mean(pred == y_true))
    ece, bins = top1_ece(probs, y_true, n_bins=n_bins)
    return {
        "top1_accuracy": acc,
        "nll": multiclass_nll(probs, y_true),
        "brier": multiclass_brier(probs, y_true),
        "ece": ece,
        "mean_top1_confidence": float(np.mean(conf)),
        "confidence_accuracy_gap": float(np.mean(conf) - acc),
        "reliability_bins": bins,
    }


def coverage_accuracy(probs: np.ndarray, y_true: np.ndarray, thresholds: Iterable[float] = (0.50, 0.60, 0.70, 0.80, 0.90)) -> list[dict]:
    probs = np.asarray(probs, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    pred = np.argmax(probs, axis=1)
    conf = np.max(probs, axis=1)
    rows = []
    for threshold in thresholds:
        m = conf >= float(threshold)
        count = int(np.sum(m))
        rows.append({
            "threshold": float(threshold),
            "coverage": float(np.mean(m)),
            "count": count,
            "accuracy": float(np.mean(pred[m] == y_true[m])) if count else None,
        })
    return rows
