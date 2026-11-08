"""DeLong test for comparing two correlated ROC-AUCs on the same test set.

Fast O(n log n) algorithm of Sun & Xu (2014), "Fast implementation of DeLong's algorithm for
comparing the areas under correlated receiver operating characteristic curves",
IEEE Signal Processing Letters 21(11). Original test: DeLong, DeLong & Clarke-Pearson (1988).
"""

from __future__ import annotations

import numpy as np
from scipy import stats


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def _fast_delong(preds_sorted: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """``preds_sorted``: (k classifiers, n samples) with the m positives first."""
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, m + n))
    for r in range(k):
        tx[r] = _midrank(pos[r])
        ty[r] = _midrank(neg[r])
        tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01))
    sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n


def _prepare(y: np.ndarray, *scores: np.ndarray) -> tuple[np.ndarray, int]:
    y = np.asarray(y).astype(int)
    order = np.argsort(-y, kind="mergesort")  # positives first, stable
    preds = np.vstack([np.asarray(s, dtype=float) for s in scores])[:, order]
    return preds, int(y.sum())


def delong_roc_variance(y: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """Return ``(auc, variance)`` of a single model's AUC."""
    preds, m = _prepare(y, score)
    aucs, cov = _fast_delong(preds, m)
    return float(aucs[0]), float(cov[0, 0])


def delong_roc_test(y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray) -> dict[str, float]:
    """Two-sided DeLong test of H0: AUC_a == AUC_b."""
    preds, m = _prepare(y, score_a, score_b)
    aucs, cov = _fast_delong(preds, m)
    diff = float(aucs[0] - aucs[1])
    var = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    z = diff / np.sqrt(var) if var > 0 else 0.0
    p_value = float(2 * stats.norm.sf(abs(z)))
    se = float(np.sqrt(max(var, 0.0)))
    return {
        "auc_a": float(aucs[0]),
        "auc_b": float(aucs[1]),
        "auc_diff": diff,
        "diff_ci_lower": float(diff - 1.959964 * se),
        "diff_ci_upper": float(diff + 1.959964 * se),
        "z": float(z),
        "p_value": p_value,
    }
