"""Ranking, calibration, threshold and targeting metrics for binary churn models."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """ECE with equal-width bins: sum_b |B_b|/N * |mean(y_b) - mean(p_b)|."""
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        mask = bins == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def ks_statistic(y: np.ndarray, p: np.ndarray) -> float:
    """Kolmogorov-Smirnov separation = max(TPR - FPR)."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def lift_at(y: np.ndarray, p: np.ndarray, fraction: float) -> dict[str, float]:
    """Precision, lift and churner capture rate when contacting the top ``fraction`` by score."""
    y = np.asarray(y)
    k = max(1, math.ceil(fraction * len(y)))
    top = np.argsort(-np.asarray(p), kind="mergesort")[:k]
    precision = float(y[top].mean())
    return {
        "fraction": fraction,
        "k": int(k),
        "precision": precision,
        "lift": precision / float(y.mean()),
        "capture_rate": float(y[top].sum() / y.sum()),
    }


def threshold_for_f1(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    return float(thr[int(np.argmax(f1))])


def threshold_metrics(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float]:
    pred = (np.asarray(p) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "specificity": float(tn / max(tn + fp, 1)),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
    }


def classification_metrics(
    y: np.ndarray, p: np.ndarray, threshold: float = 0.5, lift_fractions=(0.1, 0.2, 0.3)
) -> dict[str, object]:
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    out: dict[str, object] = {
        "n": int(len(y)),
        "base_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "brier_skill": float(1 - brier_score_loss(y, p) / (y.mean() * (1 - y.mean()))),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
        "ece": expected_calibration_error(y, p),
        "ks": ks_statistic(y, p),
    }
    out["at_threshold"] = threshold_metrics(y, p, threshold)
    out["lift"] = {f"top_{int(f * 100)}pct": lift_at(y, p, f) for f in lift_fractions}
    return out


def gains_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Decile table: churn rate, lift and cumulative capture per score decile (1 = riskiest)."""
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df = df.sort_values("p", ascending=False, kind="mergesort").reset_index(drop=True)
    df["decile"] = (np.arange(len(df)) * n_bins // len(df)) + 1
    base = df["y"].mean()
    g = df.groupby("decile").agg(
        customers=("y", "size"),
        churners=("y", "sum"),
        churn_rate=("y", "mean"),
        min_score=("p", "min"),
        max_score=("p", "max"),
    )
    g["lift"] = g["churn_rate"] / base
    g["cum_capture"] = g["churners"].cumsum() / df["y"].sum()
    g["cum_customers_pct"] = g["customers"].cumsum() / len(df)
    return g.reset_index()
