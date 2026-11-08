"""Group-wise performance audit (equal opportunity, false-positive parity, selection rates)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


def _group_row(y: np.ndarray, p: np.ndarray, d: np.ndarray) -> dict[str, float]:
    tp = int(((d == 1) & (y == 1)).sum())
    fp = int(((d == 1) & (y == 0)).sum())
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return {
        "n": int(len(y)),
        "churn_rate": float(y.mean()),
        "mean_score": float(p.mean()),
        "roc_auc": float(roc_auc_score(y, p)) if 0 < pos < len(y) else float("nan"),
        "tpr": tp / pos if pos else float("nan"),
        "fpr": fp / neg if neg else float("nan"),
        "precision": tp / (tp + fp) if (tp + fp) else float("nan"),
        "selection_rate": float(d.mean()),
    }


def fairness_report(
    y: np.ndarray, p: np.ndarray, decision: np.ndarray, groups: dict[str, pd.Series]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-group metrics and disparity summary for each sensitive attribute.

    Disparities: TPR gap (equal opportunity), FPR gap, AUC gap and the selection-rate
    ratio min/max (the "four-fifths" disparate-impact heuristic flags ratios < 0.8).
    """
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    decision = np.asarray(decision).astype(int)
    rows, summary = [], []
    for attr, values in groups.items():
        values = np.asarray(values)
        attr_rows = []
        for level in pd.unique(values):
            mask = values == level
            row = {"attribute": attr, "group": str(level), **_group_row(y[mask], p[mask], decision[mask])}
            attr_rows.append(row)
        rows.extend(attr_rows)
        frame = pd.DataFrame(attr_rows)
        sel = frame["selection_rate"]
        summary.append(
            {
                "attribute": attr,
                "tpr_gap": float(frame["tpr"].max() - frame["tpr"].min()),
                "fpr_gap": float(frame["fpr"].max() - frame["fpr"].min()),
                "auc_gap": float(frame["roc_auc"].max() - frame["roc_auc"].min()),
                "selection_rate_ratio": float(sel.min() / sel.max()) if sel.max() > 0 else float("nan"),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary)
