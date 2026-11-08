"""Stratified bootstrap confidence intervals for test-set metrics."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd


def bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    metrics: dict[str, Callable[[np.ndarray, np.ndarray], float]],
    n_boot: int = 2000,
    ci: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """Percentile CIs from a class-stratified bootstrap (keeps the churn rate fixed per replicate)."""
    y = np.asarray(y)
    p = np.asarray(p, dtype=float)
    rng = np.random.default_rng(random_state)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    draws = {name: np.empty(n_boot) for name in metrics}
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        for name, fn in metrics.items():
            draws[name][b] = fn(y[idx], p[idx])
    alpha = (1 - ci) / 2
    rows = []
    for name, fn in metrics.items():
        d = draws[name]
        rows.append(
            {
                "metric": name,
                "estimate": float(fn(y, p)),
                "ci_lower": float(np.quantile(d, alpha)),
                "ci_upper": float(np.quantile(d, 1 - alpha)),
                "std": float(d.std(ddof=1)),
            }
        )
    return pd.DataFrame(rows).set_index("metric")
