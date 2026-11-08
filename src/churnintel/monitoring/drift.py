"""Data and prediction drift monitoring with the Population Stability Index (PSI).

PSI = sum_i (cur_i - ref_i) * ln(cur_i / ref_i). Common reading: < 0.10 stable,
0.10-0.25 moderate shift (investigate), > 0.25 significant shift (consider retraining).
Only compact reference summaries (bin edges + proportions) are stored, never raw training rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

_EPS = 1e-4


def population_stability_index(ref_props: np.ndarray, cur_props: np.ndarray) -> float:
    ref = np.clip(np.asarray(ref_props, dtype=float), _EPS, None)
    cur = np.clip(np.asarray(cur_props, dtype=float), _EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


class DriftMonitor:
    def __init__(self, bins: int = 10, warn: float = 0.10, alert: float = 0.25):
        self.bins = bins
        self.warn = warn
        self.alert = alert

    def _summarize(self, s: pd.Series) -> dict:
        numeric = pd.api.types.is_numeric_dtype(s) and s.nunique(dropna=True) > 10
        if numeric:
            values = s.dropna().to_numpy(dtype=float)
            edges = np.unique(np.quantile(values, np.linspace(0, 1, self.bins + 1)))
            edges[0], edges[-1] = -np.inf, np.inf
            props = np.histogram(values, edges)[0] / max(len(values), 1)
            return {"type": "numeric", "edges": edges.tolist(), "props": props.tolist()}
        props = s.astype(str).value_counts(normalize=True)
        return {"type": "categorical", "props": props.to_dict()}

    def fit(self, X_ref: pd.DataFrame, scores_ref: np.ndarray | None = None) -> DriftMonitor:
        self.reference_ = {c: self._summarize(X_ref[c]) for c in X_ref.columns}
        if scores_ref is not None:
            self.reference_["__churn_score__"] = self._summarize(
                pd.Series(np.asarray(scores_ref, dtype=float))
            )
        self.n_reference_ = len(X_ref)
        return self

    def _psi_one(self, ref: dict, s: pd.Series) -> float:
        if ref["type"] == "numeric":
            values = pd.to_numeric(s, errors="coerce").dropna().to_numpy(dtype=float)
            cur = np.histogram(values, np.asarray(ref["edges"]))[0] / max(len(values), 1)
            return population_stability_index(np.asarray(ref["props"]), cur)
        cur = s.astype(str).value_counts(normalize=True)
        levels = sorted(set(ref["props"]) | set(cur.index))
        r = np.array([ref["props"].get(k, 0.0) for k in levels])
        c = np.array([cur.get(k, 0.0) for k in levels])
        return population_stability_index(r, c)

    def status(self, psi: float) -> str:
        return "alert" if psi >= self.alert else ("warn" if psi >= self.warn else "ok")

    def compute(self, X_cur: pd.DataFrame, scores_cur: np.ndarray | None = None) -> pd.DataFrame:
        rows = []
        for col, ref in self.reference_.items():
            if col == "__churn_score__":
                if scores_cur is None:
                    continue
                series = pd.Series(np.asarray(scores_cur, dtype=float))
            elif col in X_cur.columns:
                series = X_cur[col]
            else:
                continue
            psi = self._psi_one(ref, series)
            rows.append({"feature": col, "type": ref["type"], "psi": psi, "status": self.status(psi)})
        return pd.DataFrame(rows).sort_values("psi", ascending=False).reset_index(drop=True)
