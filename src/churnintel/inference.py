"""Serving-side scoring used by the API, the Streamlit app and batch jobs.

A *model bundle* (one joblib file) holds everything needed at inference time: the calibrated
champion, its explainer, the survival-CLV model, the action menu, thresholds, the drift reference
and metadata. Training and serving therefore cannot drift apart.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from .business.economics import RetentionAction, next_best_action, risk_tier
from .data.loader import clean
from .data.schema import ID_COLUMN, RAW_FEATURES

logger = logging.getLogger(__name__)

NO_ACTION = "no_action"


class ChurnPredictor:
    def __init__(self, bundle: dict[str, Any]):
        self.bundle = bundle
        self.model = bundle["model"]
        self.explainer = bundle.get("explainer")
        self.survival = bundle["survival"]
        self.actions = [RetentionAction(**a) for a in bundle["actions"]]
        self.risk_tiers = bundle["risk_tiers"]
        self.metadata = bundle.get("metadata", {})

    @classmethod
    def load(cls, path: str | Path) -> ChurnPredictor:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Model bundle not found at {path}. Train first: `make train`.")
        logger.info("Loading model bundle from %s", path)
        return cls(joblib.load(path))

    # ------------------------------------------------------------------ input handling
    def prepare(self, data: pd.DataFrame | list[dict] | dict) -> pd.DataFrame:
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data.copy()
        if ID_COLUMN in df.columns:
            ids = df[ID_COLUMN].astype(object).where(df[ID_COLUMN].notna(), None)
            df = df.drop(columns=[ID_COLUMN])
            df.index = pd.Index(
                [i if i is not None else f"row_{k}" for k, i in enumerate(ids)], name=ID_COLUMN
            )
        for col in RAW_FEATURES:
            if col not in df.columns:
                df[col] = np.nan
        return clean(df[RAW_FEATURES])

    # ------------------------------------------------------------------ scoring
    def predict_proba(self, data) -> np.ndarray:
        X = self.prepare(data)
        return self.model.predict_proba(X)[:, 1]

    def _proba_and_features(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray | None]:
        """Score ``X``; for an uncalibrated standard pipeline also return the transformed matrix so the
        explainer does not have to run feature engineering a second time."""
        m = self.model
        if hasattr(m, "named_steps") and "preprocess" in m.named_steps:
            Xt = m[:-1].transform(X)
            return m[-1].predict_proba(Xt)[:, 1], Xt
        return m.predict_proba(X)[:, 1], None

    def score(self, data, explain: bool = False, top_k: int = 3) -> pd.DataFrame:
        X = self.prepare(data)
        p, Xt = self._proba_and_features(X)
        remaining = self.survival.expected_remaining_months(X)
        clv = self.survival.clv(X, remaining=remaining)
        action_idx, ev = next_best_action(p, clv, self.actions)
        out = pd.DataFrame(
            {
                "churn_probability": np.round(p, 4),
                "risk_tier": risk_tier(p, self.risk_tiers["high"], self.risk_tiers["medium"]),
                "expected_remaining_months": np.round(remaining, 2),
                "clv_estimate": np.round(clv, 2),
                "recommended_action": [self.actions[i].name if i >= 0 else NO_ACTION for i in action_idx],
                "action_description": [
                    self.actions[i].label
                    if i >= 0
                    else "No proactive retention spend (negative expected value)"
                    for i in action_idx
                ],
                "expected_value": np.round(ev, 2),
            },
            index=X.index,
        )
        if explain and self.explainer is not None:
            out["explanation"] = self.explainer.reasons(X, top_k=top_k, Xt=Xt)
        return out

    def drift_report(self, data) -> pd.DataFrame:
        X = self.prepare(data)
        monitor = self.bundle["drift_monitor"]
        return monitor.compute(X, self.model.predict_proba(X)[:, 1])
