"""Probability calibration.

Expected-value decisions (``p x save_rate x CLV - cost``) are only as good as the probabilities,
so the champion is calibrated and the method (none / Platt sigmoid / isotonic) is chosen by
out-of-fold Brier score using *nested* cross-validation on the training data.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from ..evaluation.metrics import expected_calibration_error

logger = logging.getLogger(__name__)

METHODS = ("none", "sigmoid", "isotonic")


def make_calibrated(estimator, method: str, cv: int = 5, random_state: int = 42):
    """``ensemble=False``: one base model refit on all data + one calibrator learned from
    cross-validated predictions. The served model is therefore a monotone transform of a single
    pipeline, which keeps SHAP explanations of that pipeline faithful to the ranking."""
    if method == "none":
        return estimator
    return CalibratedClassifierCV(
        estimator=estimator,
        method=method,
        cv=StratifiedKFold(cv, shuffle=True, random_state=random_state),
        ensemble=False,
    )


def base_pipeline_of(model):
    """Return the fitted, uncalibrated pipeline inside a (possibly) calibrated model."""
    if isinstance(model, CalibratedClassifierCV):
        inner = model.calibrated_classifiers_[0]
        est = getattr(inner, "estimator", None)
        return est if est is not None else inner.base_estimator
    return model


def calibrated_oof(
    factory: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    method: str,
    outer_splits: int = 5,
    inner_splits: int = 5,
    random_state: int = 42,
) -> np.ndarray:
    outer = StratifiedKFold(outer_splits, shuffle=True, random_state=random_state)
    oof = np.zeros(len(y))
    for tr, va in outer.split(X, y):
        model = make_calibrated(factory(), method, inner_splits, random_state)
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    return oof


def select_calibration(
    factory: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    method: str = "auto",
    cv: int = 5,
    random_state: int = 42,
) -> tuple[str, pd.DataFrame, dict[str, np.ndarray]]:
    """Compare calibration methods on identical outer folds; return the winner by Brier score."""
    methods = METHODS if method == "auto" else ("none", method) if method != "none" else ("none",)
    rows, oofs = [], {}
    for m in methods:
        oof = calibrated_oof(factory, X, y, m, cv, cv, random_state)
        oofs[m] = oof
        rows.append(
            {
                "method": m,
                "brier": brier_score_loss(y, oof),
                "log_loss": log_loss(y, np.clip(oof, 1e-6, 1 - 1e-6)),
                "ece": expected_calibration_error(y.to_numpy(), oof),
                "roc_auc": roc_auc_score(y, oof),
            }
        )
    table = pd.DataFrame(rows).set_index("method")
    chosen = method if method not in ("auto",) else str(table["brier"].idxmin())
    logger.info("Calibration comparison (OOF on train):\n%s\n-> using '%s'", table.round(4), chosen)
    return chosen, table, oofs
