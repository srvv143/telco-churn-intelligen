"""Repeated stratified cross-validation and champion selection (one-standard-error rule)."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

from ..evaluation.metrics import expected_calibration_error

logger = logging.getLogger(__name__)


def cv_evaluate(
    factory: Callable[[], object],
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Fit ``factory()`` on every fold of a repeated stratified K-fold.

    Returns per-fold metrics and out-of-fold probabilities from the first repetition.
    The same ``random_state`` gives every model identical folds (paired comparison).
    """
    rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=random_state)
    oof = np.full(len(y), np.nan)
    rows = []
    for fold, (tr, va) in enumerate(rskf.split(X, y)):
        start = time.perf_counter()
        model = factory()
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[va])[:, 1]
        y_va = y.iloc[va].to_numpy()
        rows.append(
            {
                "fold": fold,
                "repeat": fold // n_splits,
                "roc_auc": roc_auc_score(y_va, proba),
                "pr_auc": average_precision_score(y_va, proba),
                "brier": brier_score_loss(y_va, proba),
                "log_loss": log_loss(y_va, np.clip(proba, 1e-6, 1 - 1e-6)),
                "ece": expected_calibration_error(y_va, proba),
                "fit_seconds": time.perf_counter() - start,
            }
        )
        if fold < n_splits:
            oof[va] = proba
    return pd.DataFrame(rows), oof


def summarize_cv(folds: pd.DataFrame) -> dict[str, float]:
    n = len(folds)
    out: dict[str, float] = {"n_folds": n}
    for metric in ("roc_auc", "pr_auc", "brier", "log_loss", "ece", "fit_seconds"):
        if metric not in folds:
            continue
        out[f"{metric}_mean"] = float(folds[metric].mean())
        out[f"{metric}_std"] = float(folds[metric].std(ddof=1)) if n > 1 else 0.0
    out["roc_auc_se"] = float(out["roc_auc_std"] / np.sqrt(n)) if n > 1 else 0.0
    return out


def select_champion(
    summaries: dict[str, dict[str, float]],
    folds: dict[str, pd.DataFrame],
    rule: str = "one_se",
    complexity_order: list[str] | None = None,
) -> tuple[str, dict]:
    """Pick the champion model.

    ``best``          : highest mean CV ROC-AUC.
    ``one_se``        : ESL §7.10 one-standard-error rule — the *simplest* model (by ``complexity_order``)
                        whose mean CV AUC is >= best mean - SE(best mean). A parsimony test: is the extra
                        complexity worth more than the estimation noise of the CV score itself?
    ``one_se_paired`` : same, but with the SE of the paired per-fold difference (identical folds). Much
                        stricter: it detects tiny-but-consistent gains, i.e. statistical, not practical,
                        significance.

    Paired differences vs. the best model are always computed and returned for transparency.
    """
    best = max(summaries, key=lambda k: summaries[k]["roc_auc_mean"])
    best_mean = summaries[best]["roc_auc_mean"]
    ref = folds[best].sort_values("fold")["roc_auc"].to_numpy()
    paired: dict[str, dict[str, float]] = {}
    for m in summaries:
        d = folds[m].sort_values("fold")["roc_auc"].to_numpy() - ref
        se = float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0
        wins = float((d > 0).mean())
        paired[m] = {"mean_diff": float(d.mean()), "se_diff": se, "share_of_folds_better": wins}
    details: dict = {"best_model": best, "best_cv_auc": best_mean, "rule": rule, "paired_vs_best": paired}
    if rule == "best":
        details["eligible"] = [best]
        return best, details

    if rule == "one_se_paired":
        eligible = [m for m, v in paired.items() if v["mean_diff"] >= -v["se_diff"]]
        cutoff = None
    else:
        se_best = summaries[best]["roc_auc_se"]
        cutoff = best_mean - se_best
        eligible = [m for m, s in summaries.items() if s["roc_auc_mean"] >= cutoff]
        details["best_se"] = se_best
    order = complexity_order or list(summaries)
    ranked = [m for m in order if m in eligible] + [m for m in eligible if m not in order]
    champion = ranked[0]
    details.update({"eligible": ranked, "cutoff": cutoff})
    logger.info(
        "Selection (%s): best=%s %.4f -> eligible %s -> champion=%s (paired Δ vs best %.4f ± %.4f)",
        rule,
        best,
        best_mean,
        ranked,
        champion,
        paired[champion]["mean_diff"],
        paired[champion]["se_diff"],
    )
    return champion, details
