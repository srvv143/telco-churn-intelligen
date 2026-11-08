"""Bayesian hyper-parameter search (Optuna TPE) with median pruning on CV folds."""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .zoo import DEFAULT_PARAMS, FIXED_PARAMS, build_pipeline, suggest_params

logger = logging.getLogger(__name__)


@dataclass
class TuningResult:
    name: str
    best_params: dict[str, Any]
    best_cv_auc: float
    n_trials: int
    n_pruned: int
    seconds: float
    history: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_params": self.best_params,
            "best_cv_auc": round(self.best_cv_auc, 5),
            "n_trials": self.n_trials,
            "n_pruned": self.n_pruned,
            "seconds": round(self.seconds, 1),
        }


def tune_model(
    name: str,
    X: pd.DataFrame,
    y: pd.Series,
    n_trials: int,
    cv_splits: int = 5,
    random_state: int = 42,
    fe_kwargs: dict[str, Any] | None = None,
    timeout: float | None = None,
    n_jobs: int = -1,
) -> TuningResult:
    """Maximise mean stratified-CV ROC-AUC on the *training* data only."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(X, y))

    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, name)
        scores: list[float] = []
        for step, (tr, va) in enumerate(folds):
            model = build_pipeline(name, params, fe_kwargs, random_state, n_jobs)
            model.fit(X.iloc[tr], y.iloc[tr])
            proba = model.predict_proba(X.iloc[va])[:, 1]
            scores.append(roc_auc_score(y.iloc[va], proba))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr("fold_scores", scores)
        return float(np.mean(scores))

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1),
        study_name=f"tune_{name}",
    )
    study.enqueue_trial(DEFAULT_PARAMS[name])
    start = time.perf_counter()
    study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
    seconds = time.perf_counter() - start

    states = [t.state.name for t in study.trials]
    history = study.trials_dataframe(attrs=("number", "value", "state", "params"))
    chosen = study.best_trial
    if name in SIMPLICITY_PARAM:
        chosen = paired_one_se_trial(study, *SIMPLICITY_PARAM[name])
    result = TuningResult(
        name=name,
        best_params={**FIXED_PARAMS.get(name, {}), **chosen.params},
        best_cv_auc=float(chosen.value),
        n_trials=len(study.trials),
        n_pruned=states.count("PRUNED"),
        seconds=seconds,
        history=history,
    )
    logger.info(
        "Tuned %-13s CV AUC=%.4f (best trial %.4f)  (%d trials, %d pruned, %.0fs) params=%s",
        name,
        result.best_cv_auc,
        float(study.best_value),
        result.n_trials,
        result.n_pruned,
        seconds,
        {k: (round(v, 4) if isinstance(v, float) else v) for k, v in chosen.params.items()},
    )
    return result


#: For these families, prefer the simplest setting that is statistically tied with the best trial
#: (glmnet's "lambda.1se" idea). For logistic regression: the smallest C = strongest L2 penalty.
SIMPLICITY_PARAM: dict[str, tuple[str, str]] = {"logreg": ("C", "min")}


def paired_one_se_trial(study, param: str, direction: str = "min"):
    """Among completed trials, return the simplest one whose fold-wise AUC is within one *paired*
    standard error of the best trial. All trials share the same folds, so the SE of the per-fold
    difference is far tighter (and more appropriate) than the SE of either mean."""
    best = study.best_trial
    best_scores = np.asarray(best.user_attrs["fold_scores"])
    eligible = []
    for t in study.trials:
        scores = t.user_attrs.get("fold_scores")
        if t.state.name != "COMPLETE" or scores is None or len(scores) != len(best_scores):
            continue
        d = np.asarray(scores) - best_scores
        se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else 0.0
        if d.mean() >= -se:
            eligible.append(t)
    key = (lambda t: t.params[param]) if direction == "min" else (lambda t: -t.params[param])
    return min(eligible, key=key) if eligible else best
