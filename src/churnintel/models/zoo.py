"""Model zoo: estimator factories, Optuna search spaces and ensemble builders.

Every model is wrapped in the same ``features -> preprocess -> model`` pipeline, so tuning,
cross-validation, calibration and serving all see identical preprocessing.
"""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from ..features import TelcoFeatureEngineer, build_preprocessor

MODEL_LABELS: dict[str, str] = {
    "logreg": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
    "soft_voting": "Soft-Voting Ensemble",
    "stacking": "Stacking Ensemble",
}

#: Sensible starting points; each is enqueued as Optuna's first trial so tuning never ends worse.
DEFAULT_PARAMS: dict[str, dict[str, Any]] = {
    "logreg": {"C": 1.0},
    "random_forest": {"n_estimators": 400, "max_depth": 10, "min_samples_leaf": 5, "max_features": 0.3},
    "xgboost": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 1.0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "gamma": 1e-3,
        "reg_alpha": 1e-3,
        "reg_lambda": 1.0,
    },
    "lightgbm": {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_child_samples": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1e-3,
        "reg_lambda": 1e-3,
    },
    "catboost": {"iterations": 500, "learning_rate": 0.05, "depth": 5, "l2_leaf_reg": 3.0},
}

#: Parameters that are fixed (not searched) but required.
FIXED_PARAMS: dict[str, dict[str, Any]] = {"lightgbm": {"subsample_freq": 1}}


def suggest_params(trial, name: str) -> dict[str, Any]:
    """Optuna search space for each model family."""
    if name == "logreg":
        return {"C": trial.suggest_float("C", 1e-3, 1e2, log=True)}
    if name == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=100),
            "max_depth": trial.suggest_int("max_depth", 4, 16),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "max_features": trial.suggest_float("max_features", 0.1, 0.8),
        }
    if name == "xgboost":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 20.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "gamma": trial.suggest_float("gamma", 1e-8, 5.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
    if name == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 64, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 150, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }
    if name == "catboost":
        return {
            "iterations": trial.suggest_int("iterations", 200, 600, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 6),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        }
    raise ValueError(f"Unknown model family: {name}")


def build_estimator(
    name: str, params: dict[str, Any] | None = None, random_state: int = 42, n_jobs: int = -1
):
    params = {**DEFAULT_PARAMS.get(name, {}), **FIXED_PARAMS.get(name, {}), **(params or {})}
    if name == "logreg":
        return LogisticRegression(C=params["C"], max_iter=5000, solver="lbfgs")
    if name == "random_forest":
        return RandomForestClassifier(**params, n_jobs=n_jobs, random_state=random_state)
    if name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(
            **params,
            tree_method="hist",
            eval_metric="logloss",
            n_jobs=None if n_jobs == -1 else n_jobs,
            random_state=random_state,
            verbosity=0,
        )
    if name == "lightgbm":
        from lightgbm import LGBMClassifier

        return LGBMClassifier(**params, n_jobs=n_jobs, random_state=random_state, verbose=-1)
    if name == "catboost":
        from catboost import CatBoostClassifier

        return CatBoostClassifier(
            **params,
            random_seed=random_state,
            thread_count=n_jobs,
            verbose=0,
            allow_writing_files=False,
        )
    raise ValueError(f"Unknown model family: {name}")


def build_pipeline(
    name: str,
    params: dict[str, Any] | None = None,
    fe_kwargs: dict[str, Any] | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
) -> Pipeline:
    fe = TelcoFeatureEngineer(**(fe_kwargs or {}))
    numeric, nominal = fe.output_columns()
    return Pipeline(
        [
            ("features", fe),
            ("preprocess", build_preprocessor(numeric, nominal)),
            ("model", build_estimator(name, params, random_state, n_jobs)),
        ]
    )


def build_ensemble(
    kind: str,
    members: dict[str, dict[str, Any]],
    fe_kwargs: dict[str, Any] | None = None,
    random_state: int = 42,
    n_jobs: int = -1,
):
    """Soft-voting or stacking over full member pipelines (each keeps its own preprocessing)."""
    estimators = [(n, build_pipeline(n, p, fe_kwargs, random_state, n_jobs)) for n, p in members.items()]
    if kind == "soft_voting":
        return VotingClassifier(estimators=estimators, voting="soft")
    if kind == "stacking":
        return StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(C=1.0, max_iter=2000),
            cv=StratifiedKFold(5, shuffle=True, random_state=random_state),
            stack_method="predict_proba",
        )
    raise ValueError(f"Unknown ensemble kind: {kind}")
