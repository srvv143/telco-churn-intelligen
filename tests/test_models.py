import numpy as np
import pandas as pd
import pytest

from churnintel.explain import ChurnExplainer
from churnintel.models import build_ensemble, build_pipeline, select_champion, summarize_cv, tune_model
from churnintel.models.calibration import base_pipeline_of, make_calibrated


def _folds(values):
    return pd.DataFrame(
        {"fold": range(len(values)), "roc_auc": values, "pr_auc": values, "brier": 0.1, "log_loss": 0.4}
    )


@pytest.fixture()
def cv_tables():
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 0.012, 15)  # shared fold difficulty
    folds = {
        "logreg": _folds(0.850 + noise),
        "lightgbm": _folds(0.849 + noise + rng.normal(0, 0.001, 15)),
        "stacking": _folds(0.8507 + noise),  # tiny but perfectly consistent gain
        "random_forest": _folds(0.840 + noise),
    }
    return {k: summarize_cv(v) for k, v in folds.items()}, folds


ORDER = ["logreg", "lightgbm", "random_forest", "stacking"]


def test_best_rule_picks_highest_mean(cv_tables):
    summaries, folds = cv_tables
    champ, _ = select_champion(summaries, folds, "best", ORDER)
    assert champ == "stacking"


def test_one_se_rule_prefers_simpler_model_within_noise(cv_tables):
    summaries, folds = cv_tables
    champ, details = select_champion(summaries, folds, "one_se", ORDER)
    assert champ == "logreg"
    assert "random_forest" not in details["eligible"]
    assert details["paired_vs_best"]["logreg"]["mean_diff"] < 0


def test_paired_rule_detects_consistent_gain(cv_tables):
    summaries, folds = cv_tables
    champ, _ = select_champion(summaries, folds, "one_se_paired", ORDER)
    assert champ == "stacking"


def test_tuning_prefers_regularised_logreg(split):
    X_train, _, y_train, _ = split
    Xs, ys = X_train.iloc[:1500], y_train.iloc[:1500]
    res = tune_model("logreg", Xs, ys, n_trials=6, cv_splits=3, fe_kwargs={"exclude": ["gender"]})
    assert res.n_trials == 6
    assert 1e-3 <= res.best_params["C"] <= 1e2
    best_trial_c = res.history.sort_values("value", ascending=False).iloc[0]["params_C"]
    assert res.best_params["C"] <= best_trial_c + 1e-12


def test_calibrated_wrapper_exposes_base_pipeline(split):
    X_train, X_test, y_train, _ = split
    model = make_calibrated(build_pipeline("logreg", {"C": 0.1}), "sigmoid", cv=3).fit(X_train, y_train)
    base = base_pipeline_of(model)
    assert hasattr(base, "named_steps") and "preprocess" in base.named_steps
    # sigmoid calibration is a monotone map of the base score -> identical ranking
    p_cal = model.predict_proba(X_test)[:, 1]
    p_base = base.predict_proba(X_test)[:, 1]
    assert np.corrcoef(np.argsort(np.argsort(p_cal)), np.argsort(np.argsort(p_base)))[0, 1] > 0.9999


def test_agnostic_explainer_for_ensembles(split):
    X_train, X_test, y_train, _ = split
    members = {"logreg": {"C": 0.1}, "lightgbm": {"n_estimators": 50}}
    ens = build_ensemble("soft_voting", members, {"exclude": ["gender"]}).fit(X_train, y_train)
    ex = ChurnExplainer(ens, X_train, n_background=60)
    assert ex.kind_ == "agnostic"
    reasons = ex.reasons(X_test.head(2), top_k=2)
    assert len(reasons) == 2 and len(reasons[0]["risk_drivers"]) <= 2
