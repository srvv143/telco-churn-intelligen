import numpy as np
import pandas as pd
import pytest

from churnintel.explain import ChurnExplainer, concept_of
from churnintel.explain.explainer import describe
from churnintel.models import build_pipeline
from churnintel.monitoring import DriftMonitor, population_stability_index
from churnintel.survival import SurvivalCLV


@pytest.fixture(scope="module")
def surv(split):
    X_train, _, y_train, _ = split
    return SurvivalCLV(horizon_months=24, gross_margin=0.6).fit(X_train, y_train)


def test_numpy_survival_matches_lifelines(split):
    from lifelines import CoxPHFitter

    X_train, X_test, y_train, _ = split
    model = SurvivalCLV().fit(X_train, y_train)
    Z = model.design(X_train).assign(duration=model._durations(X_train), event=y_train.to_numpy())
    cph = CoxPHFitter(penalizer=0.01).fit(Z, "duration", "event")
    times = [3.0, 12.0, 40.0]
    ours = model.survival_function(X_test.head(5), times)
    ref = cph.predict_survival_function(model.design(X_test.head(5)), times=times).to_numpy().T
    np.testing.assert_allclose(ours, ref, rtol=1e-6, atol=1e-8)


def test_survival_monotone_and_clv_bounds(surv, split):
    _, X_test, _, y_test = split
    s = surv.survival_function(X_test.head(50), np.arange(0, 80))
    assert np.all(np.diff(s, axis=1) <= 1e-12)
    rem = surv.expected_remaining_months(X_test)
    assert rem.min() >= 0 and rem.max() <= 24 + 1e-9
    assert (surv.clv(X_test) >= 0).all()
    assert surv.concordance(X_test, y_test) > 0.75


def test_month_to_month_hazard_higher(surv):
    assert surv.hazard_ratios_.loc["Contract[Month-to-month]", "hazard_ratio"] > 2


def test_psi_basics():
    assert population_stability_index([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)
    assert population_stability_index([0.9, 0.1], [0.1, 0.9]) > 0.25


def test_drift_monitor_flags_shift(split):
    X_train, X_test, _, _ = split
    mon = DriftMonitor().fit(X_train)
    stable = mon.compute(X_test)
    assert (stable["status"] == "ok").all()
    shifted = X_test.copy()
    shifted["MonthlyCharges"] *= 1.3
    shifted["Contract"] = "Month-to-month"
    res = mon.compute(shifted).set_index("feature")
    assert res.loc["MonthlyCharges", "status"] == "alert"
    assert res.loc["Contract", "status"] == "alert"


def test_concept_mapping():
    assert concept_of("Contract_Month-to-month") == "contract"
    assert concept_of("LogTenure") == "tenure"
    assert concept_of("InternetService_Fiber optic") == "internet"
    assert concept_of("TechSupport") == "protection"


def test_descriptions_never_contradict_values():
    row = pd.Series({"MonthlyCharges": 95.0, "tenure": 1, "TotalCharges": 95.0})
    assert describe("monthly_charges", row, -1.0).startswith("High")
    assert describe("tenure", row, -0.5) == "Short tenure (1 month)"


def test_linear_shap_is_additive(split):
    """For a linear model, base value + sum of SHAP values must equal the model's log-odds."""
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline("logreg", {"C": 0.1}, {"exclude": ["gender"]}).fit(X_train, y_train)
    ex = ChurnExplainer(pipe, X_train, n_background=300)
    values, base, _ = ex.shap_values(X_test.head(25))
    logit = pipe.decision_function(X_test.head(25))
    np.testing.assert_allclose(base + values.sum(axis=1), logit, atol=1e-8)


def test_tree_shap_is_additive(split):
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline("lightgbm", {"n_estimators": 60}, {"exclude": ["gender"]}).fit(X_train, y_train)
    ex = ChurnExplainer(pipe, X_train)
    values, base, _ = ex.shap_values(X_test.head(25))
    margin = pipe.predict_proba(X_test.head(25))[:, 1]
    np.testing.assert_allclose(1 / (1 + np.exp(-(base + values.sum(axis=1)))), margin, atol=1e-5)


def test_reasons_structure(split):
    X_train, X_test, y_train, _ = split
    pipe = build_pipeline("logreg", {"C": 0.1}, {"exclude": ["gender"]}).fit(X_train, y_train)
    reasons = ChurnExplainer(pipe, X_train).reasons(X_test.head(5), top_k=3)
    assert len(reasons) == 5
    for r in reasons:
        assert len(r["risk_drivers"]) <= 3
        assert all(d["impact"] > 0 for d in r["risk_drivers"])
        assert all(d["impact"] < 0 for d in r["protective_factors"])
