import numpy as np
from sklearn.metrics import roc_auc_score

from churnintel.evaluation import (
    bootstrap_ci,
    classification_metrics,
    delong_roc_test,
    delong_roc_variance,
    expected_calibration_error,
    fairness_report,
    gains_table,
    ks_statistic,
    lift_at,
)
from churnintel.evaluation.metrics import threshold_metrics

rng = np.random.default_rng(0)
Y = rng.integers(0, 2, 2000)
GOOD = np.clip(Y * 0.35 + rng.normal(0.3, 0.2, 2000), 0, 1)
NOISE = rng.random(2000)


def test_ece_zero_for_constant_base_rate():
    y = np.array([0, 1, 0, 1])
    assert expected_calibration_error(y, np.full(4, 0.5)) == 0.0


def test_ece_large_for_overconfident():
    assert expected_calibration_error(Y, (1 - Y).astype(float)) > 0.9


def test_lift_and_capture_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    res = lift_at(y, y.astype(float), 0.2)
    assert res["precision"] == 1.0 and res["capture_rate"] == 1.0 and res["lift"] == 5.0


def test_gains_table_is_cumulative():
    g = gains_table(Y, GOOD)
    assert len(g) == 10
    assert np.all(np.diff(g["cum_capture"]) >= 0)
    assert abs(g["cum_capture"].iloc[-1] - 1.0) < 1e-12


def test_ks_perfect_separation():
    assert ks_statistic(Y, Y.astype(float)) == 1.0


def test_threshold_metrics_counts_add_up():
    m = threshold_metrics(Y, GOOD, 0.5)
    assert m["tp"] + m["fp"] + m["tn"] + m["fn"] == len(Y)


def test_classification_metrics_keys():
    m = classification_metrics(Y, GOOD)
    for k in ("roc_auc", "pr_auc", "brier", "log_loss", "ece", "ks", "lift", "at_threshold"):
        assert k in m


def test_bootstrap_ci_brackets_estimate():
    ci = bootstrap_ci(Y, GOOD, {"auc": roc_auc_score}, n_boot=200)
    row = ci.loc["auc"]
    assert row["ci_lower"] <= row["estimate"] <= row["ci_upper"]


def test_delong_auc_matches_sklearn_with_ties():
    scores = np.round(GOOD, 1)  # heavy ties
    auc, var = delong_roc_variance(Y, scores)
    assert abs(auc - roc_auc_score(Y, scores)) < 1e-12
    assert var > 0


def test_delong_identical_models_not_significant():
    res = delong_roc_test(Y, GOOD, GOOD)
    assert res["auc_diff"] == 0 and res["p_value"] == 1.0


def test_delong_detects_real_difference():
    res = delong_roc_test(Y, GOOD, NOISE)
    assert res["auc_diff"] > 0.2 and res["p_value"] < 1e-6


def test_fairness_report_shapes():
    groups = {"g": np.where(np.arange(len(Y)) % 2 == 0, "a", "b")}
    detail, summary = fairness_report(Y, GOOD, (GOOD > 0.5).astype(int), groups)
    assert set(detail["group"]) == {"a", "b"}
    assert 0 <= summary.loc[0, "selection_rate_ratio"] <= 1
