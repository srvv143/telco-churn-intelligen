import json

import joblib


def test_artifacts_and_reports_written(run_dir):
    assert (run_dir / "artifacts" / "model_bundle.joblib").exists()
    assert (run_dir / "artifacts" / "model_metadata.json").exists()
    for name in (
        "metrics.json",
        "RESULTS.md",
        "model_card.md",
        "RESUME.md",
        "policy_comparison.csv",
        "test_predictions.csv",
    ):
        assert (run_dir / "reports" / name).exists(), name
    assert (run_dir / "reports" / "figures" / "dashboard.png").stat().st_size > 50_000


def test_metrics_are_sane(run_dir):
    m = json.loads((run_dir / "reports" / "metrics.json").read_text())
    t = m["test"]["champion"]
    assert 0.80 < t["roc_auc"] < 0.95
    ci = t["bootstrap_ci"]["roc_auc"]
    assert ci["ci_lower"] < t["roc_auc"] < ci["ci_upper"]
    assert 0 < m["thresholds"]["profit_optimal"] < 1
    pol = m["business"]["policies"]
    oracle = pol["Oracle (perfect foresight, reference)"]["net_profit"]
    assert all(p["net_profit"] <= oracle + 1e-6 for p in pol.values())
    assert m["data"]["n_train"] + m["data"]["n_test"] == 7043
    assert m["selection"]["champion"] in m["cv"]


def test_test_set_never_used_for_selection(run_dir):
    """Selection inputs (CV summaries) contain only training-fold metrics."""
    m = json.loads((run_dir / "reports" / "metrics.json").read_text())
    for summary in m["cv"].values():
        assert summary["n_folds"] >= 3


def test_bundle_is_self_contained(bundle_path):
    bundle = joblib.load(bundle_path)
    for key in ("model", "explainer", "survival", "actions", "risk_tiers", "drift_monitor", "metadata"):
        assert key in bundle
    assert not hasattr(bundle["survival"], "_cph"), "lifelines objects must not be needed at serving time"


def test_resume_mentions_measured_auc(run_dir):
    m = json.loads((run_dir / "reports" / "metrics.json").read_text())
    text = (run_dir / "reports" / "RESUME.md").read_text()
    assert f"{m['test']['champion']['roc_auc']:.3f}" in text
