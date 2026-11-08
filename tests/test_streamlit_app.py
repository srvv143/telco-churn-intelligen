from pathlib import Path

import pytest

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_dashboard_renders_without_exceptions(run_dir, monkeypatch):
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("CHURN_MODEL_PATH", str(run_dir / "artifacts" / "model_bundle.joblib"))
    monkeypatch.setenv("CHURN_REPORTS_DIR", str(run_dir / "reports"))
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, at.exception
    assert any("Churn probability" in m.label for m in at.metric)

    # interact: switch to a two-year contract and confirm the app re-renders cleanly
    contract = next(s for s in at.selectbox if s.label == "Contract")
    contract.set_value("Two year").run()
    assert not at.exception
