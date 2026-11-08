import numpy as np
import pandas as pd

from churnintel.inference import NO_ACTION


def test_score_columns_and_ranges(predictor, split):
    _, X_test, _, _ = split
    out = predictor.score(X_test.head(100))
    expected = {
        "churn_probability",
        "risk_tier",
        "expected_remaining_months",
        "clv_estimate",
        "recommended_action",
        "action_description",
        "expected_value",
    }
    assert expected <= set(out.columns)
    assert out["churn_probability"].between(0, 1).all()
    assert set(out["risk_tier"]) <= {"low", "medium", "high"}
    assert set(out["recommended_action"]) <= {a.name for a in predictor.actions} | {NO_ACTION}
    assert (out.loc[out["recommended_action"] == NO_ACTION, "expected_value"] == 0).all()
    assert list(out.index) == list(X_test.head(100).index)


def test_single_vs_batch_consistency(predictor, split):
    _, X_test, _, _ = split
    rows = X_test.head(20)
    batch = predictor.predict_proba(rows)
    single = np.array([predictor.predict_proba(rows.iloc[[i]])[0] for i in range(len(rows))])
    np.testing.assert_allclose(batch, single, rtol=1e-10)


def test_dict_input_with_blank_total_and_no_gender(predictor, customer):
    customer.pop("gender")
    customer["TotalCharges"] = None
    customer["tenure"] = 0
    out = predictor.score(customer, explain=True)
    assert len(out) == 1 and 0 <= out["churn_probability"].iloc[0] <= 1
    assert "risk_drivers" in out["explanation"].iloc[0]


def test_contract_effect_direction(predictor, customer):
    risky = predictor.predict_proba(customer)[0]
    safer = predictor.predict_proba(
        {**customer, "Contract": "Two year", "tenure": 40, "TotalCharges": 3900.0}
    )[0]
    assert risky > safer


def test_extra_columns_are_ignored(predictor, customer):
    a = predictor.predict_proba(customer)[0]
    b = predictor.predict_proba({**customer, "favourite_colour": "teal"})[0]
    assert a == b


def test_drift_report(predictor, split):
    _, X_test, _, _ = split
    report = predictor.drift_report(X_test)
    assert {"feature", "psi", "status"} <= set(report.columns)
    assert "__churn_score__" in set(report["feature"])
    assert isinstance(report, pd.DataFrame)
