import numpy as np
import pandas as pd
import pytest

from churnintel.features import TelcoFeatureEngineer, heuristic_risk_score


@pytest.fixture(scope="module")
def fitted(xy):
    X, _ = xy
    return TelcoFeatureEngineer(exclude=["gender"]).fit(X)


def test_output_columns_match_transform(fitted, xy):
    X, _ = xy
    out = fitted.transform(X.head(50))
    numeric, nominal = fitted.output_columns()
    assert list(out.columns) == numeric + nominal
    assert "gender" not in out.columns


def test_no_missing_values_after_transform(fitted, xy):
    X, _ = xy
    out = fitted.transform(X)
    assert out[fitted.numeric_features_].isna().sum().sum() == 0  # blank TotalCharges imputed


def test_train_serve_parity_single_row_equals_batch(fitted, xy):
    X, _ = xy
    rows = X.sample(200, random_state=0)
    batch = fitted.transform(rows)
    for i in range(len(rows)):
        single = fitted.transform(rows.iloc[[i]])
        pd.testing.assert_frame_equal(single, batch.iloc[[i]])


def test_quantiles_are_learned_on_fit_data_only(xy):
    X, _ = xy
    cheap = X[X["MonthlyCharges"] < 40]
    fe = TelcoFeatureEngineer().fit(cheap)
    assert fe.monthly_q75_ < 40
    # a $100 customer is "HighSpend" relative to the cheap training population
    assert fe.transform(X[X["MonthlyCharges"] > 100].head(1))["HighSpend"].item() == 1.0


def test_v3_single_row_high_spend_bug_is_fixed(fitted, xy):
    """v3 computed quantiles on the incoming frame: for one row, HighSpend was always 0."""
    X, _ = xy
    top = X.loc[[X["MonthlyCharges"].idxmax()]]
    assert fitted.transform(top)["HighSpend"].item() == 1.0


def test_numeric_binary_encoding_equivalent(fitted, xy):
    X, _ = xy
    rows = X.head(30)
    encoded = rows.copy()
    for col in ("Partner", "Dependents", "PhoneService", "PaperlessBilling"):
        encoded[col] = (rows[col] == "Yes").astype(int)
    pd.testing.assert_frame_equal(fitted.transform(rows), fitted.transform(encoded))


def test_missing_required_column_raises(fitted, xy):
    X, _ = xy
    with pytest.raises(ValueError, match="Contract"):
        fitted.transform(X.drop(columns=["Contract"]).head(3))


def test_engineered_flag_controls_columns(xy):
    X, _ = xy
    raw_only = TelcoFeatureEngineer(engineered=False).fit(X)
    assert "LogTenure" not in raw_only.transform(X.head(2)).columns
    assert "gender" in raw_only.transform(X.head(2)).columns


def test_heuristic_score_range(xy):
    X, _ = xy
    s = heuristic_risk_score(X)
    assert s.min() >= 0 and s.max() <= 1.0 + 1e-9
    assert np.isfinite(s).all()
