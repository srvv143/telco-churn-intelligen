import numpy as np

from churnintel.data import validate_dataframe


def test_dataset_shape_and_target(xy):
    X, y = xy
    assert X.shape == (7043, 19)
    assert X.index.is_unique and X.index.name == "customerID"
    assert abs(y.mean() - 0.2654) < 1e-3
    assert set(np.unique(y)) == {0, 1}


def test_blank_total_charges_are_new_customers(raw_df):
    blank = raw_df["TotalCharges"].isna()
    assert blank.sum() == 11
    assert (raw_df.loc[blank, "tenure"] == 0).all()


def test_validation_passes_on_real_data(raw_df):
    report = validate_dataframe(raw_df)
    assert report.ok, report.errors


def test_validation_catches_unknown_level(raw_df):
    bad = raw_df.head(20).copy()
    bad.loc[bad.index[0], "Contract"] = "Weekly"
    report = validate_dataframe(bad)
    assert not report.ok
    assert any("Contract" in e for e in report.errors)


def test_validation_catches_out_of_range_and_missing_column(raw_df):
    bad = raw_df.head(20).copy()
    bad.loc[bad.index[0], "tenure"] = -5
    assert any("tenure" in e for e in validate_dataframe(bad).errors)
    assert any("missing required" in e for e in validate_dataframe(bad.drop(columns=["Contract"])).errors)


def test_validation_flags_service_inconsistency(raw_df):
    bad = raw_df.head(50).copy()
    i = bad.index[bad["InternetService"] == "No"][0]
    bad.loc[i, "OnlineSecurity"] = "Yes"
    report = validate_dataframe(bad)
    assert any("OnlineSecurity" in w for w in report.warnings)


def test_duplicate_ids_are_errors(raw_df):
    dup = raw_df.head(5).copy()
    dup.loc[dup.index[1], "customerID"] = dup.loc[dup.index[0], "customerID"]
    assert not validate_dataframe(dup).ok
