"""Shared fixtures. A fast model bundle is trained once per session into a temp directory, so the
suite never depends on (or overwrites) the committed artifacts."""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

warnings.filterwarnings("ignore")


@pytest.fixture(scope="session")
def raw_df():
    from churnintel.data import clean, load_raw

    return clean(load_raw(DATA))


@pytest.fixture(scope="session")
def xy():
    from churnintel.data import RAW_FEATURES, load_dataset

    X, y = load_dataset(DATA)
    return X[RAW_FEATURES], y


@pytest.fixture(scope="session")
def split(xy):
    from sklearn.model_selection import train_test_split

    X, y = xy
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


@pytest.fixture(scope="session")
def run_dir(tmp_path_factory):
    from churnintel.logging_utils import setup_logging
    from churnintel.pipeline import run_training

    setup_logging("WARNING")
    out = tmp_path_factory.mktemp("fast_run")
    run_training(
        ROOT / "configs" / "config.fast.yaml",
        overrides={
            "paths": {
                "artifacts_dir": str(out / "artifacts"),
                "reports_dir": str(out / "reports"),
                "figures_dir": str(out / "reports" / "figures"),
            }
        },
    )
    return out


@pytest.fixture(scope="session")
def bundle_path(run_dir):
    return run_dir / "artifacts" / "model_bundle.joblib"


@pytest.fixture(scope="session")
def predictor(bundle_path):
    from churnintel.inference import ChurnPredictor

    return ChurnPredictor.load(bundle_path)


@pytest.fixture()
def customer() -> dict:
    from api.schemas import EXAMPLE_CUSTOMER

    return dict(EXAMPLE_CUSTOMER)
