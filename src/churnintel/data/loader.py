"""Loading and deterministic cleaning. Nothing here learns statistics from the data,
so it is safe to run before the train/test split."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import ID_COLUMN, NUMERIC_RANGES, TARGET

logger = logging.getLogger(__name__)


def load_raw(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run `python scripts/download_data.py` (or `make data`)."
        )
    return pd.read_csv(path)


def _strip(value):
    return value.strip() if isinstance(value, str) else value


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace, turn blanks into NaN and coerce numeric columns.

    The 11 blank ``TotalCharges`` values belong to customers with ``tenure == 0``; they are
    left as NaN here and imputed inside the model pipeline (``TotalCharges = Monthly x tenure``)
    so that training and serving share exactly one code path.
    """
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].map(_strip).astype(object).replace({"": np.nan})
    for col in NUMERIC_RANGES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``X`` (indexed by customerID) and a binary ``y`` (1 = churned)."""
    data = df.copy()
    if ID_COLUMN in data.columns:
        data = data.set_index(ID_COLUMN)
    target = data.pop(TARGET)
    if pd.api.types.is_numeric_dtype(target):
        y = target.astype(int)
    else:
        y = target.astype(str).str.strip().eq("Yes").astype(int)
    y.name = TARGET
    return data, y


def load_dataset(path: str | Path, sample_frac: float | None = None, random_state: int = 42):
    """Load, clean and split into ``(X, y)``. Optionally stratified-subsample for quick runs."""
    df = clean(load_raw(path))
    if sample_frac and 0 < sample_frac < 1:
        df = df.groupby(TARGET, group_keys=False).sample(frac=sample_frac, random_state=random_state)
        logger.info("Sub-sampled to %d rows (frac=%.2f)", len(df), sample_frac)
    return split_features_target(df)
