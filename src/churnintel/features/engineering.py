"""Feature engineering as a fitted scikit-learn transformer.

Why a transformer (and not a function applied to the whole frame, as in v3)?

* ``HighSpend`` / ``LowSpend`` depend on quantiles of ``MonthlyCharges``. Computing those on the
  full dataset leaks test-set information into training, and computing them on a *single* request
  at serving time silently returns 0 for every customer (a train/serve skew bug in v3).
  Here the quantiles are learned in :meth:`fit` on training folds only and reused at inference.
* Living inside the ``Pipeline`` means every CV fold, the final model and the API share one code path.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

BASE_NUMERIC = ("SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges")
BINARY_COLUMNS = (
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
SERVICE_COLUMNS = (
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
PROTECTION_COLUMNS = ("OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport")
STREAMING_COLUMNS = ("StreamingTV", "StreamingMovies")
NOMINAL_COLUMNS = ("gender", "InternetService", "Contract", "PaymentMethod")
# v3 also had NumServices, AvgHistoricalCharge, Fiber_x_Charge and ServiceDensity. They are (near-)exact
# linear combinations of columns that belong to *other* concepts (e.g. MonthlyCharges is almost a linear
# function of the subscribed services). Removing them changed CV ROC-AUC by < 0.001 but stopped the logistic
# model from learning large offsetting weights, which made SHAP reason codes contradictory
# ("$95 bill lowers risk, fiber raises it by +2.4"). Within-concept combinations are kept.
ENGINEERED_NUMERIC = (
    "NumProtectionServices",
    "NumStreamingServices",
    "ChargeDelta",
    "LogTenure",
    "IsNewCustomer",
    "IsLongTenure",
    "IsAutoPay",
    "HighSpend",
    "LowSpend",
    "MonthToMonth_x_InvTenure",
    "NewCustomer_x_Fiber",
)


def _to_binary(series: pd.Series) -> pd.Series:
    """Map Yes / No / 'No internet service' / 'No phone service' (or 1/0) to floats; NaN stays NaN."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(float)

    def conv(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return np.nan
        return 1.0 if str(v).strip().lower() in {"yes", "1", "true"} else 0.0

    return series.map(conv).astype(float)


def _num(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce").astype(float)


def _row_sum(arrays: list[np.ndarray]) -> np.ndarray:
    """Row-wise sum that is NaN only when every input is NaN (like ``sum(min_count=1)``)."""
    stack = np.column_stack(arrays)
    all_nan = np.isnan(stack).all(axis=1)
    return np.where(all_nan, np.nan, np.nansum(stack, axis=1))


class TelcoFeatureEngineer(TransformerMixin, BaseEstimator):
    """Domain features for telecom churn.

    Parameters
    ----------
    engineered : bool
        If False, only base numeric/binary encodings are produced (used by ablation A1).
    exclude : tuple[str, ...]
        Raw or engineered columns to drop (e.g. ``("gender",)``).
    """

    def __init__(self, engineered: bool = True, exclude: tuple[str, ...] | list[str] = ()):
        self.engineered = engineered
        self.exclude = exclude

    # ------------------------------------------------------------------ bookkeeping
    def output_columns(self) -> tuple[list[str], list[str]]:
        """``(numeric_columns, nominal_columns)`` produced by :meth:`transform`."""
        excl = set(self.exclude or ())
        numeric = [c for c in (*BASE_NUMERIC, *BINARY_COLUMNS) if c not in excl]
        if self.engineered:
            numeric += [c for c in ENGINEERED_NUMERIC if c not in excl]
        nominal = [c for c in NOMINAL_COLUMNS if c not in excl]
        return numeric, nominal

    def required_inputs(self) -> list[str]:
        """Raw columns that must be present (engineered features read some excluded ones too)."""
        cols = [*BASE_NUMERIC, *BINARY_COLUMNS, "InternetService", "Contract", "PaymentMethod"]
        if "gender" not in set(self.exclude or ()):
            cols.append("gender")
        return cols

    # ------------------------------------------------------------------ sklearn API
    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TelcoFeatureEngineer expects a pandas DataFrame")
        monthly = _num(X, "MonthlyCharges")
        self.monthly_q25_ = float(np.nanquantile(monthly, 0.25))
        self.monthly_q75_ = float(np.nanquantile(monthly, 0.75))
        numeric, nominal = self.output_columns()
        self.numeric_features_ = numeric
        self.nominal_features_ = nominal
        self.feature_names_out_ = np.asarray(numeric + nominal, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "feature_names_out_")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("TelcoFeatureEngineer expects a pandas DataFrame")
        excl = set(self.exclude or ())
        missing = [c for c in self.required_inputs() if c not in X.columns]
        if missing:
            raise ValueError(f"Input is missing required columns: {missing}")

        tenure = _num(X, "tenure")
        monthly = _num(X, "MonthlyCharges")
        total = _num(X, "TotalCharges")
        # Blank TotalCharges <=> brand-new customer: impute with what they have been billed so far.
        total = total.fillna(monthly * tenure)

        # Columns are collected in a dict and materialised once (column-by-column inserts into a
        # DataFrame dominated single-request latency).
        cols: dict[str, np.ndarray] = {
            "SeniorCitizen": _num(X, "SeniorCitizen").to_numpy(),
            "tenure": tenure.to_numpy(),
            "MonthlyCharges": monthly.to_numpy(),
            "TotalCharges": total.to_numpy(),
        }
        for col in BINARY_COLUMNS:
            cols[col] = _to_binary(X[col]).to_numpy()

        if self.engineered:
            t, m, tot = cols["tenure"], cols["MonthlyCharges"], cols["TotalCharges"]
            is_mtm = (X["Contract"].to_numpy(dtype=object) == "Month-to-month").astype(float)
            is_fiber = (X["InternetService"].to_numpy(dtype=object) == "Fiber optic").astype(float)
            payment = X["PaymentMethod"].astype(object).astype(str)
            new_cust = (t <= 6).astype(float)
            with np.errstate(invalid="ignore", divide="ignore"):
                avg_hist = np.where(t > 0, tot / np.clip(t, 1, None), m)

            cols["NumProtectionServices"] = _row_sum([cols[c] for c in PROTECTION_COLUMNS])
            cols["NumStreamingServices"] = _row_sum([cols[c] for c in STREAMING_COLUMNS])
            cols["ChargeDelta"] = m - avg_hist  # > 0: current bill above historical average
            cols["LogTenure"] = np.log1p(t)
            cols["IsNewCustomer"] = new_cust
            cols["IsLongTenure"] = (t >= 48).astype(float)
            cols["IsAutoPay"] = payment.str.contains("automatic", case=False, regex=False).to_numpy(
                dtype=float
            )
            cols["HighSpend"] = (m > self.monthly_q75_).astype(float)
            cols["LowSpend"] = (m < self.monthly_q25_).astype(float)
            cols["MonthToMonth_x_InvTenure"] = is_mtm / (t + 1.0)
            cols["NewCustomer_x_Fiber"] = new_cust * is_fiber

        for col in NOMINAL_COLUMNS:
            if col not in excl:
                cols[col] = X[col].to_numpy(dtype=object)

        names = list(self.feature_names_out_)
        return pd.DataFrame({c: cols[c] for c in names}, index=X.index, columns=names)

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "feature_names_out_")
        return self.feature_names_out_.copy()


def build_preprocessor(numeric: list[str], nominal: list[str]) -> ColumnTransformer:
    """Median-impute + scale numerics; mode-impute + one-hot nominals (unknown levels ignored)."""
    numeric_pipe = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    nominal_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipe, numeric), ("cat", nominal_pipe, nominal)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def heuristic_risk_score(X: pd.DataFrame) -> np.ndarray:
    """The hand-written ``ManualRiskScore`` from v3, kept as a rule-based baseline (never trained on)."""
    tenure = _num(X, "tenure")
    tech = _to_binary(X["TechSupport"]).fillna(0)
    return (
        0.4 * (X["Contract"] == "Month-to-month").astype(float)
        + 0.2 * (tenure <= 6).astype(float)
        + 0.2 * (X["InternetService"] == "Fiber optic").astype(float)
        + 0.1 * (X["PaymentMethod"] == "Electronic check").astype(float)
        + 0.1 * (1 - tech)
    ).to_numpy(dtype=float)
