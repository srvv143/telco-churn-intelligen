"""Time-to-churn modelling and survival-based customer lifetime value.

A Cox proportional-hazards model is fitted on (duration = tenure, event = churn). For a customer
who has survived ``tau`` months, the expected number of further billed months within a horizon H is

    E[remaining | tau, x] = sum_{k=1..H} S(tau + k | x) / S(tau | x),   S(t | x) = S0(t) ** exp(x.beta)

and ``CLV = gross_margin * MonthlyCharges * E[remaining]``.

Only plain numpy arrays are kept after fitting, so serving does not need ``lifelines``.
Tenure and TotalCharges (a function of tenure) are deliberately *not* covariates: they are the
duration itself.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BINARY_COVARIATES = (
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
)
#: nominal column -> (reference level, other levels)
NOMINAL_COVARIATES = {
    "Contract": ("Two year", ("Month-to-month", "One year")),
    "InternetService": ("DSL", ("Fiber optic", "No")),
    "PaymentMethod": (
        "Credit card (automatic)",
        ("Electronic check", "Mailed check", "Bank transfer (automatic)"),
    ),
}


def _binary(s: pd.Series) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(s):
        return s.fillna(0).to_numpy(dtype=float)
    return s.astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).to_numpy(dtype=float)


class SurvivalCLV:
    def __init__(
        self,
        horizon_months: int = 24,
        gross_margin: float = 0.6,
        penalizer: float = 0.01,
        method: str = "survival",
    ):
        self.horizon_months = int(horizon_months)
        self.gross_margin = float(gross_margin)
        self.penalizer = float(penalizer)
        self.method = method

    # ------------------------------------------------------------------ design matrix
    @staticmethod
    def design(X: pd.DataFrame) -> pd.DataFrame:
        cols: dict[str, np.ndarray] = {}
        for c in BINARY_COVARIATES:
            cols[c] = _binary(X[c])
        for c, (_ref, levels) in NOMINAL_COVARIATES.items():
            values = X[c].astype(str).str.strip()
            for lvl in levels:
                cols[f"{c}[{lvl}]"] = (values == lvl).to_numpy(dtype=float)
        monthly = pd.to_numeric(X["MonthlyCharges"], errors="coerce")
        cols["MonthlyCharges_per_10USD"] = (monthly.fillna(monthly.median()) / 10.0).to_numpy(dtype=float)
        return pd.DataFrame(cols, index=X.index)

    @staticmethod
    def _durations(X: pd.DataFrame) -> np.ndarray:
        t = pd.to_numeric(X["tenure"], errors="coerce").fillna(0).to_numpy(dtype=float)
        return np.clip(t, 0.5, None)

    # ------------------------------------------------------------------ fit / predict
    def fit(self, X: pd.DataFrame, y) -> SurvivalCLV:
        from lifelines import CoxPHFitter

        Z = self.design(X)
        frame = Z.assign(duration=self._durations(X), event=np.asarray(y).astype(int))
        cph = CoxPHFitter(penalizer=self.penalizer)
        cph.fit(frame, duration_col="duration", event_col="event")

        self.columns_ = list(Z.columns)
        self.coef_ = cph.params_.reindex(self.columns_).to_numpy(dtype=float)
        zero = pd.DataFrame([np.zeros(len(self.columns_))], columns=self.columns_)
        self.offset_ = float(np.asarray(cph.predict_log_partial_hazard(zero)).ravel()[0])
        base = cph.baseline_survival_
        self.baseline_times_ = base.index.to_numpy(dtype=float)
        self.baseline_survival_ = base.iloc[:, 0].to_numpy(dtype=float)
        summ = cph.summary
        self.hazard_ratios_ = pd.DataFrame(
            {
                "hazard_ratio": summ["exp(coef)"],
                "ci_lower": summ["exp(coef) lower 95%"],
                "ci_upper": summ["exp(coef) upper 95%"],
                "p_value": summ["p"],
            }
        ).reindex(self.columns_)
        self.train_concordance_ = float(cph.concordance_index_)
        logger.info("Cox PH fitted: train C-index=%.4f", self.train_concordance_)
        return self

    def log_partial_hazard(self, X: pd.DataFrame) -> np.ndarray:
        return self.design(X)[self.columns_].to_numpy(dtype=float) @ self.coef_ + self.offset_

    def _s0(self, t: np.ndarray) -> np.ndarray:
        idx = np.searchsorted(self.baseline_times_, t, side="right") - 1
        out = np.ones_like(t, dtype=float)
        valid = idx >= 0
        out[valid] = self.baseline_survival_[idx[valid]]
        return out

    def survival_function(self, X: pd.DataFrame, times) -> np.ndarray:
        """S(t | x) for every customer (rows) and time (columns)."""
        times = np.asarray(times, dtype=float)
        risk = np.exp(self.log_partial_hazard(X))[:, None]
        return self._s0(np.broadcast_to(times, (len(X), len(times))).copy()) ** risk

    def expected_remaining_months(self, X: pd.DataFrame) -> np.ndarray:
        if self.method == "horizon":
            return np.full(len(X), float(self.horizon_months))
        tau = self._durations(X)
        risk = np.exp(self.log_partial_hazard(X))[:, None]
        ahead = tau[:, None] + np.arange(1, self.horizon_months + 1)[None, :]
        s_tau = np.clip(self._s0(tau[:, None]) ** risk, 1e-12, None)
        s_ahead = self._s0(ahead) ** risk
        return (s_ahead / s_tau).sum(axis=1)

    def clv(self, X: pd.DataFrame, remaining: np.ndarray | None = None) -> np.ndarray:
        """Margin x monthly charge x expected remaining months (pass ``remaining`` to avoid recomputing it)."""
        monthly = pd.to_numeric(X["MonthlyCharges"], errors="coerce").fillna(0).to_numpy(dtype=float)
        if remaining is None:
            remaining = self.expected_remaining_months(X)
        return self.gross_margin * monthly * remaining

    def concordance(self, X: pd.DataFrame, y) -> float:
        from lifelines.utils import concordance_index

        return float(concordance_index(self._durations(X), -self.log_partial_hazard(X), np.asarray(y)))


def kaplan_meier_by_group(X: pd.DataFrame, y, column: str) -> dict:
    """Kaplan-Meier curves, median lifetimes and a multivariate log-rank test for one grouping."""
    from lifelines import KaplanMeierFitter
    from lifelines.statistics import multivariate_logrank_test

    durations = SurvivalCLV._durations(X)
    events = np.asarray(y).astype(int)
    groups = X[column].astype(str).to_numpy()
    curves, medians = {}, {}
    for g in sorted(pd.unique(groups)):
        m = groups == g
        km = KaplanMeierFitter().fit(durations[m], events[m], label=g)
        sf = km.survival_function_
        curves[g] = (sf.index.to_numpy(dtype=float), sf.iloc[:, 0].to_numpy(dtype=float))
        median = km.median_survival_time_
        medians[g] = float(median) if np.isfinite(median) else None
    test = multivariate_logrank_test(durations, groups, events)
    return {
        "column": column,
        "curves": curves,
        "median_months": medians,
        "logrank_chi2": float(test.test_statistic),
        "logrank_p": float(test.p_value),
    }
