"""Model explanations and human-readable reason codes.

* Linear models  -> exact interventional SHAP values: phi_j = beta_j * (x_j - E[x_j]) (log-odds).
* Tree boosters  -> ``shap.TreeExplainer`` (exact TreeSHAP).
* Anything else  -> concept-level occlusion in raw-feature space (model-agnostic fallback).

One-hot and engineered columns are grouped back into business *concepts* (contract, tenure,
protection add-ons, ...) by summing their SHAP values (valid because SHAP values are additive),
so a call-centre agent sees "Month-to-month contract" rather than ``Contract_Month-to-month``.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_CONCEPT_RULES: dict[str, dict[str, tuple[str, ...]]] = {
    "contract": {"exact": ("MonthToMonth_x_InvTenure",), "prefix": ("Contract_",)},
    "tenure": {"exact": ("tenure", "LogTenure", "IsNewCustomer", "IsLongTenure"), "prefix": ()},
    "internet": {"exact": ("NewCustomer_x_Fiber",), "prefix": ("InternetService_",)},
    "payment": {"exact": ("IsAutoPay",), "prefix": ("PaymentMethod_",)},
    "monthly_charges": {"exact": ("MonthlyCharges", "HighSpend", "LowSpend"), "prefix": ()},
    "price_change": {"exact": ("ChargeDelta",), "prefix": ()},
    "total_charges": {"exact": ("TotalCharges",), "prefix": ()},
    "protection": {
        "exact": (
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "NumProtectionServices",
        ),
        "prefix": (),
    },
    "streaming": {"exact": ("StreamingTV", "StreamingMovies", "NumStreamingServices"), "prefix": ()},
    "phone": {"exact": ("PhoneService", "MultipleLines"), "prefix": ()},
    "household": {"exact": ("Partner", "Dependents"), "prefix": ()},
    "senior": {"exact": ("SeniorCitizen",), "prefix": ()},
    "billing": {"exact": ("PaperlessBilling",), "prefix": ()},
    "gender": {"exact": (), "prefix": ("gender_",)},
}

CONCEPT_LABELS = {
    "contract": "Contract type",
    "tenure": "Tenure",
    "internet": "Internet service",
    "payment": "Payment method",
    "monthly_charges": "Monthly charges",
    "price_change": "Bill vs. history",
    "total_charges": "Lifetime spend",
    "protection": "Protection add-ons",
    "streaming": "Streaming add-ons",
    "phone": "Phone lines",
    "household": "Partner / dependents",
    "senior": "Senior citizen",
    "billing": "Billing type",
    "gender": "Gender",
}

#: raw input columns behind each concept (used by the model-agnostic fallback)
CONCEPT_RAW_COLUMNS = {
    "contract": ["Contract"],
    "tenure": ["tenure"],
    "internet": ["InternetService"],
    "payment": ["PaymentMethod"],
    "monthly_charges": ["MonthlyCharges"],
    "total_charges": ["TotalCharges"],
    "protection": ["OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport"],
    "streaming": ["StreamingTV", "StreamingMovies"],
    "phone": ["PhoneService", "MultipleLines"],
    "household": ["Partner", "Dependents"],
    "senior": ["SeniorCitizen"],
    "billing": ["PaperlessBilling"],
}

_PROTECTION_NAMES = {
    "OnlineSecurity": "online security",
    "OnlineBackup": "online backup",
    "DeviceProtection": "device protection",
    "TechSupport": "tech support",
}


def concept_of(feature: str) -> str:
    for concept, rule in _CONCEPT_RULES.items():
        if feature in rule["exact"] or any(feature.startswith(p) for p in rule["prefix"]):
            return concept
    return feature


def _yes(v) -> bool:
    if isinstance(v, str):
        return v.strip().lower() in {"yes", "1", "true"}
    try:
        return bool(float(v) >= 0.5)
    except (TypeError, ValueError):
        return False


def _num(v, default=0.0) -> float:
    try:
        f = float(v)
        return default if np.isnan(f) else f
    except (TypeError, ValueError):
        return default


def _months(n: float) -> str:
    return f"{n:.0f} month" + ("" if round(n) == 1 else "s")


def describe(concept: str, row: pd.Series, impact: float) -> str:
    """Plain-language description of a concept's value for one customer.

    Wording is driven by the customer's actual value (not only by the SHAP sign), so a
    description can never contradict the data (e.g. a $95 bill is never called "low").
    """
    up = impact > 0
    tenure = _num(row.get("tenure"))
    monthly = _num(row.get("MonthlyCharges"))
    if concept == "contract":
        return f"{row.get('Contract')} contract"
    if concept == "tenure":
        if tenure <= 12:
            return f"Short tenure ({_months(tenure)})"
        if tenure >= 36:
            return f"Long tenure ({_months(tenure)})"
        return f"Tenure of {_months(tenure)}"
    if concept == "internet":
        net = row.get("InternetService")
        return "No internet service" if net == "No" else f"{net} internet"
    if concept == "payment":
        return f"Pays by {str(row.get('PaymentMethod')).lower()}"
    if concept == "monthly_charges":
        if monthly >= 80:
            return f"High monthly bill (${monthly:,.0f})"
        if monthly <= 35:
            return f"Low monthly bill (${monthly:,.0f})"
        return f"Monthly bill of ${monthly:,.0f}"
    if concept == "price_change":
        total = _num(row.get("TotalCharges"), monthly * tenure)
        avg = total / tenure if tenure > 0 else monthly
        delta = monthly - avg
        if abs(delta) < 1:
            return "Bill in line with historical average"
        side = "above" if delta > 0 else "below"
        return f"Current bill ${abs(delta):,.0f} {side} historical average"
    if concept == "total_charges":
        total = _num(row.get("TotalCharges"), monthly * tenure)
        return f"Lifetime spend of ${total:,.0f}"
    if concept == "protection":
        if row.get("InternetService") == "No":
            return "No internet add-ons (no internet service)"
        missing = [name for col, name in _PROTECTION_NAMES.items() if not _yes(row.get(col))]
        if not missing:
            return "Has all four protection add-ons"
        if len(missing) == 4:
            return "No protection add-ons (security, backup, device, tech support)"
        return "No " + ", ".join(missing) if up else f"Has {4 - len(missing)} of 4 protection add-ons"
    if concept == "streaming":
        n = sum(_yes(row.get(c)) for c in ("StreamingTV", "StreamingMovies"))
        return (
            "Streams TV and movies"
            if n == 2
            else ("Streams TV or movies" if n == 1 else "No streaming add-ons")
        )
    if concept == "phone":
        if not _yes(row.get("PhoneService")):
            return "No phone service"
        return "Multiple phone lines" if _yes(row.get("MultipleLines")) else "Single phone line"
    if concept == "household":
        partner, deps = _yes(row.get("Partner")), _yes(row.get("Dependents"))
        if not partner and not deps:
            return "No partner or dependents on the account"
        return "Has partner/dependents on the account"
    if concept == "senior":
        return "Senior citizen" if _yes(row.get("SeniorCitizen")) else "Not a senior citizen"
    if concept == "billing":
        return "Paperless billing" if _yes(row.get("PaperlessBilling")) else "Paper billing"
    if concept == "gender":
        return f"Gender: {row.get('gender')}"
    return CONCEPT_LABELS.get(concept, concept)


def _model_kind(model) -> str:
    if hasattr(model, "coef_"):
        return "linear"
    name = type(model).__name__
    if name in {
        "XGBClassifier",
        "LGBMClassifier",
        "CatBoostClassifier",
        "RandomForestClassifier",
        "ExtraTreesClassifier",
        "GradientBoostingClassifier",
    }:
        return "tree"
    return "agnostic"


class ChurnExplainer:
    """Explain the (uncalibrated) champion pipeline. Calibration is a monotone map applied on top,
    so the direction and ranking of contributions are unchanged."""

    def __init__(self, pipeline, background: pd.DataFrame, n_background: int = 200, random_state: int = 42):
        self.pipeline = pipeline
        bg = background.sample(min(n_background, len(background)), random_state=random_state)
        self.background_raw_ = bg
        is_standard = hasattr(pipeline, "named_steps") and "preprocess" in getattr(
            pipeline, "named_steps", {}
        )
        if is_standard:
            self.kind_ = _model_kind(pipeline[-1])
            # sklearn forests are explained in probability space; linear models and boosters in log-odds
            forest = type(pipeline[-1]).__name__ in {"RandomForestClassifier", "ExtraTreesClassifier"}
            self.units_ = "probability" if forest else "log-odds"
            self.feature_names_ = list(pipeline.named_steps["preprocess"].get_feature_names_out())
            self.background_t_ = np.asarray(pipeline[:-1].transform(bg), dtype=float)
        else:
            self.kind_ = "agnostic"
            self.units_ = "probability"
            self.feature_names_ = list(CONCEPT_RAW_COLUMNS)
            self.background_t_ = None
        self._tree = None

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_tree"] = None  # rebuilt lazily; shap explainers are not reliably picklable
        state["_concept_matrix"] = None
        return state

    # ------------------------------------------------------------------ attributions
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return np.asarray(self.pipeline[:-1].transform(X), dtype=float)

    def shap_values(
        self, X: pd.DataFrame, Xt: np.ndarray | None = None
    ) -> tuple[np.ndarray, float, np.ndarray | None]:
        """Return ``(values [n, features], base_value, transformed X)``.

        ``Xt`` may be passed when the caller already transformed ``X`` (saves a second pass at serving time).
        """
        if self.kind_ == "agnostic":
            return self._occlusion(X), float("nan"), None
        Xt = self.transform(X) if Xt is None else np.asarray(Xt, dtype=float)
        model = self.pipeline[-1]
        if self.kind_ == "linear":
            coef = np.asarray(model.coef_).ravel()
            mean = self.background_t_.mean(axis=0)
            values = (Xt - mean) * coef
            base = float(np.asarray(model.intercept_).ravel()[0] + coef @ mean)
            return values, base, Xt
        if self._tree is None:
            import shap

            self._tree = shap.TreeExplainer(model)
        raw = self._tree.shap_values(Xt)
        if isinstance(raw, list):
            raw = raw[1]
        raw = np.asarray(raw)
        if raw.ndim == 3:
            raw = raw[:, :, 1]
        base = np.asarray(self._tree.expected_value).ravel()
        return raw, float(base[-1]), Xt

    def _occlusion(self, X: pd.DataFrame) -> np.ndarray:
        """p(x) - E_bg[p(x with concept replaced by background)] for each raw concept."""
        bg = self.background_raw_.head(50)
        p_full = self.pipeline.predict_proba(X)[:, 1]
        out = np.zeros((len(X), len(CONCEPT_RAW_COLUMNS)))
        for i in range(len(X)):
            rep = pd.concat([X.iloc[[i]]] * len(bg), ignore_index=True)
            for j, cols in enumerate(CONCEPT_RAW_COLUMNS.values()):
                mod = rep.copy()
                mod[cols] = bg[cols].to_numpy()
                out[i, j] = p_full[i] - self.pipeline.predict_proba(mod)[:, 1].mean()
        return out

    def concept_values(self, X: pd.DataFrame, Xt: np.ndarray | None = None) -> pd.DataFrame:
        values, _, _ = self.shap_values(X, Xt)
        if self.kind_ == "agnostic":
            return pd.DataFrame(values, columns=self.feature_names_, index=X.index)
        if not hasattr(self, "_concept_matrix") or self._concept_matrix is None:
            concepts = [concept_of(f) for f in self.feature_names_]
            self._concept_names = list(dict.fromkeys(concepts))
            M = np.zeros((len(self.feature_names_), len(self._concept_names)))
            for i, c in enumerate(concepts):
                M[i, self._concept_names.index(c)] = 1.0
            self._concept_matrix = M
        return pd.DataFrame(values @ self._concept_matrix, columns=self._concept_names, index=X.index)

    def reasons(
        self, X: pd.DataFrame, top_k: int = 3, include_protective: bool = True, Xt: np.ndarray | None = None
    ) -> list[dict]:
        """Top risk-increasing (and optionally risk-reducing) factors per customer."""
        concepts = self.concept_values(X, Xt)
        out = []
        for i, (_, row) in enumerate(X.iterrows()):
            contrib = concepts.iloc[i].sort_values(ascending=False)
            drivers = [
                {
                    "factor": CONCEPT_LABELS.get(c, c),
                    "description": describe(c, row, v),
                    "impact": round(float(v), 4),
                    "direction": "increases_risk",
                }
                for c, v in contrib.items()
                if v > 0
            ][:top_k]
            protect = []
            if include_protective:
                protect = [
                    {
                        "factor": CONCEPT_LABELS.get(c, c),
                        "description": describe(c, row, v),
                        "impact": round(float(v), 4),
                        "direction": "reduces_risk",
                    }
                    for c, v in contrib.iloc[::-1].items()
                    if v < 0
                ][:top_k]
            out.append({"risk_drivers": drivers, "protective_factors": protect})
        return out
