"""Data contract for the IBM Telco customer churn dataset (7,043 customers x 21 columns)."""

from __future__ import annotations

ID_COLUMN = "customerID"
TARGET = "Churn"

_YES_NO = ("Yes", "No")
_INTERNET_ADDON = ("Yes", "No", "No internet service")

#: Allowed levels for every categorical input column.
CATEGORY_LEVELS: dict[str, tuple[str, ...]] = {
    "gender": ("Female", "Male"),
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "MultipleLines": ("Yes", "No", "No phone service"),
    "InternetService": ("DSL", "Fiber optic", "No"),
    "OnlineSecurity": _INTERNET_ADDON,
    "OnlineBackup": _INTERNET_ADDON,
    "DeviceProtection": _INTERNET_ADDON,
    "TechSupport": _INTERNET_ADDON,
    "StreamingTV": _INTERNET_ADDON,
    "StreamingMovies": _INTERNET_ADDON,
    "Contract": ("Month-to-month", "One year", "Two year"),
    "PaperlessBilling": _YES_NO,
    "PaymentMethod": (
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ),
}

#: Inclusive plausible ranges for numeric inputs.
NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "SeniorCitizen": (0, 1),
    "tenure": (0, 120),
    "MonthlyCharges": (0.0, 1_000.0),
    "TotalCharges": (0.0, 120_000.0),
}

#: Columns the model consumes (before feature engineering), in canonical order.
RAW_FEATURES: list[str] = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

INTERNET_ADDONS = (
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
)
