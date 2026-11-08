"""Request/response contracts. Pydantic enforces the same data contract as training
(allowed category levels, numeric ranges and the service-hierarchy consistency rules)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

YesNo = Literal["Yes", "No"]
InternetAddon = Literal["Yes", "No", "No internet service"]

EXAMPLE_CUSTOMER = {
    "customerID": "DEMO-0001",
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 3,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 98.5,
    "TotalCharges": 295.5,
}


class Customer(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": EXAMPLE_CUSTOMER})

    customerID: str | None = Field(default=None, max_length=64)
    gender: Literal["Female", "Male"] | None = None
    SeniorCitizen: Literal[0, 1]
    Partner: YesNo
    Dependents: YesNo
    tenure: int = Field(ge=0, le=120, description="Months with the company")
    PhoneService: YesNo
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: InternetAddon
    OnlineBackup: InternetAddon
    DeviceProtection: InternetAddon
    TechSupport: InternetAddon
    StreamingTV: InternetAddon
    StreamingMovies: InternetAddon
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: YesNo
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(ge=0, le=1000)
    TotalCharges: float | None = Field(
        default=None, ge=0, le=120_000, description="Blank for brand-new customers"
    )

    @model_validator(mode="after")
    def _service_hierarchy(self) -> Customer:
        addons = (
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        )
        if self.InternetService == "No":
            bad = [a for a in addons if getattr(self, a) != "No internet service"]
            if bad:
                raise ValueError(f"InternetService='No' requires 'No internet service' for {bad}")
        else:
            bad = [a for a in addons if getattr(self, a) == "No internet service"]
            if bad:
                raise ValueError(
                    f"{bad} cannot be 'No internet service' when InternetService='{self.InternetService}'"
                )
        if self.PhoneService == "No" and self.MultipleLines != "No phone service":
            raise ValueError("PhoneService='No' requires MultipleLines='No phone service'")
        if self.PhoneService == "Yes" and self.MultipleLines == "No phone service":
            raise ValueError("MultipleLines cannot be 'No phone service' when PhoneService='Yes'")
        return self


class Reason(BaseModel):
    factor: str
    description: str
    impact: float
    direction: Literal["increases_risk", "reduces_risk"]


class Prediction(BaseModel):
    customerID: str | None
    churn_probability: float = Field(ge=0, le=1)
    risk_tier: Literal["high", "medium", "low"]
    expected_remaining_months: float
    clv_estimate: float
    recommended_action: str
    action_description: str
    expected_value: float
    risk_drivers: list[Reason] = []
    protective_factors: list[Reason] = []


class BatchRequest(BaseModel):
    customers: list[Customer] = Field(min_length=1, max_length=10_000)


class BatchSummary(BaseModel):
    n_customers: int
    mean_churn_probability: float
    risk_tier_counts: dict[str, int]
    action_counts: dict[str, int]
    total_expected_value: float


class BatchResponse(BaseModel):
    model_version: str
    summary: BatchSummary
    predictions: list[Prediction]


class ModelInfo(BaseModel):
    model_name: str
    version: str
    trained_at: str
    calibration: str
    decision_threshold: float
    test_roc_auc: float
    test_pr_auc: float
    features_excluded: list[str]
    actions: list[dict]
