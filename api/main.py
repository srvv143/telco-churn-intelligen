"""FastAPI service: churn probability, risk tier, survival-based CLV, next-best-action and
SHAP reason codes for one customer or a batch.

    uvicorn api.main:app --port 8000      # docs at http://localhost:8000/docs
"""

from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Query, Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402

from churnintel import __version__  # noqa: E402
from churnintel.inference import ChurnPredictor  # noqa: E402

from .schemas import BatchRequest, BatchResponse, BatchSummary, Customer, ModelInfo, Prediction  # noqa: E402

logger = logging.getLogger("churn-api")
MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", ROOT / "artifacts" / "model_bundle.joblib"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictor = ChurnPredictor.load(MODEL_PATH)
    app.state.n_requests = 0
    yield


app = FastAPI(
    title="Telco Churn Intelligence API",
    version=__version__,
    description="Calibrated churn risk, customer lifetime value and profit-optimal retention actions with reason codes.",
    lifespan=lifespan,
)


@app.middleware("http")
async def timing_header(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    response.headers["X-Process-Time-Ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
    return response


def _predictor(request: Request) -> ChurnPredictor:
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    request.app.state.n_requests += 1
    return predictor


def _to_predictions(scored, explain: bool) -> list[Prediction]:
    out = []
    for cid, row in scored.iterrows():
        exp = row["explanation"] if explain and "explanation" in scored.columns else {}
        out.append(
            Prediction(
                customerID=None if str(cid).startswith("row_") else str(cid),
                churn_probability=float(row["churn_probability"]),
                risk_tier=row["risk_tier"],
                expected_remaining_months=float(row["expected_remaining_months"]),
                clv_estimate=float(row["clv_estimate"]),
                recommended_action=row["recommended_action"],
                action_description=row["action_description"],
                expected_value=float(row["expected_value"]),
                risk_drivers=exp.get("risk_drivers", []) if exp else [],
                protective_factors=exp.get("protective_factors", []) if exp else [],
            )
        )
    return out


@app.get("/health")
def health(request: Request) -> dict:
    loaded = getattr(request.app.state, "predictor", None) is not None
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded, "api_version": __version__}


@app.get("/v1/model", response_model=ModelInfo)
def model_info(request: Request) -> ModelInfo:
    p = _predictor(request)
    md = p.metadata
    return ModelInfo(
        model_name=md.get("model_name", "unknown"),
        version=str(md.get("version", "")),
        trained_at=str(md.get("trained_at", "")),
        calibration=str(md.get("calibration", "")),
        decision_threshold=float(p.bundle.get("decision_threshold", 0.5)),
        test_roc_auc=float(md.get("test_roc_auc", float("nan"))),
        test_pr_auc=float(md.get("test_pr_auc", float("nan"))),
        features_excluded=list(md.get("features_excluded", [])),
        actions=[a.to_dict() for a in p.actions],
    )


@app.post("/v1/predict", response_model=Prediction)
def predict(
    customer: Customer, request: Request, explain: bool = Query(True, description="include reason codes")
):
    p = _predictor(request)
    scored = p.score([customer.model_dump()], explain=explain)
    return _to_predictions(scored, explain)[0]


@app.post("/v1/predict/batch", response_model=BatchResponse)
def predict_batch(body: BatchRequest, request: Request, explain: bool = Query(False)):
    p = _predictor(request)
    scored = p.score([c.model_dump() for c in body.customers], explain=explain)
    summary = BatchSummary(
        n_customers=len(scored),
        mean_churn_probability=float(scored["churn_probability"].mean()),
        risk_tier_counts={k: int(v) for k, v in scored["risk_tier"].value_counts().items()},
        action_counts={k: int(v) for k, v in scored["recommended_action"].value_counts().items()},
        total_expected_value=float(scored["expected_value"].sum()),
    )
    return BatchResponse(
        model_version=str(p.metadata.get("version", "")),
        summary=summary,
        predictions=_to_predictions(scored, explain),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(status_code=422, content={"detail": str(exc)})
