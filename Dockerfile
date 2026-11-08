# ---------------------------------------------------------------------------
# Telco Churn Intelligence — serving image (API by default, dashboard via CMD override)
#   docker build -t churn-intel .
#   docker run -p 8000:8000 churn-intel
# ---------------------------------------------------------------------------
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    CHURN_MODEL_PATH=/app/artifacts/model_bundle.joblib

# libgomp is required by LightGBM / XGBoost
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# exact versions that produced the bundled model artifact
COPY requirements-lock.txt .
RUN pip install -r requirements-lock.txt

COPY src ./src
COPY api ./api
COPY app ./app
COPY configs ./configs
COPY data ./data
COPY artifacts ./artifacts
COPY reports ./reports

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status == 200 else 1)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
