PY ?= python
export PYTHONPATH := src

.PHONY: help install data train train-fast ablate test lint format api dashboard batch drift benchmark reports notebook docker-build docker-up all clean

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## install runtime + dev dependencies and the package (editable)
	$(PY) -m pip install -r requirements.txt -r requirements-dev.txt
	$(PY) -m pip install -e .

data:  ## download the IBM Telco dataset (already bundled in data/raw)
	$(PY) scripts/download_data.py

train:  ## full pipeline: tuning, CV, calibration, business eval, reports (~20-30 min on 2 CPUs)
	$(PY) scripts/train.py --config configs/config.yaml

train-fast:  ## ~1 minute smoke run (writes to artifacts/ and reports/)
	$(PY) scripts/train.py --config configs/config.fast.yaml

ablate:  ## ablation studies: feature engineering, imbalance handling, gender input
	$(PY) scripts/run_ablations.py
	$(PY) scripts/make_reports.py

test:  ## run the test suite with coverage (writes reports/test_summary.json)
	$(PY) scripts/run_tests.py
	$(PY) scripts/make_reports.py

lint:  ## static checks
	ruff check .

format:  ## auto-format
	ruff format . && ruff check --fix .

api:  ## serve the REST API on :8000 (docs at /docs)
	uvicorn api.main:app --reload --port 8000

dashboard:  ## launch the Streamlit dashboard on :8501
	streamlit run app/streamlit_app.py

batch:  ## score the sample CSV with reason codes
	$(PY) scripts/batch_score.py --input data/sample/sample_customers.csv --output reports/batch_predictions.csv --explain

drift:  ## PSI drift report on a simulated production shift
	-$(PY) scripts/monitor_drift.py --simulate

benchmark:  ## measure API latency over HTTP
	$(PY) scripts/benchmark_api.py --requests 500
	$(PY) scripts/make_reports.py

reports:  ## re-render RESULTS.md / model_card.md / RESUME.md from JSON
	$(PY) scripts/make_reports.py

notebook:  ## rebuild and execute the EDA notebook
	$(PY) scripts/build_notebook.py --execute

docker-build:  ## build the serving image
	docker build -t churn-intel:latest .

docker-up:  ## run API (:8000) and dashboard (:8501)
	docker compose up

all: train ablate benchmark test  ## everything, in order

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov logs
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
