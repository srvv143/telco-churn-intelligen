#!/usr/bin/env python
"""Build (and optionally execute) notebooks/01_eda_and_results.ipynb from code, so the notebook is
reproducible and diff-friendly.

    python scripts/build_notebook.py --execute
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "01_eda_and_results.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# Telco churn — EDA and results walkthrough

This notebook explores the IBM Telco dataset, motivates the modelling choices made in `src/churnintel`, and
reads back the results produced by `make train`. Everything heavy lives in the package; the notebook only
*uses* it.""",
    ),
    (
        "code",
        """import sys, json
from pathlib import Path
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from churnintel.data import load_dataset, RAW_FEATURES
from churnintel.viz.style import apply_style, BLUE, ORANGE, AQUA, MUTED
apply_style()
pd.set_option("display.precision", 3)

X, y = load_dataset(ROOT / "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df = X.assign(Churn=y)
print(df.shape, f"churn rate = {y.mean():.1%}")""",
    ),
    ("md", "## 1. Who churns? Churn rate by key segments"),
    (
        "code",
        """def rate(col):
    g = df.groupby(col)["Churn"].agg(customers="size", churn_rate="mean")
    return g.sort_values("churn_rate", ascending=False)

for col in ["Contract", "InternetService", "PaymentMethod"]:
    display(rate(col))""",
    ),
    (
        "code",
        """df["tenure_band"] = pd.cut(df["tenure"], [-1, 6, 12, 24, 48, 72], labels=["0-6", "7-12", "13-24", "25-48", "49-72"])
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
r = df.groupby("tenure_band", observed=True)["Churn"].mean()
axes[0].bar(r.index.astype(str), r.values, color=BLUE, edgecolor="#fcfcfb", linewidth=2)
axes[0].set_title("Churn rate by tenure band (months)"); axes[0].yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))
axes[0].grid(axis="x", visible=False)
for label, color in [(0, BLUE), (1, ORANGE)]:
    axes[1].hist(df.loc[df.Churn == label, "MonthlyCharges"], bins=30, alpha=0.6, color=color, density=True,
                 label="stayed" if label == 0 else "churned")
axes[1].set_title("Monthly charges by outcome"); axes[1].legend(); axes[1].set_xlabel("$ per month")
plt.show()""",
    ),
    (
        "md",
        """**Takeaways:** month-to-month contracts, fiber-optic internet, electronic-check payment and the first
6–12 months are the high-risk segments. Churners skew towards higher monthly bills.""",
    ),
    (
        "md",
        "## 2. Why some engineered features were removed: MonthlyCharges is (almost) a function of services",
    ),
    (
        "code",
        """from sklearn.linear_model import LinearRegression
services = pd.get_dummies(df[["PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
                              "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]], drop_first=True).astype(float)
r2 = LinearRegression().fit(services, df["MonthlyCharges"]).score(services, df["MonthlyCharges"])
print(f"R^2 of MonthlyCharges ~ subscribed services: {r2:.4f}")""",
    ),
    (
        "md",
        """With R² ≈ 1, features such as `Fiber_x_Charge`, `ServiceDensity` or `NumServices` (v3) are near-linear
combinations of other columns. A logistic model can then trade huge positive weights on services against a
huge negative weight on price, which makes SHAP reason codes contradictory. v4 drops them (ΔAUC < 0.001) and
picks the logistic-regression penalty with a paired one-standard-error rule.""",
    ),
    ("md", "## 3. Results from `make train`"),
    (
        "code",
        """m = json.loads((ROOT / "reports/metrics.json").read_text())
t = m["test"]["champion"]; ci = t["bootstrap_ci"]
print(f"Champion: {m['selection']['champion_label']} ({m['calibration']['method']} calibration)")
print(f"Test ROC-AUC {t['roc_auc']:.4f}  (95% CI {ci['roc_auc']['ci_lower']:.4f}-{ci['roc_auc']['ci_upper']:.4f})")
print(f"Test PR-AUC {t['pr_auc']:.4f} | Brier {t['brier']:.4f} | top-10% lift {t['lift']['top_10pct']['lift']:.2f}x")
pd.read_csv(ROOT / "reports/model_comparison.csv", index_col=0)[
    ["model", "roc_auc_mean", "roc_auc_std", "brier_mean", "test_roc_auc", "test_pr_auc"]
].sort_values("roc_auc_mean", ascending=False)""",
    ),
    (
        "code",
        """pd.read_csv(ROOT / "reports/policy_comparison.csv", index_col=0)[
    ["customers_targeted", "churner_capture_rate", "campaign_cost", "net_profit", "net_profit_per_1k_customers", "roi"]
]""",
    ),
    ("md", "## 4. Scoring customers with reason codes (same code path as the API)"),
    (
        "code",
        """from churnintel.inference import ChurnPredictor
predictor = ChurnPredictor.load(ROOT / "artifacts/model_bundle.joblib")
test_ids = pd.read_csv(ROOT / "reports/test_predictions.csv")["customerID"]
sample = X.loc[test_ids].sample(5, random_state=3)
scored = predictor.score(sample, explain=True)
scored["top_reasons"] = [" | ".join(r["description"] for r in e["risk_drivers"]) for e in scored.pop("explanation")]
scored[["churn_probability", "risk_tier", "clv_estimate", "recommended_action", "expected_value", "top_reasons"]]""",
    ),
    ("md", "## 5. Dashboard\n\n![dashboard](../reports/figures/dashboard.png)"),
]


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        nbf.v4.new_markdown_cell(src) if kind == "md" else nbf.v4.new_code_cell(src) for kind, src in CELLS
    ]
    return nb


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    nb = build()
    if args.execute:
        from nbclient import NotebookClient

        NotebookClient(
            nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(NB.parent)}}
        ).execute()
    NB.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB)
    print(f"wrote {NB}")


if __name__ == "__main__":
    main()
