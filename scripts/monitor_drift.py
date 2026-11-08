#!/usr/bin/env python
"""Compare a new batch of customers to the training reference with the Population Stability Index.

    python scripts/monitor_drift.py --input new_customers.csv
    python scripts/monitor_drift.py --simulate     # demo: +15% price, contract lapse on the test split

Exit code 2 when any feature (or the model score) is in ALERT, so it can gate a scheduled job.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from churnintel.inference import ChurnPredictor  # noqa: E402
from churnintel.logging_utils import setup_logging  # noqa: E402
from churnintel.pipeline import simulate_drift  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--model", default=str(ROOT / "artifacts" / "model_bundle.joblib"))
    p.add_argument("--output", default=str(ROOT / "reports" / "drift_report.csv"))
    args = p.parse_args()
    setup_logging("WARNING")

    predictor = ChurnPredictor.load(args.model)
    if args.simulate:
        test_ids = pd.read_csv(ROOT / "reports" / "test_predictions.csv")["customerID"]
        raw = pd.read_csv(ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv")
        batch = simulate_drift(predictor.prepare(raw[raw["customerID"].isin(test_ids)]))
    elif args.input:
        batch = pd.read_csv(args.input)
    else:
        p.error("pass --input or --simulate")

    report = predictor.drift_report(batch)
    report.to_csv(args.output, index=False)
    print(report.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    alerts = report[report["status"] == "alert"]
    print(f"\n{len(alerts)} alert(s); report -> {args.output}")
    sys.exit(2 if len(alerts) else 0)


if __name__ == "__main__":
    main()
