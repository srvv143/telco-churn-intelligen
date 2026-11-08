#!/usr/bin/env python
"""Score a CSV of customers: probability, risk tier, CLV, next-best-action, reason codes.

python scripts/batch_score.py --input data/sample/sample_customers.csv --output reports/batch_predictions.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402

from churnintel.data import validate_dataframe  # noqa: E402
from churnintel.data.loader import clean  # noqa: E402
from churnintel.inference import ChurnPredictor  # noqa: E402
from churnintel.logging_utils import setup_logging  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--output", default="reports/batch_predictions.csv")
    p.add_argument("--model", default=str(ROOT / "artifacts" / "model_bundle.joblib"))
    p.add_argument("--explain", action="store_true", help="add top reason codes (slower)")
    args = p.parse_args()
    setup_logging()

    df = pd.read_csv(args.input)
    report = validate_dataframe(clean(df), require_target=False)
    for w in report.warnings:
        print("WARN:", w)
    report.raise_if_failed()

    predictor = ChurnPredictor.load(args.model)
    scored = predictor.score(df, explain=args.explain)
    if args.explain:
        scored["top_reasons"] = [
            " | ".join(r["description"] for r in e["risk_drivers"]) for e in scored.pop("explanation")
        ]
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(args.output)
    print(f"Scored {len(scored):,} customers -> {args.output}")
    print(scored["recommended_action"].value_counts().to_string())
    print(f"Total expected value of recommended actions: ${scored['expected_value'].sum():,.0f}")


if __name__ == "__main__":
    main()
