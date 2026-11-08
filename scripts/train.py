#!/usr/bin/env python
"""Train the full pipeline and write artifacts + reports.

python scripts/train.py                              # full run (configs/config.yaml)
python scripts/train.py --config configs/config.fast.yaml   # ~1 minute smoke run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from churnintel.logging_utils import setup_logging  # noqa: E402
from churnintel.pipeline import run_training  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    setup_logging(args.log_level)
    metrics = run_training(args.config)
    t = metrics["test"]["champion"]
    ci = t["bootstrap_ci"]["roc_auc"]
    print("\n" + "=" * 72)
    print(
        f" Champion          : {metrics['selection']['champion_label']} ({metrics['calibration']['method']} calibration)"
    )
    print(f" Test ROC-AUC      : {t['roc_auc']:.4f}  (95% CI {ci['ci_lower']:.4f}-{ci['ci_upper']:.4f})")
    print(f" Test PR-AUC       : {t['pr_auc']:.4f}   Brier {t['brier']:.4f}   ECE {t['ece']:.4f}")
    print(f" Top-10% lift      : {t['lift']['top_10pct']['lift']:.2f}x")
    print(f" NBA net profit/1k : ${metrics['business']['headline']['nba_net_profit_per_1k']:,.0f}")
    print(f" Runtime           : {metrics['project']['runtime_seconds'] / 60:.1f} min")
    print("=" * 72)


if __name__ == "__main__":
    main()
