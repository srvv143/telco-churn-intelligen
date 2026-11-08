#!/usr/bin/env python
"""Re-render RESULTS.md, model_card.md and RESUME.md from the JSON outputs (no retraining)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churnintel.reporting import write_reports  # noqa: E402

if __name__ == "__main__":
    for name, path in write_reports(ROOT / "reports").items():
        print(f"{name:<11} -> {path.relative_to(ROOT)}")
