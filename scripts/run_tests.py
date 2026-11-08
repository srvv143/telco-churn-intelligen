#!/usr/bin/env python
"""Run the test suite with coverage and write reports/test_summary.json (used by the reports).

python scripts/run_tests.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        junit, cov = Path(tmp) / "junit.xml", Path(tmp) / "coverage.json"
        start = time.perf_counter()
        code = subprocess.call(
            [
                sys.executable,
                "-m",
                "pytest",
                "--cov",
                f"--cov-report=json:{cov}",
                "--cov-report=term",
                f"--junitxml={junit}",
                "-p",
                "no:cacheprovider",
            ],
            cwd=ROOT,
        )
        seconds = time.perf_counter() - start
        suite = ET.parse(junit).getroot()
        suite = suite if suite.tag == "testsuite" else suite.find("testsuite")
        total = int(suite.get("tests", 0))
        failed = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
        skipped = int(suite.get("skipped", 0))
        coverage = json.loads(cov.read_text())["totals"]["percent_covered"]
    summary = {
        "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "total": total,
        "passed": total - failed - skipped,
        "failed": failed,
        "skipped": skipped,
        "coverage_pct": round(coverage, 1),
        "seconds": round(seconds, 1),
    }
    (ROOT / "reports" / "test_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return code


if __name__ == "__main__":
    sys.exit(main())
