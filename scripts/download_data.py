#!/usr/bin/env python
"""Download the IBM Telco Customer Churn dataset (Apache-2.0) into data/raw/."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from churnintel.config import load_config  # noqa: E402


def main() -> None:
    cfg = load_config()
    target = cfg.path("raw_data")
    if target.exists():
        print(f"Already present: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    url = cfg["data"]["source_url"]
    print(f"Downloading {url}\n  -> {target}")
    urllib.request.urlretrieve(url, target)
    with open(target, encoding="utf-8") as fh:
        n = sum(1 for _ in fh) - 1
    print(f"Done: {n:,} rows")


if __name__ == "__main__":
    main()
