#!/usr/bin/env python
"""Measure real HTTP latency of the API (uvicorn, single worker) and write reports/api_benchmark.json.

python scripts/benchmark_api.py --requests 500
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

import httpx  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def wait_for(url: str, timeout: float = 60.0) -> None:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if httpx.get(url, timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"API did not start: {url}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--requests", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=1000)
    p.add_argument("--port", type=int, default=8765)
    args = p.parse_args()

    raw = pd.read_csv(ROOT / "data" / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv").drop(columns=["Churn"])
    raw["TotalCharges"] = pd.to_numeric(raw["TotalCharges"].replace(" ", np.nan), errors="coerce")
    records = json.loads(
        raw.sample(args.requests, random_state=0, replace=args.requests > len(raw)).to_json(orient="records")
    )
    batch = json.loads(
        raw.sample(args.batch_size, random_state=1, replace=args.batch_size > len(raw)).to_json(
            orient="records"
        )
    )

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    server = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "api.main:app", "--port", str(args.port), "--log-level", "warning"],
        cwd=ROOT,
        env=env,
    )
    base = f"http://127.0.0.1:{args.port}"
    try:
        wait_for(base + "/health")
        with httpx.Client(base_url=base, timeout=30) as client:
            for r in records[:20]:  # warm-up (lazy explainer init, caches)
                client.post("/v1/predict", json=r)
            lat, server_ms = [], []
            for r in records:
                t = time.perf_counter()
                resp = client.post("/v1/predict", json=r)
                lat.append((time.perf_counter() - t) * 1000)
                resp.raise_for_status()
                server_ms.append(float(resp.headers["X-Process-Time-Ms"]))
            t = time.perf_counter()
            resp = client.post("/v1/predict/batch", json={"customers": batch})
            batch_s = time.perf_counter() - t
            resp.raise_for_status()
    finally:
        server.terminate()
        server.wait(timeout=10)

    lat = np.asarray(lat)
    result = {
        "measured_at": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "machine": f"{platform.system()} {platform.machine()}, {os.cpu_count()} vCPU, Python {platform.python_version()}",
        "n_requests": len(lat),
        "single": {
            "endpoint": "POST /v1/predict?explain=true",
            "p50_ms": float(np.percentile(lat, 50)),
            "p95_ms": float(np.percentile(lat, 95)),
            "p99_ms": float(np.percentile(lat, 99)),
            "mean_ms": float(lat.mean()),
            "server_p50_ms": float(np.percentile(server_ms, 50)),
            "throughput_rps_sequential": float(1000.0 / lat.mean()),
        },
        "batch": {
            "endpoint": "POST /v1/predict/batch?explain=false",
            "batch_size": len(batch),
            "seconds": batch_s,
            "rows_per_second": len(batch) / batch_s,
        },
    }
    out = ROOT / "reports" / "api_benchmark.json"
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
