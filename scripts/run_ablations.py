#!/usr/bin/env python
"""Ablation studies on the training split — every variant uses the *same* repeated stratified folds,
so differences are paired (mean ± SE of the per-fold difference).

A1  Feature engineering   engineered features vs. raw inputs only
A2  Class imbalance       none (reference) vs. class_weight='balanced' vs. SMOTE inside each fold
A3  Gender input          excluded (reference) vs. included

Models: logistic regression and LightGBM with the hyper-parameters tuned by `make train`
(falls back to defaults if reports/metrics.json is missing).

    python scripts/run_ablations.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from churnintel.config import load_config  # noqa: E402
from churnintel.data import RAW_FEATURES, load_dataset  # noqa: E402
from churnintel.features import TelcoFeatureEngineer, build_preprocessor  # noqa: E402
from churnintel.logging_utils import setup_logging  # noqa: E402
from churnintel.models import (  # noqa: E402
    DEFAULT_PARAMS,
    build_estimator,
    build_pipeline,
    cv_evaluate,
    summarize_cv,
)

MODELS = {"logreg": "Logistic Regression", "lightgbm": "LightGBM"}


def smote_pipeline(name: str, params: dict, fe_kwargs: dict, rs: int):
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    fe = TelcoFeatureEngineer(**fe_kwargs)
    num, nom = fe.output_columns()
    return ImbPipeline(
        [
            ("features", fe),
            ("preprocess", build_preprocessor(num, nom)),
            ("smote", SMOTE(random_state=rs)),
            ("model", build_estimator(name, params, rs)),
        ]
    )


def main() -> None:
    setup_logging("WARNING")
    cfg = load_config()
    rs = cfg["project"]["random_state"]
    n_splits, n_repeats = cfg["cv"]["n_splits"], cfg["cv"]["n_repeats"]
    X, y = load_dataset(cfg.path("raw_data"))
    X = X[RAW_FEATURES]
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=rs
    )

    metrics_path = cfg.path("reports_dir") / "metrics.json"
    tuned = json.loads(metrics_path.read_text())["tuning"] if metrics_path.exists() else {}
    base_fe = {"engineered": True, "exclude": list(cfg["features"]["exclude"])}

    def variants(name: str, params: dict) -> dict:
        def with_weight():
            pipe = build_pipeline(name, params, base_fe, rs)
            pipe.set_params(model__class_weight="balanced")
            return pipe

        return {
            ("reference", "default"): lambda: build_pipeline(name, params, base_fe, rs),
            ("A1 feature engineering", "raw inputs only"): lambda: build_pipeline(
                name, params, {**base_fe, "engineered": False}, rs
            ),
            ("A2 class imbalance", "class_weight=balanced"): with_weight,
            ("A2 class imbalance", "SMOTE (in-fold)"): lambda: smote_pipeline(name, params, base_fe, rs),
            ("A3 gender input", "gender included"): lambda: build_pipeline(
                name, params, {**base_fe, "exclude": [c for c in base_fe["exclude"] if c != "gender"]}, rs
            ),
        }

    rows, folds_store = [], {}
    start = time.perf_counter()
    for name, label in MODELS.items():
        params = tuned.get(name, {}).get("best_params", DEFAULT_PARAMS[name])
        for (exp, variant), factory in variants(name, params).items():
            folds, _ = cv_evaluate(factory, X_train, y_train, n_splits, n_repeats, rs)
            folds_store[(name, exp, variant)] = folds
            s = summarize_cv(folds)
            print(
                f"{label:<20} {exp:<24} {variant:<24} AUC {s['roc_auc_mean']:.4f}  Brier {s['brier_mean']:.4f}  ECE {s['ece_mean']:.4f}"
            )
            rows.append({"model": label, "model_key": name, "experiment": exp, "variant": variant, **s})

    ref = {name: folds_store[(name, "reference", "default")].sort_values("fold") for name in MODELS}
    for r in rows:
        if r["experiment"] == "reference":
            r["delta_auc"] = r["delta_auc_se"] = r["delta_brier"] = None
            continue
        f = folds_store[(r["model_key"], r["experiment"], r["variant"])].sort_values("fold")
        d = f["roc_auc"].to_numpy() - ref[r["model_key"]]["roc_auc"].to_numpy()
        db = f["brier"].to_numpy() - ref[r["model_key"]]["brier"].to_numpy()
        r["delta_auc"] = float(d.mean())
        r["delta_auc_se"] = float(d.std(ddof=1) / np.sqrt(len(d)))
        r["delta_brier"] = float(db.mean())

    def verdict(r) -> str:
        z = r["delta_auc"] / r["delta_auc_se"] if r["delta_auc_se"] else 0.0
        if abs(r["delta_auc"]) < 0.001:
            word = "negligible AUC change"
        elif abs(z) < 2:
            word = "no significant AUC change"
        else:
            word = "AUC gain" if z > 0 else "AUC loss"
        return (
            f"{r['model']} — {r['variant']}: ΔAUC {r['delta_auc']:+.4f} ± {r['delta_auc_se']:.4f} ({word}), "
            f"ΔBrier {r['delta_brier']:+.4f} ({'worse' if r['delta_brier'] > 0 else 'better'} calibration/accuracy of probabilities)"
        )

    conclusions = [verdict(r) for r in rows if r["experiment"] != "reference"]
    table = pd.DataFrame(rows)
    out = cfg.path("reports_dir")
    table.to_csv(out / "ablation_results.csv", index=False)
    keep = [
        "experiment",
        "variant",
        "model",
        "roc_auc_mean",
        "roc_auc_std",
        "brier_mean",
        "ece_mean",
        "delta_auc",
        "delta_auc_se",
        "delta_brier",
    ]
    (out / "ablation_results.json").write_text(
        json.dumps(
            {
                "protocol": f"{n_repeats}x{n_splits} repeated stratified CV on the training split; paired differences vs. reference",
                "rows": table[keep].replace({np.nan: None}).to_dict(orient="records"),
                "conclusions": conclusions,
                "seconds": round(time.perf_counter() - start, 1),
            },
            indent=2,
        )
    )
    print("\n" + "\n".join(conclusions))
    print(f"\nSaved -> {out / 'ablation_results.json'}")


if __name__ == "__main__":
    main()
