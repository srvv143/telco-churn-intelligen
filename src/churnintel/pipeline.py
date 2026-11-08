"""End-to-end training pipeline.

Protocol (leakage-free):
1. Stratified 80/20 split. The test set is used exactly once, in step 8.
2. Optuna tuning, repeated-CV model comparison, ensembles, champion selection (1-SE rule),
   calibration choice and decision thresholds all use *training data only* (CV / out-of-fold).
3. Survival-CLV model fitted on training data only.
4. One final evaluation on the test set: discrimination, calibration, bootstrap CIs, DeLong tests,
   business policies, fairness and drift. Everything is persisted for the API and the reports.
"""

from __future__ import annotations

import json
import logging
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from . import __version__
from .business import (
    compare_policies,
    load_actions,
    next_best_action,
    optimal_threshold,
    profit_curve,
    realized_value,
    sensitivity_grid,
)
from .config import Config, load_config
from .data import RAW_FEATURES, clean, load_raw, split_features_target, validate_dataframe
from .evaluation import (
    bootstrap_ci,
    classification_metrics,
    delong_roc_test,
    fairness_report,
    gains_table,
    lift_at,
    threshold_for_f1,
)
from .evaluation.metrics import threshold_metrics
from .explain import CONCEPT_LABELS, ChurnExplainer
from .features import heuristic_risk_score
from .models import (
    MODEL_LABELS,
    base_pipeline_of,
    build_ensemble,
    build_pipeline,
    cv_evaluate,
    make_calibrated,
    select_calibration,
    select_champion,
    summarize_cv,
    tune_model,
)
from .monitoring import DriftMonitor
from .survival import SurvivalCLV, kaplan_meier_by_group

logger = logging.getLogger(__name__)


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return None if not np.isfinite(obj) else float(obj)
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (pd.Timestamp, datetime, Path)):
        return str(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


class _Timer:
    def __init__(self, store: dict[str, float], key: str):
        self.store, self.key = store, key

    def __enter__(self):
        self.t = time.perf_counter()
        logger.info("── %s", self.key)
        return self

    def __exit__(self, *exc):
        self.store[self.key] = round(time.perf_counter() - self.t, 2)


def simulate_drift(X: pd.DataFrame, random_state: int = 42) -> pd.DataFrame:
    """Synthetic production shift: +15% price rise and 25% of term contracts lapsing to monthly."""
    rng = np.random.default_rng(random_state)
    out = X.copy()
    out["MonthlyCharges"] = out["MonthlyCharges"] * 1.15
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce") * 1.15
    term = (out["Contract"] != "Month-to-month").to_numpy()
    flip = term & (rng.random(len(out)) < 0.25)
    out.loc[flip, "Contract"] = "Month-to-month"
    return out


def run_training(config_path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    cfg: Config = load_config(config_path, overrides)
    rs = int(cfg["project"]["random_state"])
    n_jobs = int(cfg["project"].get("n_jobs", -1))
    art_dir, rep_dir, fig_dir = cfg.path("artifacts_dir"), cfg.path("reports_dir"), cfg.path("figures_dir")
    for d in (art_dir, rep_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t_start = time.perf_counter()

    # ------------------------------------------------------------------ 1. data
    with _Timer(timings, "data"):
        raw = clean(load_raw(cfg.path("raw_data")))
        validation = validate_dataframe(raw)
        validation.raise_if_failed()
        frac = cfg["data"].get("sample_frac")
        if frac and 0 < frac < 1:
            raw = raw.groupby("Churn", group_keys=False).sample(frac=frac, random_state=rs)
        X, y = split_features_target(raw)
        X = X[RAW_FEATURES]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=cfg["data"]["test_size"], stratify=y, random_state=rs
        )
        fe_kwargs = {
            "engineered": bool(cfg["features"]["engineered"]),
            "exclude": list(cfg["features"]["exclude"]),
        }
        logger.info("Train %d | Test %d | churn rate %.2f%%", len(X_train), len(X_test), 100 * y.mean())

    candidates = list(cfg["models"]["candidates"])
    n_splits, n_repeats = int(cfg["cv"]["n_splits"]), int(cfg["cv"]["n_repeats"])

    # ------------------------------------------------------------------ 2. tuning
    params: dict[str, dict] = {}
    tuning: dict[str, dict] = {}
    with _Timer(timings, "tuning"):
        for name in candidates:
            if cfg["tuning"]["enabled"]:
                res = tune_model(
                    name,
                    X_train,
                    y_train,
                    n_trials=int(cfg["tuning"]["n_trials"].get(name, 20)),
                    cv_splits=int(cfg["cv"]["tuning_splits"]),
                    random_state=rs,
                    fe_kwargs=fe_kwargs,
                    timeout=cfg["tuning"].get("timeout_seconds"),
                    n_jobs=n_jobs,
                )
                params[name] = res.best_params
                tuning[name] = res.to_dict()
            else:
                params[name] = {}

    # ------------------------------------------------------------------ 3. repeated CV + ensembles
    cv_folds: dict[str, pd.DataFrame] = {}
    cv_summary: dict[str, dict] = {}
    factories: dict[str, Any] = {}
    with _Timer(timings, "cross_validation"):
        for name in candidates:
            factories[name] = lambda name=name: build_pipeline(name, params[name], fe_kwargs, rs, n_jobs)
            cv_folds[name], _ = cv_evaluate(factories[name], X_train, y_train, n_splits, n_repeats, rs)
            cv_summary[name] = summarize_cv(cv_folds[name])
            logger.info(
                "CV %-14s AUC %.4f ± %.4f | Brier %.4f",
                name,
                cv_summary[name]["roc_auc_mean"],
                cv_summary[name]["roc_auc_std"],
                cv_summary[name]["brier_mean"],
            )
        ranked = sorted(candidates, key=lambda n: cv_summary[n]["roc_auc_mean"], reverse=True)
        members = {n: params[n] for n in ranked[: int(cfg["models"]["ensemble_size"])]}
        for kind in cfg["models"].get("ensembles", []):
            factories[kind] = lambda kind=kind: build_ensemble(kind, members, fe_kwargs, rs, n_jobs)
            cv_folds[kind], _ = cv_evaluate(factories[kind], X_train, y_train, n_splits, n_repeats, rs)
            cv_summary[kind] = summarize_cv(cv_folds[kind])
            logger.info(
                "CV %-14s AUC %.4f ± %.4f",
                kind,
                cv_summary[kind]["roc_auc_mean"],
                cv_summary[kind]["roc_auc_std"],
            )

    # ------------------------------------------------------------------ 4. champion
    champion, selection = select_champion(
        cv_summary, cv_folds, cfg["models"]["selection_rule"], list(cfg["models"]["complexity_order"])
    )
    selection["ensemble_members"] = list(members)

    # ------------------------------------------------------------------ 5. calibration
    with _Timer(timings, "calibration"):
        cal_method, cal_table, cal_oofs = select_calibration(
            factories[champion],
            X_train,
            y_train,
            cfg["calibration"]["method"],
            int(cfg["calibration"]["cv"]),
            rs,
        )
        oof = cal_oofs[cal_method]
        final_model = make_calibrated(factories[champion](), cal_method, int(cfg["calibration"]["cv"]), rs)
        final_model.fit(X_train, y_train)
        base_pipe = base_pipeline_of(final_model)

    # ------------------------------------------------------------------ 6. survival CLV
    with _Timer(timings, "survival"):
        clv_cfg = cfg["business"]["clv"]
        survival = SurvivalCLV(
            horizon_months=clv_cfg["horizon_months"],
            gross_margin=cfg["business"]["gross_margin"],
            penalizer=clv_cfg.get("cox_penalizer", 0.01),
            method=clv_cfg.get("method", "survival"),
        ).fit(X_train, y_train)
        clv_train, clv_test = survival.clv(X_train), survival.clv(X_test)
        c_index_test = survival.concordance(X_test, y_test)
        km = kaplan_meier_by_group(X_train, y_train, "Contract")

    # ------------------------------------------------------------------ 7. thresholds (train OOF only)
    actions = load_actions(cfg["business"])
    by_name = {a.name: a for a in actions}
    t_action = by_name[cfg["business"]["threshold_policy_action"]]
    t_star, t_grid, t_prof = optimal_threshold(oof, y_train, clv_train, t_action)
    t_f1 = threshold_for_f1(y_train.to_numpy(), oof)
    scen_thr, scen_prof = {}, {}
    for label, cost in cfg["business"]["threshold_scenarios"].items():
        a = t_action.with_(incentive_cost=float(cost))
        pretty = f"{label.replace('_', ' ')} (${cost:,.0f})"
        scen_thr[pretty], _, scen_prof[pretty] = optimal_threshold(oof, y_train, clv_train, a)

    # ------------------------------------------------------------------ 8. TEST SET (used once)
    with _Timer(timings, "test_evaluation"):
        yt = y_test.to_numpy()
        p_test = final_model.predict_proba(X_test)[:, 1]
        p_test_uncal = base_pipe.predict_proba(X_test)[:, 1]
        heur_test = heuristic_risk_score(X_test)

        model_scores: dict[str, np.ndarray] = {}
        for name in list(cv_summary):
            if name == champion:
                model_scores[name] = p_test_uncal
            else:
                m = factories[name]()
                m.fit(X_train, y_train)
                model_scores[name] = m.predict_proba(X_test)[:, 1]
        test_models = {
            n: {
                "roc_auc": roc_auc_score(yt, s),
                "pr_auc": average_precision_score(yt, s),
                "brier": brier_score_loss(yt, s),
            }
            for n, s in model_scores.items()
        }
        test_models["heuristic"] = {
            "roc_auc": roc_auc_score(yt, heur_test),
            "pr_auc": average_precision_score(yt, heur_test),
        }

        lift_fracs = tuple(cfg["evaluation"]["lift_fractions"])
        champ_metrics = classification_metrics(yt, p_test, 0.5, lift_fracs)
        champ_metrics["at_profit_threshold"] = threshold_metrics(yt, p_test, t_star)
        champ_metrics["at_f1_threshold"] = threshold_metrics(yt, p_test, t_f1)
        ci = bootstrap_ci(
            yt,
            p_test,
            {
                "roc_auc": roc_auc_score,
                "pr_auc": average_precision_score,
                "brier": brier_score_loss,
                "lift_top10": lambda a, b: lift_at(a, b, 0.10)["lift"],
                "capture_top30": lambda a, b: lift_at(a, b, 0.30)["capture_rate"],
                "accuracy_at_0.5": lambda a, b: float(((b >= 0.5).astype(int) == a).mean()),
            },
            n_boot=int(cfg["evaluation"]["bootstrap_samples"]),
            ci=float(cfg["evaluation"]["ci_level"]),
            random_state=rs,
        )
        delong = {n: delong_roc_test(yt, p_test, s) for n, s in model_scores.items() if n != champion}
        delong["heuristic"] = delong_roc_test(yt, p_test, heur_test)
        gains = gains_table(yt, p_test)
        cm = confusion_matrix(yt, (p_test >= t_star).astype(int), labels=[0, 1])

    # ------------------------------------------------------------------ 9. business on test
    with _Timer(timings, "business"):
        policies = compare_policies(p_test, yt, clv_test, heur_test, actions, t_action.name, t_star)
        fr = np.linspace(0, 1, 101)
        _, prof_model = profit_curve(p_test, yt, clv_test, t_action, fr)
        _, prof_heur = profit_curve(heur_test, yt, clv_test, t_action, fr)
        prof_random = fr * realized_value(yt, clv_test, t_action).sum()
        sens = sensitivity_grid(
            oof,
            y_train.to_numpy(),
            clv_train,
            p_test,
            yt,
            clv_test,
            t_action,
            cfg["business"]["sensitivity"]["save_rates"],
            cfg["business"]["sensitivity"]["incentive_costs"],
        )
        nba_idx, nba_ev = next_best_action(p_test, clv_test, actions)

    # ------------------------------------------------------------------ 10. explainability
    with _Timer(timings, "explainability"):
        explainer = ChurnExplainer(base_pipe, X_train, int(cfg["explain"]["background_rows"]), rs)
        n_sum = min(int(cfg["explain"]["summary_rows"]), len(X_test))
        if explainer.kind_ == "agnostic":  # occlusion is O(rows x concepts x background): keep it small
            n_sum = min(n_sum, 100)
        X_sum = X_test.sample(n_sum, random_state=rs)
        shap_vals, _, Xt_sum = explainer.shap_values(X_sum)
        concept_imp = explainer.concept_values(X_sum).abs().mean().sort_values(ascending=False)
        concept_imp.index = [CONCEPT_LABELS.get(c, c) for c in concept_imp.index]
        top_idx = int(np.argmax(p_test))
        top_customer = X_test.iloc[[top_idx]]
        top_reasons = explainer.reasons(top_customer, top_k=int(cfg["explain"]["top_k_reasons"]))[0]
        top_concepts = explainer.concept_values(top_customer).iloc[0]
        top_contrib = sorted(
            [
                (desc_for(c, top_customer.iloc[0], v), float(v))
                for c, v in top_concepts.items()
                if abs(v) > 1e-9
            ],
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:8]

    # ------------------------------------------------------------------ 11. fairness + drift
    with _Timer(timings, "fairness_drift"):
        decision = (p_test >= t_star).astype(int)
        fair_groups = {g: X_test[g].astype(str) for g in cfg["fairness"]["groups"] if g in X_test.columns}
        fair_detail, fair_summary = fairness_report(yt, p_test, decision, fair_groups)
        mon_cfg = cfg["monitoring"]
        monitor = DriftMonitor(mon_cfg["psi_bins"], mon_cfg["psi_warn"], mon_cfg["psi_alert"]).fit(
            X_train, oof
        )
        psi_test = monitor.compute(X_test, p_test)
        X_sim = simulate_drift(X_test, rs)
        psi_sim = monitor.compute(X_sim, final_model.predict_proba(X_sim)[:, 1])

    # ------------------------------------------------------------------ 12. persist bundle
    trained_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    metadata = {
        "model_name": MODEL_LABELS.get(champion, champion),
        "model_key": champion,
        "version": cfg["project"]["version"],
        "package_version": __version__,
        "trained_at": trained_at,
        "calibration": cal_method,
        "decision_threshold": t_star,
        "params": params.get(champion, members if champion in ("soft_voting", "stacking") else {}),
        "features_excluded": fe_kwargs["exclude"],
        "n_train": len(X_train),
        "test_roc_auc": champ_metrics["roc_auc"],
        "test_pr_auc": champ_metrics["pr_auc"],
        "sklearn_version": sklearn.__version__,
        "python_version": platform.python_version(),
    }
    bundle = {
        "model": final_model,
        "explainer": explainer,
        "survival": survival,
        "actions": [a.to_dict() for a in actions],
        "risk_tiers": dict(cfg["evaluation"]["risk_tiers"]),
        "decision_threshold": t_star,
        "drift_monitor": monitor,
        "metadata": metadata,
    }
    bundle_path = art_dir / "model_bundle.joblib"
    joblib.dump(bundle, bundle_path, compress=3)
    (art_dir / "model_metadata.json").write_text(json.dumps(to_jsonable(metadata), indent=2))
    logger.info("Saved model bundle -> %s (%.1f MB)", bundle_path, bundle_path.stat().st_size / 1e6)

    # ------------------------------------------------------------------ 13. reports
    timings["total"] = round(time.perf_counter() - t_start, 1)
    pd.concat({k: v for k, v in cv_folds.items()}, names=["model"]).to_csv(rep_dir / "cv_folds.csv")
    cv_table = pd.DataFrame(cv_summary).T
    cv_table.insert(0, "model", [MODEL_LABELS.get(k, k) for k in cv_table.index])
    cv_table["test_roc_auc"] = [test_models[k]["roc_auc"] for k in cv_table.index]
    cv_table["test_pr_auc"] = [test_models[k]["pr_auc"] for k in cv_table.index]
    cv_table.to_csv(rep_dir / "model_comparison.csv")
    policies.to_csv(rep_dir / "policy_comparison.csv")
    gains.to_csv(rep_dir / "gains_table.csv", index=False)
    cal_table.to_csv(rep_dir / "calibration_comparison.csv")
    fair_detail.to_csv(rep_dir / "fairness_groups.csv", index=False)
    psi_test.to_csv(rep_dir / "drift_test_vs_train.csv", index=False)
    psi_sim.to_csv(rep_dir / "drift_simulated.csv", index=False)
    survival.hazard_ratios_.to_csv(rep_dir / "cox_hazard_ratios.csv")
    sens.to_csv(rep_dir / "profit_sensitivity.csv")
    pd.DataFrame(
        {
            "customerID": X_test.index,
            "churned": yt,
            "churn_probability": p_test,
            "uncalibrated_score": p_test_uncal,
            "heuristic_score": heur_test,
            "clv_estimate": clv_test,
            "recommended_action": [actions[i].name if i >= 0 else "no_action" for i in nba_idx],
            "expected_value": nba_ev,
        }
    ).to_csv(rep_dir / "test_predictions.csv", index=False)

    base_actions = {a.name: a.to_dict() for a in actions}
    nba = policies.loc["ML next-best-action (EV-optimal)"]
    heur_row = policies.loc["Rule-based heuristic (same volume)"]
    thr_row = policies.loc[[i for i in policies.index if i.startswith("ML threshold")][0]]
    metrics = {
        "project": {
            "version": cfg["project"]["version"],
            "trained_at": trained_at,
            "runtime_seconds": timings["total"],
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
        },
        "data": {
            "n_rows": len(X),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "churn_rate": float(y.mean()),
            "n_raw_features": len(RAW_FEATURES),
            "n_model_features": len(explainer.feature_names_),
            "validation": validation.to_dict(),
        },
        "tuning": tuning,
        "cv": cv_summary,
        "selection": selection
        | {"champion": champion, "champion_label": MODEL_LABELS.get(champion, champion)},
        "calibration": {"method": cal_method, "comparison": cal_table.to_dict(orient="index")},
        "test": {
            "champion": champ_metrics | {"bootstrap_ci": ci.to_dict(orient="index")},
            "uncalibrated_champion": {
                "brier": brier_score_loss(yt, p_test_uncal),
                "ece": float(classification_metrics(yt, p_test_uncal)["ece"]),
            },
            "models": test_models,
            "delong_vs_champion": delong,
            "gains": gains.to_dict(orient="records"),
            "confusion_matrix_at_profit_threshold": cm,
        },
        "thresholds": {"profit_optimal": t_star, "f1_optimal": t_f1, "scenarios": scen_thr},
        "business": {
            "assumptions": {
                "gross_margin": cfg["business"]["gross_margin"],
                "clv_method": clv_cfg.get("method"),
                "clv_horizon_months": clv_cfg["horizon_months"],
                "actions": base_actions,
                "threshold_policy_action": t_action.name,
            },
            "clv_test_mean": float(clv_test.mean()),
            "clv_test_median": float(np.median(clv_test)),
            "policies": policies.to_dict(orient="index"),
            "headline": {
                "nba_net_profit_test": float(nba["net_profit"]),
                "nba_net_profit_per_1k": float(nba["net_profit_per_1k_customers"]),
                "nba_roi": float(nba["roi"]),
                "nba_pct_targeted": float(nba["pct_targeted"]),
                "nba_capture_rate": float(nba["churner_capture_rate"]),
                "threshold_policy_net_profit": float(thr_row["net_profit"]),
                "threshold_policy_capture_rate": float(thr_row["churner_capture_rate"]),
                "heuristic_same_volume_net_profit": float(heur_row["net_profit"]),
                "uplift_vs_heuristic_same_volume": float(thr_row["net_profit"] / heur_row["net_profit"])
                if heur_row["net_profit"] > 0
                else None,
                "blanket_net_profit": float(
                    policies.loc["Blanket campaign (contact everyone)", "net_profit"]
                ),
                "oracle_net_profit": float(
                    policies.loc["Oracle (perfect foresight, reference)", "net_profit"]
                ),
                "sensitivity_profitable_share": float((sens.to_numpy(dtype=float) > 0).mean()),
            },
            "action_mix_test": pd.Series([actions[i].name if i >= 0 else "no_action" for i in nba_idx])
            .value_counts()
            .to_dict(),
        },
        "survival": {
            "c_index_train": survival.train_concordance_,
            "c_index_test": c_index_test,
            "km_median_months_by_contract": km["median_months"],
            "logrank_p_contract": km["logrank_p"],
            "hazard_ratios": survival.hazard_ratios_.to_dict(orient="index"),
        },
        "explainability": {
            "explainer": explainer.kind_,
            "units": explainer.units_,
            "global_concept_importance": concept_imp.round(4).to_dict(),
            "highest_risk_customer": {
                "customerID": str(X_test.index[top_idx]),
                "churn_probability": float(p_test[top_idx]),
                "reasons": top_reasons,
            },
        },
        "fairness": {
            "summary": fair_summary.to_dict(orient="records"),
            "groups": fair_detail.to_dict(orient="records"),
        },
        "drift": {
            "test_vs_train_max_psi": float(psi_test["psi"].max()),
            "test_vs_train_alerts": int((psi_test["status"] != "ok").sum()),
            "simulated": psi_sim.head(8).to_dict(orient="records"),
            "simulated_alerts": int((psi_sim["status"] == "alert").sum()),
        },
        "timings_seconds": timings,
    }
    metrics = to_jsonable(metrics)
    (rep_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # ------------------------------------------------------------------ 14. figures
    with _Timer(timings, "figures"):
        from .viz.figures import make_all_figures

        make_all_figures(
            fig_dir,
            champion_label=MODEL_LABELS.get(champion, champion),
            cv_summary={MODEL_LABELS.get(k, k): v for k, v in cv_summary.items()},
            selection=selection,
            y_test=yt,
            model_scores={MODEL_LABELS.get(k, k): v for k, v in model_scores.items()},
            heuristic=heur_test,
            p_test=p_test,
            p_test_uncal=p_test_uncal,
            cal_method=cal_method,
            gains=gains,
            profit_curves={
                "Champion model ranking": (fr, prof_model),
                "Rule-based heuristic ranking": (fr, prof_heur),
                "Random targeting": (fr, prof_random),
            },
            threshold_grid=t_grid,
            scenario_profits=scen_prof,
            scenario_thresholds=scen_thr,
            sensitivity=sens,
            base_action=(t_action.save_rate, t_action.incentive_cost),
            policies=policies,
            shap_values=shap_vals,
            shap_X=Xt_sum,
            feature_names=explainer.feature_names_,
            concept_importance=concept_imp,
            explainer_kind=explainer.kind_,
            shap_units=explainer.units_,
            local_contrib=top_contrib,
            local_prob=float(p_test[top_idx]),
            local_id=str(X_test.index[top_idx]),
            km=km,
            hazard_ratios=survival.hazard_ratios_,
            fairness=fair_detail,
            fairness_attrs=list(fair_groups),
            psi_test=psi_test,
            psi_sim=psi_sim,
            psi_levels=(mon_cfg["psi_warn"], mon_cfg["psi_alert"]),
            confusion=cm,
            t_star=t_star,
            headline={
                "auc": champ_metrics["roc_auc"],
                "auc_ci": (ci.loc["roc_auc", "ci_lower"], ci.loc["roc_auc", "ci_upper"]),
                "cv_auc": cv_summary[champion]["roc_auc_mean"],
                "cv_std": cv_summary[champion]["roc_auc_std"],
            },
        )

    from .reporting import write_reports

    write_reports(rep_dir)
    logger.info(
        "Done in %.1fs. Champion=%s test AUC=%.4f", timings["total"], champion, champ_metrics["roc_auc"]
    )
    return metrics


def desc_for(concept: str, row: pd.Series, value: float) -> str:
    from .explain.explainer import describe

    return describe(concept, row, value)
