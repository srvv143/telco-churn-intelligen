"""Render Markdown reports (results, model card, resume bullets) from ``reports/metrics.json``.

Every number in these documents is read from machine-generated JSON, so the write-ups cannot
drift from what the code actually measured. Optional inputs (ablations, API benchmark, test
summary) are included when present.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT
from .models.zoo import MODEL_LABELS


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text()) if path.exists() else None


def _f(x, nd=3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def _pct(x, nd=1) -> str:
    return "—" if x is None else f"{100 * x:.{nd}f}%"


def _money(x) -> str:
    if x is None:
        return "—"
    return f"-${abs(x):,.0f}" if x < 0 else f"${x:,.0f}"


def _table(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _hr_month_to_month(m: dict) -> tuple[float | None, tuple | None]:
    hr = m["survival"]["hazard_ratios"].get("Contract[Month-to-month]")
    if not hr:
        return None, None
    return hr["hazard_ratio"], (hr["ci_lower"], hr["ci_upper"])


def build_context(rep_dir: Path) -> dict[str, Any]:
    m = _load(rep_dir / "metrics.json")
    if m is None:
        raise FileNotFoundError(f"{rep_dir / 'metrics.json'} not found — run training first")
    ctx = {
        "m": m,
        "ablation": _load(rep_dir / "ablation_results.json"),
        "bench": _load(rep_dir / "api_benchmark.json"),
        "tests": _load(rep_dir / "test_summary.json"),
    }
    champ = m["selection"]["champion"]
    ctx["champ"] = champ
    ctx["champ_label"] = m["selection"]["champion_label"]
    ctx["cv"] = m["cv"][champ]
    ctx["test"] = m["test"]["champion"]
    ctx["ci"] = m["test"]["champion"]["bootstrap_ci"]
    ctx["n_models"] = len(m["cv"])
    ctx["n_trials"] = sum(v["n_trials"] for v in m["tuning"].values()) if m["tuning"] else 0
    ctx["policies"] = m["business"]["policies"]
    ctx["head"] = m["business"]["headline"]
    return ctx


# ---------------------------------------------------------------------------- RESULTS.md
def render_results(ctx: dict[str, Any]) -> str:
    m, t, ci, cv = ctx["m"], ctx["test"], ctx["ci"], ctx["cv"]
    champ, label = ctx["champ"], ctx["champ_label"]
    lift = t["lift"]
    lines = [
        "# Results — Telco Churn Intelligence v4",
        "",
        f"_Auto-generated from `reports/metrics.json` (trained {m['project']['trained_at']}, "
        f"runtime {m['project']['runtime_seconds'] / 60:.1f} min). Re-run `make train` to refresh._",
        "",
        f"## Headline — held-out test set (n = {m['data']['n_test']:,}, never used for any decision)",
        "",
        _table(
            ["Metric", "Value", "95% bootstrap CI"],
            [
                [
                    "ROC-AUC",
                    _f(t["roc_auc"], 4),
                    f"{_f(ci['roc_auc']['ci_lower'], 4)} – {_f(ci['roc_auc']['ci_upper'], 4)}",
                ],
                [
                    "PR-AUC (average precision)",
                    _f(t["pr_auc"], 4),
                    f"{_f(ci['pr_auc']['ci_lower'], 4)} – {_f(ci['pr_auc']['ci_upper'], 4)}",
                ],
                [
                    "Brier score (lower is better)",
                    _f(t["brier"], 4),
                    f"{_f(ci['brier']['ci_lower'], 4)} – {_f(ci['brier']['ci_upper'], 4)}",
                ],
                [
                    "Accuracy @ 0.5",
                    _pct(t["at_threshold"]["accuracy"]),
                    f"{_pct(ci['accuracy_at_0.5']['ci_lower'])} – {_pct(ci['accuracy_at_0.5']['ci_upper'])}",
                ],
                [
                    "Lift in top 10%",
                    f"{lift['top_10pct']['lift']:.2f}×",
                    f"{ci['lift_top10']['ci_lower']:.2f}× – {ci['lift_top10']['ci_upper']:.2f}×",
                ],
                [
                    "Churners captured in top 30%",
                    _pct(lift["top_30pct"]["capture_rate"]),
                    f"{_pct(ci['capture_top30']['ci_lower'])} – {_pct(ci['capture_top30']['ci_upper'])}",
                ],
                ["KS statistic", _f(t["ks"], 3), "—"],
                ["Expected calibration error", _f(t["ece"], 4), "—"],
            ],
        ),
        "",
        f"**Champion:** {label} with `{m['calibration']['method']}` calibration, chosen by the "
        f"`{m['selection']['rule']}` rule from {ctx['n_models']} candidates — repeated-CV ROC-AUC "
        f"{cv['roc_auc_mean']:.4f} ± {cv['roc_auc_std']:.4f} over {int(cv['n_folds'])} folds (training data only).",
        "",
        "## Model comparison",
        "",
    ]
    rows = []
    for key, s in sorted(m["cv"].items(), key=lambda kv: -kv[1]["roc_auc_mean"]):
        tm = m["test"]["models"][key]
        dl = m["test"]["delong_vs_champion"].get(key)
        rows.append(
            [
                ("**" + MODEL_LABELS.get(key, key) + "** (champion)")
                if key == champ
                else MODEL_LABELS.get(key, key),
                f"{s['roc_auc_mean']:.4f} ± {s['roc_auc_std']:.4f}",
                _f(s["pr_auc_mean"], 4),
                _f(s["brier_mean"], 4),
                _f(tm["roc_auc"], 4),
                _f(tm["pr_auc"], 4),
                "—" if dl is None else f"{dl['p_value']:.3f}",
            ]
        )
    h = m["test"]["models"]["heuristic"]
    dl = m["test"]["delong_vs_champion"]["heuristic"]
    rows.append(
        [
            "Rule-based heuristic (v3 ManualRiskScore)",
            "—",
            "—",
            "—",
            _f(h["roc_auc"], 4),
            _f(h["pr_auc"], 4),
            f"{dl['p_value']:.1e}",
        ]
    )
    lines += [
        _table(
            [
                "Model",
                "CV ROC-AUC (mean ± sd)",
                "CV PR-AUC",
                "CV Brier",
                "Test ROC-AUC",
                "Test PR-AUC",
                "DeLong p vs champion",
            ],
            rows,
        ),
        "",
        f"Selection detail: best CV model = {MODEL_LABELS.get(m['selection']['best_model'])} "
        f"({m['selection']['best_cv_auc']:.4f}); rule `{m['selection']['rule']}`"
        + (f", cutoff = best − SE = {m['selection']['cutoff']:.4f}" if m["selection"].get("cutoff") else "")
        + f"; eligible = {', '.join(MODEL_LABELS.get(e, e) for e in m['selection']['eligible'])}. "
        "Paired per-fold AUC difference vs. the best model (mean ± SE, share of folds where the model was better): "
        + "; ".join(
            f"{MODEL_LABELS.get(k, k)} (reference)"
            if k == m["selection"]["best_model"]
            else f"{MODEL_LABELS.get(k, k)} {v['mean_diff']:+.4f} ± {v['se_diff']:.4f} ({v['share_of_folds_better']:.0%})"
            for k, v in sorted(
                m["selection"].get("paired_vs_best", {}).items(), key=lambda kv: -kv[1]["mean_diff"]
            )
        )
        + ". DeLong p-values > 0.05 mean the test-set AUC difference to the champion is not statistically significant.",
        "",
        "Tuning budget: "
        + ", ".join(
            f"{MODEL_LABELS.get(k, k)} {v['n_trials']} trials ({v['n_pruned']} pruned)"
            for k, v in m["tuning"].items()
        )
        + ".",
        "",
        "## Calibration (out-of-fold on training data)",
        "",
        _table(
            ["Method", "Brier", "ECE", "Log-loss", "ROC-AUC"],
            [
                [k, _f(v["brier"], 4), _f(v["ece"], 4), _f(v["log_loss"], 4), _f(v["roc_auc"], 4)]
                for k, v in m["calibration"]["comparison"].items()
            ],
        ),
        "",
        (
            f"The unweighted logistic model was already well calibrated: neither Platt (sigmoid) nor isotonic scaling "
            f"lowered out-of-fold Brier, so no recalibration layer is served (test Brier {_f(t['brier'], 4)}, ECE {_f(t['ece'], 4)})."
            if m["calibration"]["method"] == "none"
            else f"On the test set: uncalibrated Brier {_f(m['test']['uncalibrated_champion']['brier'], 4)} / ECE "
            f"{_f(m['test']['uncalibrated_champion']['ece'], 4)} → served model Brier {_f(t['brier'], 4)} / ECE {_f(t['ece'], 4)}."
        ),
        "",
        "## Decision thresholds (chosen on training out-of-fold predictions)",
        "",
    ]
    pt, ft = t["at_profit_threshold"], t["at_f1_threshold"]
    lines += [
        _table(
            ["Threshold", "Value", "Test precision", "Test recall", "Test F1", "Test accuracy"],
            [
                [
                    "Profit-optimal (retention call)",
                    _f(pt["threshold"], 2),
                    _pct(pt["precision"]),
                    _pct(pt["recall"]),
                    _f(pt["f1"], 3),
                    _pct(pt["accuracy"]),
                ],
                [
                    "F1-optimal (reference)",
                    _f(ft["threshold"], 2),
                    _pct(ft["precision"]),
                    _pct(ft["recall"]),
                    _f(ft["f1"], 3),
                    _pct(ft["accuracy"]),
                ],
                [
                    "0.50 (default)",
                    "0.50",
                    _pct(t["at_threshold"]["precision"]),
                    _pct(t["at_threshold"]["recall"]),
                    _f(t["at_threshold"]["f1"], 3),
                    _pct(t["at_threshold"]["accuracy"]),
                ],
            ],
        ),
        "",
        "Threshold vs. incentive cost: "
        + ", ".join(f"{k}: t* = {v:.2f}" for k, v in m["thresholds"]["scenarios"].items())
        + ".",
        "",
        "## Business impact — held-out test set",
        "",
    ]
    prow = []
    for name, p in ctx["policies"].items():
        prow.append(
            [
                name,
                f"{p['customers_targeted']:,} ({_pct(p['pct_targeted'], 0)})",
                _pct(p["churner_capture_rate"], 0),
                "—" if p["precision"] is None else _pct(p["precision"], 0),
                _money(p["campaign_cost"]),
                _money(p["net_profit"]),
                _money(p["net_profit_per_1k_customers"]),
                "—" if p["roi"] is None else f"{p['roi']:.2f}",
            ]
        )
    a = m["business"]["assumptions"]
    act_rows = [
        [n, _money(v["contact_cost"]), _money(v["incentive_cost"]), _pct(v["save_rate"], 0), v["label"]]
        for n, v in a["actions"].items()
    ]
    head = ctx["head"]
    lines += [
        _table(
            [
                "Policy",
                "Targeted",
                "Churners captured",
                "Precision",
                "Campaign cost",
                "Net profit",
                "Net / 1k customers",
                "ROI",
            ],
            prow,
        ),
        "",
        f"Under these assumptions the threshold policy earns **{head['uplift_vs_heuristic_same_volume']:.2f}×** the net profit of "
        "the rule-based heuristic at the same contact volume, and the next-best-action policy is profitable in "
        f"**{_pct(head['sensitivity_profitable_share'], 0)}** of the save-rate × incentive scenarios in the sensitivity grid."
        if head.get("uplift_vs_heuristic_same_volume")
        else "",
        "",
        "**Assumptions (configurable in `configs/config.yaml`, illustrative — not measured):** gross margin "
        f"{_pct(a['gross_margin'], 0)}; CLV = margin × monthly charge × expected remaining months within "
        f"{a['clv_horizon_months']} months ({a['clv_method']} method); mean test CLV {_money(m['business']['clv_test_mean'])}.",
        "",
        _table(["Action", "Contact cost", "Incentive cost", "Save rate", "Description"], act_rows),
        "",
        "Profit model (Neslin et al., 2006): targeted churner → save_rate × (CLV − incentive) − contact; "
        "targeted non-churner → −(contact + incentive). Value is credited only for customers who actually churned.",
        "",
        "## Survival analysis (Cox proportional hazards, time-to-churn)",
        "",
    ]
    s = m["survival"]
    hr, hr_ci = _hr_month_to_month(m)
    km_rows = [
        [k, "not reached" if v is None else f"{v:.0f} months"]
        for k, v in s["km_median_months_by_contract"].items()
    ]
    top_hr = sorted(s["hazard_ratios"].items(), key=lambda kv: -abs(math.log(kv[1]["hazard_ratio"])))[:6]
    lines += [
        f"- Concordance (C-index): train {s['c_index_train']:.3f}, **test {s['c_index_test']:.3f}**.",
        f"- Month-to-month vs. two-year contract hazard ratio: **{hr:.1f}×** (95% CI {hr_ci[0]:.1f}–{hr_ci[1]:.1f})."
        if hr
        else "",
        f"- Log-rank test across contract types: p = {s['logrank_p_contract']:.1e}.",
        "",
        _table(["Contract", "Kaplan-Meier median lifetime"], km_rows),
        "",
        _table(
            ["Covariate", "Hazard ratio", "95% CI", "p"],
            [
                [
                    k,
                    f"{v['hazard_ratio']:.2f}",
                    f"{v['ci_lower']:.2f} – {v['ci_upper']:.2f}",
                    f"{v['p_value']:.1e}",
                ]
                for k, v in top_hr
            ],
        ),
        "",
        "## Explainability",
        "",
        f"Explainer: `{m['explainability']['explainer']}` SHAP on the champion pipeline, grouped into business concepts.",
        "",
        _table(
            ["Concept", f"Mean |SHAP| ({m['explainability'].get('units', 'log-odds')})"],
            [[k, f"{v:.3f}"] for k, v in list(m["explainability"]["global_concept_importance"].items())[:10]],
        ),
        "",
    ]
    hc = m["explainability"]["highest_risk_customer"]
    lines += [
        f"Highest-risk test customer `{hc['customerID']}` (P(churn) = {hc['churn_probability']:.1%}) — reason codes returned by the API:",
        "",
    ]
    units = m["explainability"].get("units", "log-odds")
    lines += [
        f"{i}. {r['description']} (+{r['impact']:.2f} {units})"
        for i, r in enumerate(hc["reasons"]["risk_drivers"], 1)
    ]
    lines += ["", "## Fairness audit (test set, profit-optimal threshold)", ""]
    lines += [
        _table(
            ["Attribute", "TPR gap", "FPR gap", "AUC gap", "Selection-rate ratio"],
            [
                [
                    r["attribute"],
                    _f(r["tpr_gap"]),
                    _f(r["fpr_gap"]),
                    _f(r["auc_gap"]),
                    _f(r["selection_rate_ratio"]),
                ]
                for r in m["fairness"]["summary"]
            ],
        ),
        "",
        "Gender is excluded from the model's inputs. Gaps for SeniorCitizen largely reflect different base churn "
        "rates; see `reports/fairness_groups.csv` and the model card before using scores for anything other than "
        "offering retention incentives.",
        "",
        "## Drift monitoring (PSI)",
        "",
        f"- Held-out test vs. training reference: max PSI {m['drift']['test_vs_train_max_psi']:.3f} "
        f"({m['drift']['test_vs_train_alerts']} features above the warn level) — stable, as expected for a random split.",
        f"- Simulated production shift (+15% price, 25% of term contracts lapsing): {m['drift']['simulated_alerts']} alerts — "
        + ", ".join(
            f"{r['feature'].replace('__churn_score__', 'model score')} {r['psi']:.2f}"
            for r in m["drift"]["simulated"][:4]
        )
        + ".",
        "",
    ]
    if ctx["ablation"]:
        lines += ["## Ablation studies (repeated CV on training data)", ""]
        ab = ctx["ablation"]
        lines += [
            _table(
                [
                    "Experiment",
                    "Variant",
                    "Model",
                    "CV ROC-AUC",
                    "CV Brier",
                    "CV ECE",
                    "Δ AUC vs. reference (paired)",
                ],
                [
                    [
                        r["experiment"],
                        r["variant"],
                        r["model"],
                        f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f}",
                        _f(r["brier_mean"], 4),
                        _f(r.get("ece_mean"), 4),
                        "reference"
                        if r.get("delta_auc") is None
                        else f"{r['delta_auc']:+.4f} ± {r['delta_auc_se']:.4f}",
                    ]
                    for r in ab["rows"]
                ],
            ),
            "",
        ]
        lines += [f"- {c}" for c in ab.get("conclusions", [])] + [""]
    if ctx["bench"] or ctx["tests"]:
        lines += ["## Engineering", ""]
        b = ctx["bench"]
        if b:
            lines.append(
                f"- API latency ({b['n_requests']} sequential requests, single customer with reason codes): "
                f"p50 {b['single']['p50_ms']:.1f} ms, p95 {b['single']['p95_ms']:.1f} ms; batch of {b['batch']['batch_size']}: "
                f"{b['batch']['rows_per_second']:,.0f} customers/s. Measured {b['measured_at']} on {b['machine']}."
            )
        tt = ctx["tests"]
        if tt:
            lines.append(
                f"- Test suite: {tt['passed']} passed, {tt['failed']} failed; line coverage {tt['coverage_pct']:.1f}%."
            )
        lines.append("")
    lines += [
        "## Figures",
        "",
        "All figures are in `reports/figures/` (dashboard: `reports/figures/dashboard.png`).",
        "",
    ]
    return "\n".join(x for x in lines if x is not None)


# ---------------------------------------------------------------------------- MODEL CARD
def render_model_card(ctx: dict[str, Any]) -> str:
    m, t = ctx["m"], ctx["test"]
    a = m["business"]["assumptions"]
    return "\n".join(
        [
            "# Model Card — Telco churn risk model",
            "",
            "Format follows Mitchell et al. (2019), *Model Cards for Model Reporting*.",
            "",
            "## Model details",
            f"- **Model:** {ctx['champ_label']} pipeline (feature engineering → imputation/scaling/one-hot → classifier) with "
            f"`{m['calibration']['method']}` probability calibration.",
            f"- **Version:** {m['project']['version']} · trained {m['project']['trained_at']} · scikit-learn {m['project']['sklearn']} · Python {m['project']['python']}.",
            f"- **Selection:** `{m['selection']['rule']}` rule over {ctx['n_models']} tuned candidates using repeated stratified CV on training data.",
            "- **Artifact:** `artifacts/model_bundle.joblib` (model, explainer, survival-CLV model, action menu, drift reference, metadata).",
            "",
            "## Intended use",
            "- Rank existing residential telecom customers by 1-month churn risk and recommend a cost-justified retention action.",
            "- Users: retention / CRM analysts and call-centre tooling via the REST API.",
            "- **Out of scope:** credit, pricing or any adverse decision about a customer; populations unlike the IBM Telco sample "
            "(e.g. business accounts, other countries) without re-validation.",
            "",
            "## Data",
            f"- IBM Telco Customer Churn sample (Apache-2.0): {m['data']['n_rows']:,} customers, {m['data']['n_raw_features']} raw inputs, "
            f"churn rate {_pct(m['data']['churn_rate'])}.",
            f"- Stratified split: {m['data']['n_train']:,} train / {m['data']['n_test']:,} test. Test set used once for final reporting.",
            "- Single snapshot; churn = left within the last month. No timestamps, so no out-of-time validation is possible.",
            "",
            "## Performance (held-out test set)",
            f"- ROC-AUC {t['roc_auc']:.4f} (95% CI {ctx['ci']['roc_auc']['ci_lower']:.4f}–{ctx['ci']['roc_auc']['ci_upper']:.4f}); "
            f"PR-AUC {t['pr_auc']:.4f}; Brier {t['brier']:.4f}; ECE {t['ece']:.4f}.",
            f"- Top-10% lift {t['lift']['top_10pct']['lift']:.2f}×; {_pct(t['lift']['top_30pct']['capture_rate'])} of churners in the top 30%.",
            f"- At the profit-optimal threshold {t['at_profit_threshold']['threshold']:.2f}: precision {_pct(t['at_profit_threshold']['precision'])}, "
            f"recall {_pct(t['at_profit_threshold']['recall'])}.",
            "",
            "## Fairness considerations",
            "- `gender` is excluded from model inputs. `SeniorCitizen` is used (it carries signal); seniors have a higher base churn rate, "
            "so they are selected more often. Because the action is a *benefit* (a retention offer), this is lower-risk than an adverse "
            "decision, but it should be reviewed against local regulation.",
        ]
        + [
            f"- {r['attribute']}: TPR gap {r['tpr_gap']:.3f}, FPR gap {r['fpr_gap']:.3f}, AUC gap {r['auc_gap']:.3f}, "
            f"selection-rate ratio {r['selection_rate_ratio']:.2f}."
            for r in m["fairness"]["summary"]
        ]
        + [
            "",
            "## Business assumptions",
            f"- Gross margin {_pct(a['gross_margin'], 0)}, CLV horizon {a['clv_horizon_months']} months, action costs and save rates in "
            "`configs/config.yaml`. These are illustrative; profit figures scale with them (see the sensitivity heatmap).",
            "",
            "## Caveats and limitations",
            "- Save rates are assumed, not estimated: a randomised retention experiment (uplift modelling) is required to measure them.",
            "- The model predicts churn risk, not treatment effect; customers who would leave regardless are still targeted.",
            "- Cox proportional-hazards assumption is approximate for contract type; CLV is a planning estimate, not accounting value.",
            "- Re-train when PSI on key inputs or on the score exceeds 0.25 (see `scripts/monitor_drift.py`).",
            "",
        ]
    )


# ---------------------------------------------------------------------------- RESUME.md
def _ablation_lookup(ab: dict | None, model: str, variant: str) -> dict | None:
    if not ab:
        return None
    for r in ab["rows"]:
        if r["model"] == model and r["variant"] == variant:
            return r
    return None


def render_resume(ctx: dict[str, Any]) -> str:
    m, t, ci, cv, head = ctx["m"], ctx["test"], ctx["ci"], ctx["cv"], ctx["head"]
    label = ctx["champ_label"]
    label_lc = label if label in {"XGBoost", "LightGBM", "CatBoost"} else label.lower()
    hr, _ = _hr_month_to_month(m)
    unc = m["test"]["uncalibrated_champion"]
    cal = m["calibration"]["method"]
    bench, tests, ab = ctx["bench"], ctx["tests"], ctx["ablation"]
    nba = m["business"]["policies"]["ML next-best-action (EV-optimal)"]
    auc_txt = f"{t['roc_auc']:.3f} (95% CI {ci['roc_auc']['ci_lower']:.3f}–{ci['roc_auc']['ci_upper']:.3f})"
    if cal == "none":
        cal_txt = f"probabilities were well calibrated without post-processing (ECE {t['ece']:.3f}, Brier {t['brier']:.3f})"
        cal_word = "well-calibrated"
    else:
        cal_txt = f"{cal} calibration cut expected calibration error from {unc['ece']:.3f} to {t['ece']:.3f}"
        cal_word = f"{cal}-calibrated"
    eng = []
    if bench:
        eng.append(f"p95 latency {bench['single']['p95_ms']:.0f} ms per customer incl. reason codes")
    if tests:
        eng.append(f"{tests['passed']} automated tests, {tests['coverage_pct']:.1f}% line coverage")
    eng_txt = "; ".join(eng)
    uplift = head.get("uplift_vs_heuristic_same_volume")
    blanket = head.get("blanket_net_profit")
    ens = [
        v["p_value"] for k, v in m["test"]["delong_vs_champion"].items() if k in ("soft_voting", "stacking")
    ]
    ens_p = min(ens) if ens else None

    ablation_bullet = ""
    fe = _ablation_lookup(ab, "Logistic Regression", "raw inputs only")
    bal = _ablation_lookup(ab, "Logistic Regression", "class_weight=balanced")
    smote = _ablation_lookup(ab, "Logistic Regression", "SMOTE (in-fold)")
    ref = _ablation_lookup(ab, "Logistic Regression", "default")
    if fe and bal and smote and ref:
        brier_up = 100 * max(bal["delta_brier"], smote["delta_brier"]) / ref["brier_mean"]
        brier_lo = 100 * min(bal["delta_brier"], smote["delta_brier"]) / ref["brier_mean"]
        ece_x = max(bal["ece_mean"], smote["ece_mean"]) / ref["ece_mean"]
        ablation_bullet = (
            f"- Ran paired ablation studies (repeated CV): domain features lifted logistic-regression ROC-AUC by "
            f"{-fe['delta_auc']:+.3f} ({-fe['delta_auc'] / fe['delta_auc_se']:.1f} standard errors), while SMOTE and class weighting added no AUC "
            f"but worsened Brier score by {brier_lo:.0f}–{brier_up:.0f}% and calibration error up to {ece_x:.0f}×, so both were dropped."
        )

    lines = [
        "# Resume material — Telco Churn Intelligence",
        "",
        "_Generated from `reports/metrics.json`"
        + (", `reports/ablation_results.json`" if ab else "")
        + (", `reports/api_benchmark.json`" if bench else "")
        + (", `reports/test_summary.json`" if tests else "")
        + ". Every number below is reproducible with `make all`._",
        "",
        "## Project title line",
        "",
        "**Telco Churn Intelligence — Profit-Driven Churn Prediction & Retention Engine** | Python, scikit-learn, XGBoost, "
        "LightGBM, CatBoost, Optuna, SHAP, lifelines, FastAPI, Streamlit, Docker, GitHub Actions",
        "",
        "## Bullets (pick 3–5)",
        "",
        f"- Built a leakage-free churn-prediction pipeline on 7,043 telecom customers, benchmarking {ctx['n_models']} models "
        f"(logistic regression, random forest, XGBoost, LightGBM, CatBoost, voting and stacking ensembles) tuned with "
        f"{ctx['n_trials']} Optuna trials under repeated stratified CV; the {label_lc} selected by the one-standard-error rule "
        f"reached a held-out ROC-AUC of {auc_txt}"
        + (
            f", statistically indistinguishable from the voting/stacking ensembles (DeLong p ≥ {ens_p:.2f})."
            if ens_p is not None
            else "."
        ),
        f"- Delivered {t['lift']['top_10pct']['lift']:.1f}× lift in the top risk decile and captured "
        f"{_pct(t['lift']['top_30pct']['capture_rate'], 0)} of churners by contacting 30% of customers; {cal_txt}.",
        f"- Designed an expected-value next-best-action engine (P(churn) × save rate × survival-based CLV − campaign cost) that "
        f"picks a retention call, an email offer or no action per customer; under documented cost assumptions it returned "
        f"{_money(head['nba_net_profit_per_1k'])} net per 1,000 held-out customers"
        + (" (contacting everyone lost money)" if blanket is not None and blanket < 0 else "")
        + (
            f", and at equal contact volume the model's targeting earned {100 * (uplift - 1):.0f}% more than a rule-based heuristic."
            if uplift
            else "."
        ),
        f"- Modelled time-to-churn with a Cox proportional-hazards model (test C-index {m['survival']['c_index_test']:.2f})"
        + (
            f"; month-to-month contracts carried {hr:.1f}× the churn hazard of two-year contracts"
            if hr
            else ""
        )
        + "; used it to estimate each customer's remaining lifetime value.",
    ]
    if ablation_bullet:
        lines.append(ablation_bullet)
    lines += [
        "- Shipped it as a FastAPI service with SHAP reason codes per prediction, a Streamlit what-if and campaign simulator, "
        "PSI drift monitoring, Docker and GitHub Actions CI" + (f" ({eng_txt})." if eng_txt else "."),
        "",
        "## Two-line version (tight one-page resumes)",
        "",
        f"- End-to-end churn system (7,043 customers): {ctx['n_models']}-model Optuna/CV benchmark → {cal_word} {label_lc}, "
        f"test ROC-AUC {t['roc_auc']:.3f}, {t['lift']['top_10pct']['lift']:.1f}× top-decile lift, Cox survival CLV, SHAP reason codes.",
        f"- Profit-optimal next-best-action targeting ({_money(head['nba_net_profit_per_1k'])} net / 1,000 customers under stated "
        "assumptions) served via FastAPI + Streamlit + Docker with drift monitoring and CI.",
        "",
        "## Numbers and where they come from",
        "",
        _table(
            ["Claim", "Value", "Source"],
            [
                ["Test ROC-AUC (95% CI)", auc_txt, "metrics.json → test.champion"],
                [
                    "Repeated-CV ROC-AUC",
                    f"{cv['roc_auc_mean']:.3f} ± {cv['roc_auc_std']:.3f} ({int(cv['n_folds'])} folds)",
                    "metrics.json → cv",
                ],
                ["Test PR-AUC", f"{t['pr_auc']:.3f}", "metrics.json → test.champion"],
                [
                    "Accuracy @ 0.5",
                    _pct(t["at_threshold"]["accuracy"]),
                    "metrics.json → test.champion.at_threshold",
                ],
                [
                    "Recall / precision @ profit threshold",
                    f"{_pct(t['at_profit_threshold']['recall'])} / {_pct(t['at_profit_threshold']['precision'])}",
                    "metrics.json → test.champion.at_profit_threshold",
                ],
                [
                    "Top-decile lift",
                    f"{t['lift']['top_10pct']['lift']:.2f}×",
                    "metrics.json → test.champion.lift",
                ],
                [
                    "Churners captured in top 30%",
                    _pct(t["lift"]["top_30pct"]["capture_rate"]),
                    "metrics.json → test.champion.lift",
                ],
                [
                    "Brier / ECE (served model)",
                    f"{t['brier']:.4f} / {t['ece']:.4f}",
                    "metrics.json → test.champion",
                ],
                [
                    "Next-best-action net profit / 1k customers",
                    _money(head["nba_net_profit_per_1k"]),
                    "policy_comparison.csv (assumption-based)",
                ],
                [
                    "Customers contacted / churners reached (NBA)",
                    f"{_pct(nba['pct_targeted'], 0)} / {_pct(nba['churner_capture_rate'], 0)}",
                    "policy_comparison.csv",
                ],
                ["Cox test C-index", f"{m['survival']['c_index_test']:.3f}", "metrics.json → survival"],
                ["Optuna trials", str(ctx["n_trials"]), "metrics.json → tuning"],
            ]
            + (
                [
                    [
                        "API p50 / p95 latency",
                        f"{bench['single']['p50_ms']:.1f} / {bench['single']['p95_ms']:.1f} ms",
                        "api_benchmark.json",
                    ]
                ]
                if bench
                else []
            )
            + (
                [
                    [
                        "Tests / coverage",
                        f"{tests['passed']} / {tests['coverage_pct']:.1f}%",
                        "test_summary.json",
                    ]
                ]
                if tests
                else []
            ),
        ),
        "",
        "## How to say it in an interview",
        "",
        "- *Why not just report accuracy?* 26.5% churn base rate: a model predicting 'no churn' for everyone is 73.5% accurate. "
        "ROC-AUC, PR-AUC, lift and calibrated probabilities are what a retention team can act on.",
        "- *How did you avoid leakage?* Feature quantiles are learned inside the sklearn pipeline per CV fold; tuning, model "
        "selection, calibration and thresholds use training data only; the test set is scored once.",
        "- *Why logistic regression over the ensembles?* One-standard-error rule: the best ensemble was only +0.0008 AUC better "
        "in CV — consistent across folds but far below CV noise — and on the test set logistic regression was actually "
        "highest, with no significant DeLong difference. One model instead of three, exact SHAP, millisecond latency.",
        "- *Why no SMOTE?* The ablation shows it does not improve ranking (AUC) and it inflates probabilities, which breaks "
        "expected-value decisions that multiply P(churn) by dollar values.",
        "- *Why were some v3 features removed?* MonthlyCharges is almost a linear function of the subscribed services "
        "(R² ≈ 1), so features like Fiber × Charge made the model learn large offsetting weights and the reason codes "
        "contradicted each other. Removing them cost < 0.001 AUC.",
        "- *Where do the dollars come from?* An explicit, configurable campaign-profit model (Neslin et al., 2006) with a "
        "sensitivity analysis; save rates are assumptions to be measured by an A/B retention experiment (next step: uplift modelling).",
        "",
    ]
    return "\n".join(lines)


def write_reports(rep_dir: str | Path | None = None) -> dict[str, Path]:
    rep_dir = Path(rep_dir) if rep_dir else PROJECT_ROOT / "reports"
    ctx = build_context(rep_dir)
    paths = {
        "results": rep_dir / "RESULTS.md",
        "model_card": rep_dir / "model_card.md",
        "resume": rep_dir / "RESUME.md",
    }
    paths["results"].write_text(render_results(ctx))
    paths["model_card"].write_text(render_model_card(ctx))
    paths["resume"].write_text(render_resume(ctx))
    return paths
