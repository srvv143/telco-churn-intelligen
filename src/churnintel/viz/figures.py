"""Assemble every report figure plus the composite dashboard."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from . import plots as P
from .style import BLUE, INK, INK2, ORANGE

logger = logging.getLogger(__name__)

_POLICY_SHORT = {
    "Blanket campaign (contact everyone)": "Contact everyone",
    "Rule-based heuristic (same volume)": "Rule-based heuristic (same volume)",
    "Random targeting (same volume, expected)": "Random (same volume)",
    "ML next-best-action (EV-optimal)": "ML next-best-action",
    "Oracle (perfect foresight, reference)": "Oracle (upper bound)",
}


def _short_policies(policies: pd.DataFrame) -> pd.DataFrame:
    d = policies.drop(index=[i for i in policies.index if i == "No campaign"])
    d = d.rename(index=lambda i: _POLICY_SHORT.get(i, i.replace("policy ", "").replace(">=", "≥")))
    return d


def make_all_figures(fig_dir: str | Path, **k) -> list[Path]:
    fig_dir = Path(fig_dir)
    out: list[Path] = []
    y = k["y_test"]
    champ = k["champion_label"]
    summary = pd.DataFrame(k["cv_summary"]).T
    cutoff = k["selection"].get("cutoff")

    scores = dict(k["model_scores"])
    scores["Rule-based heuristic"] = k["heuristic"]
    roc_curves = {lbl: (*roc_curve(y, s)[:2], roc_auc_score(y, s)) for lbl, s in scores.items()}
    pr_curves = {}
    for lbl, s in scores.items():
        prec, rec, _ = precision_recall_curve(y, s)
        pr_curves[lbl] = (rec, prec, average_precision_score(y, s))
    highlight = {champ: BLUE, "Rule-based heuristic": ORANGE}
    cal_probs = (
        {"Uncalibrated": k["p_test_uncal"], f"Calibrated ({k['cal_method']})": k["p_test"]}
        if k["cal_method"] != "none"
        else {"Served model (no recalibration needed)": k["p_test"]}
    )
    frac, prof = k["profit_curves"]["Champion model ranking"]
    opt = (float(frac[int(np.argmax(prof))]), float(np.max(prof)))
    policies = _short_policies(k["policies"])

    def single(name, drawer, size=(7.2, 4.6)):
        fig, ax = P.new_figure(*size)
        drawer(fig, ax)
        out.append(P.save_figure(fig, fig_dir / name))

    single(
        "01_model_selection_cv.png", lambda f, a: P.draw_cv_comparison(a, summary, champ, cutoff), (7.6, 4.4)
    )
    single("02_roc_test.png", lambda f, a: P.draw_roc(a, roc_curves, highlight), (6.4, 5.6))
    single(
        "03_precision_recall_test.png",
        lambda f, a: P.draw_pr(a, pr_curves, highlight, float(np.mean(y))),
        (6.4, 5.6),
    )
    single("04_calibration.png", lambda f, a: P.draw_calibration(a, y, cal_probs), (6.4, 5.6))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    P.draw_gains(axes[0], k["gains"])
    P.draw_lift(axes[1], k["gains"])
    out.append(P.save_figure(fig, fig_dir / "05_gains_and_lift.png"))

    single("06_profit_curve.png", lambda f, a: P.draw_profit_curves(a, k["profit_curves"], opt))
    single(
        "07_threshold_scenarios.png",
        lambda f, a: P.draw_threshold_scenarios(
            a, k["threshold_grid"], k["scenario_profits"], k["scenario_thresholds"]
        ),
    )
    single(
        "08_profit_sensitivity.png",
        lambda f, a: P.draw_sensitivity(a, k["sensitivity"], k["base_action"]),
        (8.2, 5.2),
    )
    single("09_policy_comparison.png", lambda f, a: P.draw_policies(a, policies, ("ML",)), (8.4, 4.4))

    if k["explainer_kind"] != "agnostic" and k["shap_X"] is not None:
        single(
            "10_shap_summary.png",
            lambda f, a: P.draw_beeswarm(
                f, a, k["shap_values"], k["shap_X"], k["feature_names"], units=k.get("shap_units", "log-odds")
            ),
            (8.0, 6.4),
        )
    single(
        "11_churn_drivers_by_concept.png",
        lambda f, a: P.draw_concept_importance(a, k["concept_importance"], k.get("shap_units", "log-odds")),
        (7.2, 4.8),
    )
    single(
        "12_local_explanation.png",
        lambda f, a: P.draw_local_explanation(
            a, k["local_contrib"], k["local_prob"], k["local_id"], k.get("shap_units", "log-odds")
        ),
        (8.4, 4.6),
    )
    single("13_kaplan_meier_contract.png", lambda f, a: P.draw_km(a, k["km"]), (8.0, 4.8))
    single("14_cox_hazard_ratios.png", lambda f, a: P.draw_hazard_ratios(a, k["hazard_ratios"]), (7.6, 6.2))

    attrs = k["fairness_attrs"]
    if attrs:
        fig, axes = plt.subplots(1, len(attrs), figsize=(6.4 * len(attrs), 4.4), squeeze=False)
        for ax, attr in zip(axes[0], attrs, strict=False):
            P.draw_fairness(ax, k["fairness"], attr)
        out.append(P.save_figure(fig, fig_dir / "15_fairness_audit.png"))

    warn, alert = k["psi_levels"]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8))
    P.draw_drift(axes[0], k["psi_test"], "Drift: held-out test vs. training", warn, alert)
    P.draw_drift(axes[1], k["psi_sim"], "Drift: simulated +15% price & contract lapse", warn, alert)
    out.append(P.save_figure(fig, fig_dir / "16_drift_monitoring.png"))

    single(
        "17_confusion_matrix.png",
        lambda f, a: P.draw_confusion(a, np.asarray(k["confusion"]), k["t_star"]),
        (5.6, 4.6),
    )

    # ------------------------------------------------------------------ dashboard
    fig = plt.figure(figsize=(22, 16.5))
    gs = GridSpec(3, 3, figure=fig, hspace=0.42, wspace=0.42, top=0.92, bottom=0.05, left=0.07, right=0.97)
    P.draw_cv_comparison(fig.add_subplot(gs[0, 0]), summary, champ, cutoff)
    P.draw_roc(fig.add_subplot(gs[0, 1]), roc_curves, highlight)
    P.draw_calibration(fig.add_subplot(gs[0, 2]), y, cal_probs)
    P.draw_gains(fig.add_subplot(gs[1, 0]), k["gains"])
    P.draw_profit_curves(fig.add_subplot(gs[1, 1]), k["profit_curves"], opt)
    P.draw_policies(fig.add_subplot(gs[1, 2]), policies, ("ML",))
    P.draw_concept_importance(
        fig.add_subplot(gs[2, 0]), k["concept_importance"], k.get("shap_units", "log-odds")
    )
    P.draw_km(fig.add_subplot(gs[2, 1]), k["km"])
    P.draw_sensitivity(fig.add_subplot(gs[2, 2]), k["sensitivity"], k["base_action"])
    h = k["headline"]
    fig.suptitle(
        "Telco Churn Intelligence — leakage-free, calibrated, profit-driven churn modelling",
        x=0.07,
        y=0.985,
        ha="left",
        fontsize=19,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.07,
        0.955,
        f"Champion: {champ}  ·  held-out test ROC-AUC {h['auc']:.3f} (95% CI {h['auc_ci'][0]:.3f}–{h['auc_ci'][1]:.3f})"
        f"  ·  repeated-CV ROC-AUC {h['cv_auc']:.3f} ± {h['cv_std']:.3f}  ·  IBM Telco, 7,043 customers",
        ha="left",
        fontsize=12.5,
        color=INK2,
    )
    out.append(P.save_figure(fig, fig_dir / "dashboard.png"))
    logger.info("Wrote %d figures to %s", len(out), fig_dir)
    return out
