"""Report figures.

Each ``draw_*`` function draws on a provided Axes (so panels can be composed into the dashboard);
``save_figure`` writes a PNG. Conventions: champion emphasised in blue with other models in gray,
red = increases churn risk / blue = reduces it, solid hairline grids, direct labels where useful.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

from ..evaluation.metrics import expected_calibration_error
from .style import (
    AQUA,
    AXIS,
    BLUE,
    DEEMPH,
    DIVERGING,
    INK,
    INK2,
    MUTED,
    NEGATIVE,
    ORANGE,
    POSITIVE,
    SEQUENTIAL,
    SERIES,
    STATUS,
    SURFACE,
    apply_style,
)

apply_style()


def save_figure(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def new_figure(width: float = 7.2, height: float = 4.6):
    return plt.subplots(figsize=(width, height))


def _money(axis, unit: str = "k") -> None:
    if unit == "k":
        axis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"-${abs(v) / 1000:,.0f}k" if v < 0 else f"${v / 1000:,.0f}k")
        )
    else:
        axis.set_major_formatter(
            mticker.FuncFormatter(lambda v, _: f"-${abs(v):,.0f}" if v < 0 else f"${v:,.0f}")
        )


def _pct(axis) -> None:
    axis.set_major_formatter(mticker.PercentFormatter(1.0, decimals=0))


# ---------------------------------------------------------------------------- model comparison
def draw_cv_comparison(ax, summary: pd.DataFrame, champion: str, cutoff: float | None = None):
    """Dot plot of mean CV ROC-AUC with +/-1 SD whiskers (dot plots need no zero baseline)."""
    d = summary.sort_values("roc_auc_mean")
    span = d["roc_auc_std"].max()
    for i, (label, row) in enumerate(d.iterrows()):
        color = BLUE if label == champion else MUTED
        lo, hi = row["roc_auc_mean"] - row["roc_auc_std"], row["roc_auc_mean"] + row["roc_auc_std"]
        ax.plot([lo, hi], [i, i], color=color, lw=2, solid_capstyle="round", zorder=2)
        ax.scatter(row["roc_auc_mean"], i, s=70, color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
        weight = "bold" if label == champion else "normal"
        ax.text(
            hi + span * 0.15,
            i,
            f"{row['roc_auc_mean']:.4f}",
            va="center",
            fontsize=8.5,
            color=INK2,
            weight=weight,
        )
    ax.set_yticks(range(len(d)), [f"{lbl}  (champion)" if lbl == champion else lbl for lbl in d.index])
    if cutoff is not None:
        ax.axvline(cutoff, color=MUTED, lw=1)
        ax.text(cutoff, len(d) - 0.45, "  1-SE cutoff", fontsize=8, color=INK2, va="bottom")
    lo = (d["roc_auc_mean"] - d["roc_auc_std"]).min()
    hi = (d["roc_auc_mean"] + d["roc_auc_std"]).max()
    ax.set_xlim(lo - 0.003, hi + (hi - lo) * 0.35)
    ax.set_ylim(-0.6, len(d) - 0.1)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("ROC-AUC, mean ± 1 SD across repeated stratified CV folds (train only)")
    ax.set_title("Model selection by cross-validation")
    return ax


def draw_roc(ax, curves: dict, highlight: dict[str, str]):
    for label, (fpr, tpr, auc) in curves.items():
        if label not in highlight:
            ax.plot(fpr, tpr, color=DEEMPH, lw=1.1, label=f"{label} ({auc:.3f})")
    for label, color in highlight.items():
        if label in curves:
            fpr, tpr, auc = curves[label]
            ax.plot(fpr, tpr, color=color, lw=2.2, label=f"{label} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8)
    ax.text(0.62, 0.56, "random", color=MUTED, fontsize=8, rotation=33)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=7.8, title="Held-out test ROC-AUC", title_fontsize=8)
    ax.set_title("ROC curves on the held-out test set")
    return ax


def draw_pr(ax, curves: dict, highlight: dict[str, str], base_rate: float):
    for label, (rec, prec, ap) in curves.items():
        if label not in highlight:
            ax.plot(rec, prec, color=DEEMPH, lw=1.1, label=f"{label} ({ap:.3f})")
    for label, color in highlight.items():
        if label in curves:
            rec, prec, ap = curves[label]
            ax.plot(rec, prec, color=color, lw=2.2, label=f"{label} ({ap:.3f})")
    ax.axhline(base_rate, color=MUTED, lw=0.8)
    ax.text(0.02, base_rate + 0.015, f"churn base rate {base_rate:.1%}", color=MUTED, fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="upper right", fontsize=7.8, title="Average precision", title_fontsize=8)
    ax.set_title("Precision-recall on the held-out test set")
    return ax


def draw_calibration(ax, y, probs: dict[str, np.ndarray], n_bins: int = 10):
    # the served model is always the last entry and is drawn in the champion's blue
    colors = [BLUE] if len(probs) == 1 else [ORANGE, BLUE, AQUA]
    for (label, p), color in zip(probs.items(), colors, strict=False):
        frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        brier = brier_score_loss(y, p)
        ece = expected_calibration_error(np.asarray(y), np.asarray(p))
        ax.plot(
            mean, frac, "-o", color=color, ms=4.5, lw=1.8, label=f"{label}: Brier {brier:.4f} · ECE {ece:.3f}"
        )
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.8)
    ax.text(0.68, 0.62, "perfect", color=MUTED, fontsize=8, rotation=33)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted churn probability (decile bins)")
    ax.set_ylabel("Observed churn rate")
    ax.legend(loc="upper left", fontsize=7.8)
    ax.set_title("Probability calibration (test set)")
    return ax


# ---------------------------------------------------------------------------- targeting
def draw_gains(ax, gains: pd.DataFrame):
    x = np.concatenate([[0], gains["cum_customers_pct"].to_numpy()])
    yv = np.concatenate([[0], gains["cum_capture"].to_numpy()])
    ax.plot(x, yv, "-o", color=BLUE, ms=4, label="Champion model")
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, label="Random targeting")
    for k in (2, 3):
        ax.annotate(
            f"top {x[k]:.0%} → {yv[k]:.0%} of churners",
            (x[k], yv[k]),
            xytext=(12, -16),
            textcoords="offset points",
            fontsize=8,
            color=INK2,
        )
    _pct(ax.xaxis)
    _pct(ax.yaxis)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Customers contacted (ranked by churn risk)")
    ax.set_ylabel("Churners captured")
    ax.legend(loc="lower right")
    ax.set_title("Cumulative gains")
    return ax


def draw_lift(ax, gains: pd.DataFrame):
    ax.bar(gains["decile"], gains["lift"], color=BLUE, width=0.8, edgecolor=SURFACE, linewidth=2)
    ax.axhline(1.0, color=MUTED, lw=0.9)
    first = gains.iloc[0]
    ax.text(
        first["decile"], first["lift"] + 0.06, f"{first['lift']:.2f}×", ha="center", fontsize=8.5, color=INK2
    )
    ax.set_xticks(gains["decile"])
    ax.set_xlabel("Risk decile (1 = highest predicted risk)")
    ax.set_ylabel("Lift over base churn rate")
    ax.grid(axis="x", visible=False)
    ax.set_title("Lift by decile")
    return ax


def draw_profit_curves(ax, curves: dict, optimum: tuple[float, float] | None = None):
    colors = {0: BLUE, 1: ORANGE}
    for i, (label, (frac, profit)) in enumerate(curves.items()):
        color = colors.get(i, MUTED)
        ax.plot(frac, profit, color=color, lw=2.2 if i == 0 else 1.6, label=label)
    if optimum is not None:
        f, v = optimum
        ax.scatter([f], [v], s=60, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=4)
        ax.annotate(
            f"max ${v:,.0f} at top {f:.0%}",
            (f, v),
            xytext=(10, 8),
            textcoords="offset points",
            fontsize=8.5,
            color=INK,
        )
    ax.axhline(0, color=AXIS, lw=0.9)
    _pct(ax.xaxis)
    _money(ax.yaxis)
    ax.set_xlabel("Customers contacted (ranked by each strategy)")
    ax.set_ylabel("Realised net profit, test set")
    ax.legend(loc="upper right")
    ax.set_title("Profit curve: whom to contact")
    return ax


def draw_threshold_scenarios(ax, grid, profits: dict[str, np.ndarray], thresholds: dict[str, float]):
    for (label, prof), color in zip(profits.items(), [BLUE, ORANGE, AQUA], strict=False):
        ax.plot(grid, prof, color=color, lw=1.9, label=f"{label} (t* = {thresholds[label]:.2f})")
        t = thresholds[label]
        k = int(np.argmin(np.abs(np.asarray(grid) - t)))
        ax.scatter([t], [prof[k]], s=45, color=color, edgecolor=SURFACE, linewidth=2, zorder=4)
    ax.axhline(0, color=AXIS, lw=0.9)
    _money(ax.yaxis)
    ax.set_xlabel("Decision threshold on calibrated P(churn)")
    ax.set_ylabel("Net profit (train, out-of-fold)")
    ax.legend(loc="upper right", title="Incentive-cost scenario", title_fontsize=8)
    ax.set_title("Profit-optimal threshold moves with campaign cost")
    return ax


def draw_sensitivity(ax, grid: pd.DataFrame, base: tuple[float, float] | None = None):
    values = grid.to_numpy(dtype=float)
    vmax = np.nanmax(np.abs(values)) or 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = ax.imshow(values, cmap=DIVERGING.reversed(), norm=norm, aspect="auto", origin="lower")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            color = "#ffffff" if abs(v) > 0.55 * vmax else INK
            ax.text(j, i, f"{v / 1000:,.0f}k", ha="center", va="center", fontsize=7.5, color=color)
    ax.set_xticks(range(values.shape[1]), [f"${c:,.0f}" for c in grid.columns])
    ax.set_yticks(range(values.shape[0]), [f"{r:.0%}" for r in grid.index])
    ax.set_xlabel("Incentive cost per targeted customer")
    ax.set_ylabel("Save rate of contacted churners")
    ax.grid(False)
    if base is not None and base[0] in list(grid.index) and base[1] in list(grid.columns):
        i = list(grid.index).index(base[0])
        j = list(grid.columns).index(base[1])
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor=INK, lw=1.6))
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v / 1000:,.0f}k"))
    ax.set_title("Net profit sensitivity (test set, $)")
    return ax


def draw_policies(ax, policies: pd.DataFrame, emphasis: tuple[str, ...]):
    d = policies["net_profit_per_1k_customers"].sort_values()
    colors = [BLUE if any(e in p for e in emphasis) else DEEMPH for p in d.index]
    ax.barh(range(len(d)), d.to_numpy(), color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    span = d.abs().max()
    for i, v in enumerate(d.to_numpy()):
        x = v + 0.02 * span if v >= 0 else 0.02 * span
        txt = f"${v:,.0f}" if v >= 0 else f"-${abs(v):,.0f}"
        ax.text(x, i, txt, va="center", ha="left", fontsize=8, color=INK2)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_yticks(range(len(d)), d.index, fontsize=8)
    _money(ax.xaxis, unit="$")
    ax.set_xlim(min(d.min() * 1.15, -0.08 * span), d.max() * 1.3)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Net profit per 1,000 customers (held-out test set)")
    ax.set_title("Targeting strategies compared")
    return ax


# ---------------------------------------------------------------------------- explainability
def draw_concept_importance(ax, mean_abs: pd.Series, units: str = "log-odds", top: int = 12):
    d = mean_abs.sort_values().tail(top)
    ax.barh(range(len(d)), d.to_numpy(), color=BLUE, height=0.62, edgecolor=SURFACE, linewidth=2)
    ax.set_yticks(range(len(d)), d.index)
    ax.text(d.iloc[-1], len(d) - 1, f"  {d.iloc[-1]:.3f}", va="center", fontsize=8, color=INK2)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Mean |SHAP| ({units}), grouped by business concept")
    ax.set_title("What drives churn risk (global)")
    return ax


def draw_beeswarm(
    fig,
    ax,
    values: np.ndarray,
    Xt: np.ndarray,
    names: list[str],
    max_display: int = 15,
    units: str = "log-odds",
):
    imp = np.abs(values).mean(axis=0)
    order = np.argsort(imp)[::-1][:max_display][::-1]
    rng = np.random.default_rng(0)
    for row, j in enumerate(order):
        ranks = pd.Series(Xt[:, j]).rank(pct=True).to_numpy()
        jitter = np.clip(rng.normal(0, 0.11, len(ranks)), -0.32, 0.32)
        ax.scatter(
            values[:, j], row + jitter, c=ranks, cmap=DIVERGING, vmin=0, vmax=1, s=7, alpha=0.85, linewidths=0
        )
    ax.set_yticks(range(len(order)), [names[j] for j in order], fontsize=8)
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"SHAP value (impact on churn {units})")
    cb = fig.colorbar(
        ScalarMappable(norm=Normalize(0, 1), cmap=DIVERGING), ax=ax, ticks=[0, 1], fraction=0.035, pad=0.02
    )
    cb.ax.set_yticklabels(["low", "high"])
    cb.set_label("Feature value", color=INK2, fontsize=8.5)
    cb.outline.set_visible(False)
    ax.set_title("SHAP summary: top features (test customers)")
    return ax


def draw_local_explanation(
    ax, contributions: list[tuple[str, float]], probability: float, customer: str, units: str = "log-odds"
):
    d = sorted(contributions, key=lambda t: t[1])
    labels = [t[0] for t in d]
    vals = np.array([t[1] for t in d])
    colors = [POSITIVE if v > 0 else NEGATIVE for v in vals]
    ax.barh(range(len(vals)), vals, color=colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    span = np.abs(vals).max()
    for i, v in enumerate(vals):
        ax.text(
            v + (0.03 * span if v >= 0 else -0.03 * span),
            i,
            f"{v:+.2f}",
            va="center",
            ha="left" if v >= 0 else "right",
            fontsize=8,
            color=INK2,
        )
    ax.axvline(0, color=AXIS, lw=0.9)
    ax.set_yticks(range(len(vals)), labels, fontsize=8.5)
    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"Contribution to churn {units}   (red raises risk · blue lowers risk)")
    ax.set_title(f"Why customer {customer} is flagged: P(churn) = {probability:.0%}")
    return ax


# ---------------------------------------------------------------------------- survival
def draw_km(ax, km: dict):
    for (group, (t, s)), color in zip(km["curves"].items(), SERIES, strict=False):
        ax.step(t, s, where="post", color=color, lw=2)
        med = km["median_months"].get(group)
        suffix = f" (median {med:.0f} mo)" if med else " (median not reached)"
        ax.text(t[-1] + 0.8, s[-1], f"{group}{suffix}", color=INK2, fontsize=8, va="center")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(0, max(t[-1] for t, _ in km["curves"].values()) * 1.42)
    _pct(ax.yaxis)
    ax.set_xlabel("Months since sign-up (tenure)")
    ax.set_ylabel("Share of customers still active")
    p = km["logrank_p"]
    ptxt = "p < 1e-16" if p < 1e-16 else f"p = {p:.1e}"
    ax.set_title(f"Kaplan-Meier retention by {km['column']} (log-rank {ptxt})")
    return ax


def _pretty_covariate(name: str) -> str:
    refs = {"Contract": "Two year", "InternetService": "DSL", "PaymentMethod": "Credit card (auto)"}
    if "[" in name:
        base, lvl = name.split("[", 1)
        return f"{base}: {lvl.rstrip(']')} vs {refs.get(base, 'ref')}"
    return name.replace("_per_10USD", " (+$10)")


def draw_hazard_ratios(ax, hr: pd.DataFrame):
    d = hr.sort_values("hazard_ratio")
    for i, (_name, row) in enumerate(d.iterrows()):
        sig = row["p_value"] < 0.05
        color = (POSITIVE if row["hazard_ratio"] > 1 else NEGATIVE) if sig else MUTED
        ax.plot([row["ci_lower"], row["ci_upper"]], [i, i], color=color, lw=2, solid_capstyle="round")
        ax.scatter(row["hazard_ratio"], i, s=40, color=color, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.set_xscale("log")
    ax.axvline(1.0, color=AXIS, lw=1)
    ax.set_yticks(range(len(d)), [_pretty_covariate(n) for n in d.index], fontsize=7.8)
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_xlabel("Hazard ratio, 95% CI, log scale (gray = not significant)")
    ax.set_title("Cox proportional-hazards model of time-to-churn")
    return ax


# ---------------------------------------------------------------------------- fairness & drift
def draw_fairness(ax, frame: pd.DataFrame, attribute: str):
    d = frame[frame["attribute"] == attribute]
    metrics = ["roc_auc", "tpr", "fpr", "selection_rate"]
    names = ["ROC-AUC", "TPR (recall)", "FPR", "Selection rate"]
    groups = list(d["group"])
    width = 0.8 / max(len(groups), 1)
    for g_i, (group, color) in enumerate(zip(groups, SERIES, strict=False)):
        row = d[d["group"] == group].iloc[0]
        xs = np.arange(len(metrics)) + (g_i - (len(groups) - 1) / 2) * width
        vals = [row[m] for m in metrics]
        ax.bar(
            xs,
            vals,
            width=width,
            color=color,
            edgecolor=SURFACE,
            linewidth=2,
            label=f"{attribute} = {group} (n={int(row['n'])})",
        )
        for x, v in zip(xs, vals, strict=False):
            ax.text(x, v + 0.015, f"{v:.2f}", ha="center", fontsize=7.2, color=INK2)
    ax.set_xticks(range(len(metrics)), names)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=7.8)
    ax.set_title(f"Fairness audit by {attribute}")
    return ax


def draw_drift(ax, psi: pd.DataFrame, title: str, warn: float = 0.10, alert: float = 0.25, top: int = 12):
    d = psi.head(top).iloc[::-1]
    colors = [STATUS[s] for s in d["status"]]
    ax.barh(range(len(d)), d["psi"], color=colors, height=0.62, edgecolor=SURFACE, linewidth=2)
    for i, (v, s) in enumerate(zip(d["psi"], d["status"], strict=False)):
        ax.text(v + 0.004, i, f"{v:.3f} {s.upper()}", va="center", fontsize=7.5, color=INK2)
    ax.axvline(warn, color=MUTED, lw=0.9)
    ax.axvline(alert, color=MUTED, lw=0.9)
    ax.text(warn, len(d) - 0.35, " warn", fontsize=7.5, color=INK2)
    ax.text(alert, len(d) - 0.35, " alert", fontsize=7.5, color=INK2)
    labels = ["model score" if f == "__churn_score__" else f for f in d["feature"]]
    ax.set_yticks(range(len(d)), labels, fontsize=8)
    ax.set_xlim(0, max(d["psi"].max() * 1.35, alert * 1.3))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Population Stability Index vs. training reference")
    ax.set_title(title)
    return ax


def draw_confusion(ax, cm: np.ndarray, threshold: float):
    ax.imshow(cm, cmap=SEQUENTIAL, aspect="auto")
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            v = cm[i, j]
            ax.text(
                j,
                i,
                f"{v:,}\n{v / total:.1%}",
                ha="center",
                va="center",
                fontsize=11,
                color="#ffffff" if v > cm.max() * 0.55 else INK,
            )
    ax.set_xticks([0, 1], ["Predicted stay", "Predicted churn"])
    ax.set_yticks([0, 1], ["Actual stay", "Actual churn"])
    ax.grid(False)
    ax.set_title(f"Confusion matrix at profit-optimal threshold {threshold:.2f}")
    return ax
