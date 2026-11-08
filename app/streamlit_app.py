"""Streamlit dashboard: what-if scoring, batch scoring, campaign simulator and model report.

streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from churnintel.business import (  # noqa: E402
    RetentionAction,
    compare_policies,
    next_best_action,
    optimal_threshold,
    profit_curve,
)
from churnintel.explain.explainer import describe  # noqa: E402
from churnintel.inference import ChurnPredictor  # noqa: E402
from churnintel.viz import plots as P  # noqa: E402

MODEL_PATH = Path(os.getenv("CHURN_MODEL_PATH", ROOT / "artifacts" / "model_bundle.joblib"))
REPORTS = Path(os.getenv("CHURN_REPORTS_DIR", ROOT / "reports"))
SAMPLE = ROOT / "data" / "sample" / "sample_customers.csv"

st.set_page_config(page_title="Telco Churn Intelligence", layout="wide")


@st.cache_resource
def load_predictor() -> ChurnPredictor | None:
    return ChurnPredictor.load(MODEL_PATH) if MODEL_PATH.exists() else None


@st.cache_data
def load_reports() -> tuple[dict | None, pd.DataFrame | None]:
    m = json.loads((REPORTS / "metrics.json").read_text()) if (REPORTS / "metrics.json").exists() else None
    tp = (
        pd.read_csv(REPORTS / "test_predictions.csv") if (REPORTS / "test_predictions.csv").exists() else None
    )
    return m, tp


predictor = load_predictor()
metrics, test_preds = load_reports()

st.title("Telco Churn Intelligence")
if predictor is None:
    st.error("No trained model found. Run `make train` (or `python scripts/train.py`) first.")
    st.stop()

md = predictor.metadata
st.caption(
    f"Model: **{md.get('model_name')}** · {md.get('calibration')} calibration · trained {md.get('trained_at')} · "
    f"held-out ROC-AUC {md.get('test_roc_auc', float('nan')):.3f}"
)

tab_whatif, tab_batch, tab_sim, tab_report = st.tabs(
    ["Customer what-if", "Batch scoring", "Campaign simulator", "Model report"]
)

# ---------------------------------------------------------------------------- what-if
with tab_whatif:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.subheader("Account")
        tenure = st.slider("Tenure (months)", 0, 72, 3)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        payment = st.selectbox(
            "Payment method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )
        paperless = st.radio("Paperless billing", ["Yes", "No"], horizontal=True)
        monthly = st.number_input("Monthly charges ($)", 18.0, 120.0, 98.5, step=0.5)
        total = st.number_input(
            "Total charges ($)", 0.0, 9000.0, float(round(monthly * tenure, 2)), step=10.0
        )
    with c2:
        st.subheader("Services")
        phone = st.radio("Phone service", ["Yes", "No"], horizontal=True)
        multi = st.selectbox("Multiple lines", ["No", "Yes"] if phone == "Yes" else ["No phone service"])
        internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
        addon_opts = ["No", "Yes"] if internet != "No" else ["No internet service"]
        security = st.selectbox("Online security", addon_opts)
        backup = st.selectbox("Online backup", addon_opts)
        device = st.selectbox("Device protection", addon_opts)
        support = st.selectbox("Tech support", addon_opts)
        tv = st.selectbox("Streaming TV", addon_opts[::-1] if internet != "No" else addon_opts)
        movies = st.selectbox("Streaming movies", addon_opts[::-1] if internet != "No" else addon_opts)
    with c3:
        st.subheader("Household")
        senior = st.radio(
            "Senior citizen", [0, 1], horizontal=True, format_func=lambda v: "Yes" if v else "No"
        )
        partner = st.radio("Partner", ["No", "Yes"], horizontal=True)
        dependents = st.radio("Dependents", ["No", "Yes"], horizontal=True)

    customer = {
        "customerID": "WHAT-IF",
        "gender": "Female",
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multi,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": device,
        "TechSupport": support,
        "StreamingTV": tv,
        "StreamingMovies": movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
    }
    row = predictor.score([customer], explain=True).iloc[0]
    st.divider()
    k1, k2, k3, k4 = st.columns(4)
    k1.metric(
        "Churn probability", f"{row['churn_probability']:.0%}", help="Calibrated probability of churning"
    )
    k2.metric("Risk tier", row["risk_tier"].title())
    k3.metric(
        "Value if retained (CLV)", f"${row['clv_estimate']:,.0f}", help="Survival-based, margin-adjusted"
    )
    k4.metric("Expected value of action", f"${row['expected_value']:,.0f}")
    st.markdown(f"**Recommended action:** `{row['recommended_action']}` — {row['action_description']}")

    left, right = st.columns([1, 1.25])
    exp = row["explanation"]
    with left:
        st.markdown("**Why this score**")
        for r in exp["risk_drivers"]:
            st.markdown(f"- :red[▲] {r['description']}  ({r['impact']:+.2f})")
        for r in exp["protective_factors"]:
            st.markdown(f"- :blue[▼] {r['description']}  ({r['impact']:+.2f})")
    with right:
        X = predictor.prepare([customer])
        concepts = predictor.explainer.concept_values(X).iloc[0]
        contrib = sorted(
            [(describe(c, X.iloc[0], v), float(v)) for c, v in concepts.items() if abs(v) > 1e-6],
            key=lambda t: abs(t[1]),
            reverse=True,
        )[:8]
        if contrib:
            fig, ax = P.new_figure(7.5, 4.0)
            P.draw_local_explanation(
                ax, contrib, float(row["churn_probability"]), "what-if", predictor.explainer.units_
            )
            st.pyplot(fig, clear_figure=True)

# ---------------------------------------------------------------------------- batch
with tab_batch:
    st.write("Upload a CSV with the 19 raw Telco columns (customerID optional) or use the bundled sample.")
    uploaded = st.file_uploader("Customers CSV", type=["csv"])
    use_sample = st.checkbox("Use sample file", value=uploaded is None)
    df = (
        pd.read_csv(uploaded)
        if uploaded is not None
        else (pd.read_csv(SAMPLE) if use_sample and SAMPLE.exists() else None)
    )
    if df is not None:
        scored = predictor.score(df, explain=False)
        a, b, c = st.columns(3)
        a.metric("Customers", f"{len(scored):,}")
        b.metric("Mean churn probability", f"{scored['churn_probability'].mean():.1%}")
        c.metric("Total expected value", f"${scored['expected_value'].sum():,.0f}")
        st.dataframe(scored.sort_values("churn_probability", ascending=False), use_container_width=True)
        st.download_button(
            "Download scored CSV", scored.to_csv().encode(), "scored_customers.csv", "text/csv"
        )

# ---------------------------------------------------------------------------- simulator
with tab_sim:
    if test_preds is None or metrics is None:
        st.info("Run training to generate reports/test_predictions.csv.")
    else:
        st.write(
            "Re-run the retention economics on the **held-out test customers** with your own assumptions. "
            "Profit is realised profit: value is credited only for customers who actually churned."
        )
        base = {a.name: a for a in predictor.actions}
        call, email = base.get("retention_call"), base.get("email_offer")
        s1, s2, s3 = st.columns(3)
        with s1:
            st.markdown("**Retention call**")
            call_save = st.slider("Save rate (call)", 0.05, 0.8, float(call.save_rate), 0.05)
            call_inc = st.slider("Incentive ($, call)", 0, 200, int(call.incentive_cost), 5)
            call_contact = st.slider("Contact cost ($, call)", 0, 100, int(call.contact_cost), 1)
        with s2:
            st.markdown("**Email offer**")
            em_save = st.slider("Save rate (email)", 0.0, 0.5, float(email.save_rate), 0.01)
            em_inc = st.slider("Incentive ($, email)", 0, 100, int(email.incentive_cost), 5)
            em_contact = st.slider("Contact cost ($, email)", 0, 20, int(email.contact_cost), 1)
        with s3:
            st.markdown("**Value**")
            base_margin = float(metrics["business"]["assumptions"]["gross_margin"])
            margin = st.slider("Gross margin", 0.2, 0.9, base_margin, 0.05)

        actions = [
            RetentionAction(
                "retention_call", call.label, float(call_contact), float(call_inc), float(call_save)
            ),
            RetentionAction("email_offer", email.label, float(em_contact), float(em_inc), float(em_save)),
        ]
        y = test_preds["churned"].to_numpy()
        p = test_preds["churn_probability"].to_numpy()
        clv = test_preds["clv_estimate"].to_numpy() * margin / base_margin
        t_star, _, _ = optimal_threshold(p, y, clv, actions[0])
        pol = compare_policies(
            p, y, clv, test_preds["heuristic_score"].to_numpy(), actions, "retention_call", t_star
        )
        nba = pol.loc["ML next-best-action (EV-optimal)"]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Customers contacted", f"{int(nba['customers_targeted']):,}", f"{nba['pct_targeted']:.0%} of base"
        )
        m2.metric("Churners reached", f"{nba['churner_capture_rate']:.0%}")
        m3.metric("Net profit (test set)", f"${nba['net_profit']:,.0f}")
        m4.metric("ROI", f"{nba['roi']:.2f}" if np.isfinite(nba["roi"]) else "—")
        st.caption(
            "Threshold here is tuned on the test set itself for the simulation; the trained pipeline tunes it on "
            "training out-of-fold predictions."
        )
        view = pol[
            ["customers_targeted", "churner_capture_rate", "campaign_cost", "net_profit", "roi"]
        ].copy()
        st.dataframe(
            view.style.format(
                {
                    "churner_capture_rate": "{:.0%}",
                    "campaign_cost": "${:,.0f}",
                    "net_profit": "${:,.0f}",
                    "roi": "{:.2f}",
                }
            ),
            use_container_width=True,
        )
        fr = np.linspace(0, 1, 101)
        curves = {
            "Champion model ranking": profit_curve(p, y, clv, actions[0], fr),
            "Rule-based heuristic ranking": profit_curve(
                test_preds["heuristic_score"].to_numpy(), y, clv, actions[0], fr
            ),
        }
        fig, ax = P.new_figure(8, 4)
        f0, v0 = curves["Champion model ranking"]
        P.draw_profit_curves(ax, curves, (float(f0[int(np.argmax(v0))]), float(np.max(v0))))
        st.pyplot(fig, clear_figure=True)
        idx, _ = next_best_action(p, clv, actions)
        st.write(
            "Action mix:",
            pd.Series([actions[i].name if i >= 0 else "no_action" for i in idx])
            .value_counts()
            .rename("customers"),
        )

# ---------------------------------------------------------------------------- report
with tab_report:
    if metrics is None:
        st.info("Run training to generate reports.")
    else:
        t = metrics["test"]["champion"]
        ci = t["bootstrap_ci"]["roc_auc"]
        r1, r2, r3, r4 = st.columns(4)
        r1.metric(
            "Test ROC-AUC",
            f"{t['roc_auc']:.3f}",
            f"95% CI {ci['ci_lower']:.3f}–{ci['ci_upper']:.3f}",
            delta_color="off",
        )
        r2.metric("Test PR-AUC", f"{t['pr_auc']:.3f}")
        r3.metric("Top-10% lift", f"{t['lift']['top_10pct']['lift']:.2f}×")
        r4.metric("Brier score", f"{t['brier']:.4f}")
        if (REPORTS / "model_comparison.csv").exists():
            st.dataframe(pd.read_csv(REPORTS / "model_comparison.csv", index_col=0), use_container_width=True)
        for name in [
            "dashboard.png",
            "10_shap_summary.png",
            "14_cox_hazard_ratios.png",
            "15_fairness_audit.png",
        ]:
            path = REPORTS / "figures" / name
            if path.exists():
                st.image(str(path), use_container_width=True)
