# Model Card — Telco churn risk model

Format follows Mitchell et al. (2019), *Model Cards for Model Reporting*.

## Model details
- **Model:** Logistic Regression pipeline (feature engineering → imputation/scaling/one-hot → classifier) with `none` probability calibration.
- **Version:** 4.0.0 · trained 2026-11-06T15:31:14Z · scikit-learn 1.9.1 · Python 3.13.18.
- **Selection:** `one_se` rule over 7 tuned candidates using repeated stratified CV on training data.
- **Artifact:** `artifacts/model_bundle.joblib` (model, explainer, survival-CLV model, action menu, drift reference, metadata).

## Intended use
- Rank existing residential telecom customers by 1-month churn risk and recommend a cost-justified retention action.
- Users: retention / CRM analysts and call-centre tooling via the REST API.
- **Out of scope:** credit, pricing or any adverse decision about a customer; populations unlike the IBM Telco sample (e.g. business accounts, other countries) without re-validation.

## Data
- IBM Telco Customer Churn sample (Apache-2.0): 7,043 customers, 19 raw inputs, churn rate 26.5%.
- Stratified split: 5,634 train / 1,409 test. Test set used once for final reporting.
- Single snapshot; churn = left within the last month. No timestamps, so no out-of-time validation is possible.

## Performance (held-out test set)
- ROC-AUC 0.8497 (95% CI 0.8284–0.8690); PR-AUC 0.6728; Brier 0.1335; ECE 0.0153.
- Top-10% lift 2.91×; 66.3% of churners in the top 30%.
- At the profit-optimal threshold 0.33: precision 56.9%, recall 75.7%.

## Fairness considerations
- `gender` is excluded from model inputs. `SeniorCitizen` is used (it carries signal); seniors have a higher base churn rate, so they are selected more often. Because the action is a *benefit* (a retention offer), this is lower-risk than an adverse decision, but it should be reviewed against local regulation.
- gender: TPR gap 0.043, FPR gap 0.003, AUC gap 0.006, selection-rate ratio 0.98.
- SeniorCitizen: TPR gap 0.178, FPR gap 0.232, AUC gap 0.058, selection-rate ratio 0.49.

## Business assumptions
- Gross margin 60%, CLV horizon 24 months, action costs and save rates in `configs/config.yaml`. These are illustrative; profit figures scale with them (see the sensitivity heatmap).

## Caveats and limitations
- Save rates are assumed, not estimated: a randomised retention experiment (uplift modelling) is required to measure them.
- The model predicts churn risk, not treatment effect; customers who would leave regardless are still targeted.
- Cox proportional-hazards assumption is approximate for contract type; CLV is a planning estimate, not accounting value.
- Re-train when PSI on key inputs or on the score exceeds 0.25 (see `scripts/monitor_drift.py`).
