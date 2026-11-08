# Resume material — Telco Churn Intelligence

_Generated from `reports/metrics.json`, `reports/ablation_results.json`, `reports/api_benchmark.json`, `reports/test_summary.json`. Every number below is reproducible with `make all`._

## Project title line

**Telco Churn Intelligence — Profit-Driven Churn Prediction & Retention Engine** | Python, scikit-learn, XGBoost, LightGBM, CatBoost, Optuna, SHAP, lifelines, FastAPI, Streamlit, Docker, GitHub Actions

## Bullets (pick 3–5)

- Built a leakage-free churn-prediction pipeline on 7,043 telecom customers, benchmarking 7 models (logistic regression, random forest, XGBoost, LightGBM, CatBoost, voting and stacking ensembles) tuned with 150 Optuna trials under repeated stratified CV; the logistic regression selected by the one-standard-error rule reached a held-out ROC-AUC of 0.850 (95% CI 0.828–0.869), statistically indistinguishable from the voting/stacking ensembles (DeLong p ≥ 0.61).
- Delivered 2.9× lift in the top risk decile and captured 66% of churners by contacting 30% of customers; probabilities were well calibrated without post-processing (ECE 0.015, Brier 0.133).
- Designed an expected-value next-best-action engine (P(churn) × save rate × survival-based CLV − campaign cost) that picks a retention call, an email offer or no action per customer; under documented cost assumptions it returned $31,077 net per 1,000 held-out customers (contacting everyone lost money), and at equal contact volume the model's targeting earned 24% more than a rule-based heuristic.
- Modelled time-to-churn with a Cox proportional-hazards model (test C-index 0.85); month-to-month contracts carried 10.3× the churn hazard of two-year contracts; used it to estimate each customer's remaining lifetime value.
- Ran paired ablation studies (repeated CV): domain features lifted logistic-regression ROC-AUC by +0.005 (5.8 standard errors), while SMOTE and class weighting added no AUC but worsened Brier score by 20–21% and calibration error up to 5×, so both were dropped.
- Shipped it as a FastAPI service with SHAP reason codes per prediction, a Streamlit what-if and campaign simulator, PSI drift monitoring, Docker and GitHub Actions CI (p95 latency 30 ms per customer incl. reason codes; 74 automated tests, 95.5% line coverage).

## Two-line version (tight one-page resumes)

- End-to-end churn system (7,043 customers): 7-model Optuna/CV benchmark → well-calibrated logistic regression, test ROC-AUC 0.850, 2.9× top-decile lift, Cox survival CLV, SHAP reason codes.
- Profit-optimal next-best-action targeting ($31,077 net / 1,000 customers under stated assumptions) served via FastAPI + Streamlit + Docker with drift monitoring and CI.

## Numbers and where they come from

| Claim | Value | Source |
|---|---|---|
| Test ROC-AUC (95% CI) | 0.850 (95% CI 0.828–0.869) | metrics.json → test.champion |
| Repeated-CV ROC-AUC | 0.851 ± 0.012 (15 folds) | metrics.json → cv |
| Test PR-AUC | 0.673 | metrics.json → test.champion |
| Accuracy @ 0.5 | 80.9% | metrics.json → test.champion.at_threshold |
| Recall / precision @ profit threshold | 75.7% / 56.9% | metrics.json → test.champion.at_profit_threshold |
| Top-decile lift | 2.91× | metrics.json → test.champion.lift |
| Churners captured in top 30% | 66.3% | metrics.json → test.champion.lift |
| Brier / ECE (served model) | 0.1335 / 0.0153 | metrics.json → test.champion |
| Next-best-action net profit / 1k customers | $31,077 | policy_comparison.csv (assumption-based) |
| Customers contacted / churners reached (NBA) | 45% / 79% | policy_comparison.csv |
| Cox test C-index | 0.853 | metrics.json → survival |
| Optuna trials | 150 | metrics.json → tuning |
| API p50 / p95 latency | 20.6 / 29.8 ms | api_benchmark.json |
| Tests / coverage | 74 / 95.5% | test_summary.json |

## How to say it in an interview

- *Why not just report accuracy?* 26.5% churn base rate: a model predicting 'no churn' for everyone is 73.5% accurate. ROC-AUC, PR-AUC, lift and calibrated probabilities are what a retention team can act on.
- *How did you avoid leakage?* Feature quantiles are learned inside the sklearn pipeline per CV fold; tuning, model selection, calibration and thresholds use training data only; the test set is scored once.
- *Why logistic regression over the ensembles?* One-standard-error rule: the best ensemble was only +0.0008 AUC better in CV — consistent across folds but far below CV noise — and on the test set logistic regression was actually highest, with no significant DeLong difference. One model instead of three, exact SHAP, millisecond latency.
- *Why no SMOTE?* The ablation shows it does not improve ranking (AUC) and it inflates probabilities, which breaks expected-value decisions that multiply P(churn) by dollar values.
- *Why were some v3 features removed?* MonthlyCharges is almost a linear function of the subscribed services (R² ≈ 1), so features like Fiber × Charge made the model learn large offsetting weights and the reason codes contradicted each other. Removing them cost < 0.001 AUC.
- *Where do the dollars come from?* An explicit, configurable campaign-profit model (Neslin et al., 2006) with a sensitivity analysis; save rates are assumptions to be measured by an A/B retention experiment (next step: uplift modelling).
