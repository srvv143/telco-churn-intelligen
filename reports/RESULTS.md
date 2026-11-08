# Results — Telco Churn Intelligence v4

_Auto-generated from `reports/metrics.json` (trained 2026-11-06T15:31:14Z, runtime 7.3 min). Re-run `make train` to refresh._

## Headline — held-out test set (n = 1,409, never used for any decision)

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| ROC-AUC | 0.8497 | 0.8284 – 0.8690 |
| PR-AUC (average precision) | 0.6728 | 0.6307 – 0.7150 |
| Brier score (lower is better) | 0.1335 | 0.1254 – 0.1424 |
| Accuracy @ 0.5 | 80.9% | 79.1% – 82.7% |
| Lift in top 10% | 2.91× | 2.65× – 3.18× |
| Churners captured in top 30% | 66.3% | 62.6% – 70.3% |
| KS statistic | 0.551 | — |
| Expected calibration error | 0.0153 | — |

**Champion:** Logistic Regression with `none` calibration, chosen by the `one_se` rule from 7 candidates — repeated-CV ROC-AUC 0.8506 ± 0.0122 over 15 folds (training data only).

## Model comparison

| Model | CV ROC-AUC (mean ± sd) | CV PR-AUC | CV Brier | Test ROC-AUC | Test PR-AUC | DeLong p vs champion |
|---|---|---|---|---|---|---|
| Stacking Ensemble | 0.8513 ± 0.0124 | 0.6720 | 0.1344 | 0.8492 | 0.6709 | 0.632 |
| Soft-Voting Ensemble | 0.8513 ± 0.0124 | 0.6713 | 0.1329 | 0.8490 | 0.6706 | 0.608 |
| **Logistic Regression** (champion) | 0.8506 ± 0.0122 | 0.6734 | 0.1328 | 0.8497 | 0.6728 | — |
| CatBoost | 0.8502 ± 0.0124 | 0.6680 | 0.1333 | 0.8478 | 0.6682 | 0.359 |
| XGBoost | 0.8502 ± 0.0124 | 0.6672 | 0.1335 | 0.8470 | 0.6654 | 0.165 |
| LightGBM | 0.8491 ± 0.0128 | 0.6655 | 0.1341 | 0.8478 | 0.6655 | 0.455 |
| Random Forest | 0.8471 ± 0.0130 | 0.6601 | 0.1352 | 0.8468 | 0.6599 | 0.351 |
| Rule-based heuristic (v3 ManualRiskScore) | — | — | — | 0.8206 | 0.5831 | 5.6e-06 |

Selection detail: best CV model = Stacking Ensemble (0.8513); rule `one_se`, cutoff = best − SE = 0.8481; eligible = Logistic Regression, LightGBM, XGBoost, CatBoost, Soft-Voting Ensemble, Stacking Ensemble. Paired per-fold AUC difference vs. the best model (mean ± SE, share of folds where the model was better): Stacking Ensemble (reference); Soft-Voting Ensemble -0.0000 ± 0.0001 (47%); Logistic Regression -0.0008 ± 0.0002 (27%); CatBoost -0.0012 ± 0.0004 (20%); XGBoost -0.0012 ± 0.0002 (7%); LightGBM -0.0023 ± 0.0005 (13%); Random Forest -0.0042 ± 0.0007 (7%). DeLong p-values > 0.05 mean the test-set AUC difference to the champion is not statistically significant.

Tuning budget: Logistic Regression 30 trials (15 pruned), Random Forest 25 trials (8 pruned), XGBoost 40 trials (17 pruned), LightGBM 40 trials (6 pruned), CatBoost 15 trials (2 pruned).

## Calibration (out-of-fold on training data)

| Method | Brier | ECE | Log-loss | ROC-AUC |
|---|---|---|---|---|
| none | 0.1331 | 0.0105 | 0.4110 | 0.8493 |
| sigmoid | 0.1332 | 0.0101 | 0.4110 | 0.8493 |
| isotonic | 0.1333 | 0.0118 | 0.4192 | 0.8470 |

The unweighted logistic model was already well calibrated: neither Platt (sigmoid) nor isotonic scaling lowered out-of-fold Brier, so no recalibration layer is served (test Brier 0.1335, ECE 0.0153).

## Decision thresholds (chosen on training out-of-fold predictions)

| Threshold | Value | Test precision | Test recall | Test F1 | Test accuracy |
|---|---|---|---|---|---|
| Profit-optimal (retention call) | 0.33 | 56.9% | 75.7% | 0.650 | 78.4% |
| F1-optimal (reference) | 0.33 | 57.1% | 75.4% | 0.650 | 78.4% |
| 0.50 (default) | 0.50 | 68.2% | 52.7% | 0.594 | 80.9% |

Threshold vs. incentive cost: low cost ($30): t* = 0.19, base ($60): t* = 0.33, high cost ($120): t* = 0.48.

## Business impact — held-out test set

| Policy | Targeted | Churners captured | Precision | Campaign cost | Net profit | Net / 1k customers | ROI |
|---|---|---|---|---|---|---|---|
| No campaign | 0 (0%) | 0% | — | $0 | $0 | $0 | — |
| Blanket campaign (contact everyone) | 1,409 (100%) | 100% | 27% | $105,179 | -$5,862 | -$4,160 | -0.06 |
| ML threshold policy (p >= 0.33) | 497 (35%) | 76% | 57% | $31,208 | $42,605 | $30,238 | 1.37 |
| Rule-based heuristic (same volume) | 497 (35%) | 71% | 53% | $31,949 | $34,233 | $24,296 | 1.07 |
| Random targeting (same volume, expected) | 497 (35%) | 35% | 27% | $37,100 | -$2,068 | -$1,467 | -0.06 |
| ML next-best-action (EV-optimal) | 628 (45%) | 79% | 47% | $33,135 | $43,787 | $31,077 | 1.32 |
| Oracle (perfect foresight, reference) | 374 (27%) | 100% | 100% | $17,204 | $82,113 | $58,278 | 4.77 |

Under these assumptions the threshold policy earns **1.24×** the net profit of the rule-based heuristic at the same contact volume, and the next-best-action policy is profitable in **100%** of the save-rate × incentive scenarios in the sensitivity grid.

**Assumptions (configurable in `configs/config.yaml`, illustrative — not measured):** gross margin 60%; CLV = margin × monthly charge × expected remaining months within 24 months (survival method); mean test CLV $747.

| Action | Contact cost | Incentive cost | Save rate | Description |
|---|---|---|---|---|
| retention_call | $25 | $60 | 35% | Retention specialist call + 1-month bill credit for moving to a 12-month contract |
| email_offer | $2 | $20 | 12% | Personalised email + $20 bill credit on contract upgrade |

Profit model (Neslin et al., 2006): targeted churner → save_rate × (CLV − incentive) − contact; targeted non-churner → −(contact + incentive). Value is credited only for customers who actually churned.

## Survival analysis (Cox proportional hazards, time-to-churn)

- Concordance (C-index): train 0.867, **test 0.853**.
- Month-to-month vs. two-year contract hazard ratio: **10.3×** (95% CI 8.2–13.0).
- Log-rank test across contract types: p = 0.0e+00.

| Contract | Kaplan-Meier median lifetime |
|---|---|
| Month-to-month | 36 months |
| One year | not reached |
| Two year | not reached |

| Covariate | Hazard ratio | 95% CI | p |
|---|---|---|---|
| Contract[Month-to-month] | 10.31 | 8.18 – 13.01 | 2.3e-86 |
| InternetService[No] | 0.36 | 0.27 – 0.48 | 3.9e-12 |
| Contract[One year] | 2.02 | 1.57 – 2.61 | 6.4e-08 |
| OnlineBackup | 0.51 | 0.45 – 0.57 | 8.0e-28 |
| OnlineSecurity | 0.52 | 0.45 – 0.60 | 4.4e-19 |
| PaymentMethod[Electronic check] | 1.92 | 1.65 – 2.23 | 1.2e-17 |

## Explainability

Explainer: `linear` SHAP on the champion pipeline, grouped into business concepts.

| Concept | Mean |SHAP| (log-odds) |
|---|---|
| Contract type | 0.673 |
| Monthly charges | 0.445 |
| Tenure | 0.369 |
| Internet service | 0.342 |
| Protection add-ons | 0.291 |
| Phone lines | 0.218 |
| Billing type | 0.183 |
| Payment method | 0.159 |
| Streaming add-ons | 0.131 |
| Lifetime spend | 0.108 |

Highest-risk test customer `5178-LMXOP` (P(churn) = 94.7%) — reason codes returned by the API:

1. Month-to-month contract (+1.48 log-odds)
2. Short tenure (1 month) (+0.60 log-odds)
3. High monthly bill ($95) (+0.51 log-odds)

## Fairness audit (test set, profit-optimal threshold)

| Attribute | TPR gap | FPR gap | AUC gap | Selection-rate ratio |
|---|---|---|---|---|
| gender | 0.043 | 0.003 | 0.006 | 0.979 |
| SeniorCitizen | 0.178 | 0.232 | 0.058 | 0.487 |

Gender is excluded from the model's inputs. Gaps for SeniorCitizen largely reflect different base churn rates; see `reports/fairness_groups.csv` and the model card before using scores for anything other than offering retention incentives.

## Drift monitoring (PSI)

- Held-out test vs. training reference: max PSI 0.012 (0 features above the warn level) — stable, as expected for a random split.
- Simulated production shift (+15% price, 25% of term contracts lapsing): 1 alerts — MonthlyCharges 0.85, Contract 0.05, TotalCharges 0.03, model score 0.02.

## Ablation studies (repeated CV on training data)

| Experiment | Variant | Model | CV ROC-AUC | CV Brier | CV ECE | Δ AUC vs. reference (paired) |
|---|---|---|---|---|---|---|
| reference | default | Logistic Regression | 0.8506 ± 0.0122 | 0.1328 | 0.0286 | reference |
| A1 feature engineering | raw inputs only | Logistic Regression | 0.8455 ± 0.0119 | 0.1352 | 0.0263 | -0.0051 ± 0.0009 |
| A2 class imbalance | class_weight=balanced | Logistic Regression | 0.8506 ± 0.0122 | 0.1609 | 0.1465 | +0.0001 ± 0.0001 |
| A2 class imbalance | SMOTE (in-fold) | Logistic Regression | 0.8495 ± 0.0126 | 0.1593 | 0.1371 | -0.0011 ± 0.0004 |
| A3 gender input | gender included | Logistic Regression | 0.8503 ± 0.0122 | 0.1330 | 0.0277 | -0.0003 ± 0.0001 |
| reference | default | LightGBM | 0.8491 ± 0.0128 | 0.1341 | 0.0297 | reference |
| A1 feature engineering | raw inputs only | LightGBM | 0.8494 ± 0.0125 | 0.1339 | 0.0285 | +0.0003 ± 0.0003 |
| A2 class imbalance | class_weight=balanced | LightGBM | 0.8489 ± 0.0127 | 0.1614 | 0.1465 | -0.0002 ± 0.0002 |
| A2 class imbalance | SMOTE (in-fold) | LightGBM | 0.8462 ± 0.0119 | 0.1486 | 0.1016 | -0.0029 ± 0.0008 |
| A3 gender input | gender included | LightGBM | 0.8494 ± 0.0126 | 0.1340 | 0.0287 | +0.0003 ± 0.0002 |

- Logistic Regression — raw inputs only: ΔAUC -0.0051 ± 0.0009 (AUC loss), ΔBrier +0.0024 (worse calibration/accuracy of probabilities)
- Logistic Regression — class_weight=balanced: ΔAUC +0.0001 ± 0.0001 (negligible AUC change), ΔBrier +0.0281 (worse calibration/accuracy of probabilities)
- Logistic Regression — SMOTE (in-fold): ΔAUC -0.0011 ± 0.0004 (AUC loss), ΔBrier +0.0265 (worse calibration/accuracy of probabilities)
- Logistic Regression — gender included: ΔAUC -0.0003 ± 0.0001 (negligible AUC change), ΔBrier +0.0002 (worse calibration/accuracy of probabilities)
- LightGBM — raw inputs only: ΔAUC +0.0003 ± 0.0003 (negligible AUC change), ΔBrier -0.0002 (better calibration/accuracy of probabilities)
- LightGBM — class_weight=balanced: ΔAUC -0.0002 ± 0.0002 (negligible AUC change), ΔBrier +0.0273 (worse calibration/accuracy of probabilities)
- LightGBM — SMOTE (in-fold): ΔAUC -0.0029 ± 0.0008 (AUC loss), ΔBrier +0.0146 (worse calibration/accuracy of probabilities)
- LightGBM — gender included: ΔAUC +0.0003 ± 0.0002 (negligible AUC change), ΔBrier -0.0001 (better calibration/accuracy of probabilities)

## Engineering

- API latency (500 sequential requests, single customer with reason codes): p50 20.6 ms, p95 29.8 ms; batch of 1000: 11,986 customers/s. Measured 2026-11-06 on Linux x86_64, 2 vCPU, Python 3.13.18.
- Test suite: 74 passed, 0 failed; line coverage 95.5%.

## Figures

All figures are in `reports/figures/` (dashboard: `reports/figures/dashboard.png`).
