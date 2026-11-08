# Data

`WA_Fn-UseC_-Telco-Customer-Churn.csv` — IBM Telco Customer Churn sample dataset
(7,043 customers × 21 columns, churn rate 26.5%).

- Source: <https://github.com/IBM/telco-customer-churn-on-icp4d> (`data/Telco-Customer-Churn.csv`), Apache License 2.0.
- Same content as the widely used Kaggle file `WA_Fn-UseC_-Telco-Customer-Churn.csv`.
- Re-download with `python scripts/download_data.py`.

| Column | Meaning |
|---|---|
| customerID | Unique id |
| gender, SeniorCitizen, Partner, Dependents | Demographics / household |
| tenure | Months with the company (0–72) |
| PhoneService, MultipleLines | Phone products |
| InternetService | DSL / Fiber optic / No |
| OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies | Internet add-ons |
| Contract | Month-to-month / One year / Two year |
| PaperlessBilling, PaymentMethod | Billing |
| MonthlyCharges, TotalCharges | Current monthly bill and lifetime billed amount (blank for 11 brand-new customers) |
| Churn | Left within the last month (target) |
