# CS3268
Bank Account Fraud Detection

## Data

Source: Bank Account Fraud (BAF) suite, NeurIPS 2022 (Kaggle: sgpjesus/bank-account-fraud-dataset-neurips-2022).
Stored in Drive: `data/base.parquet` (primary) and `data/variant2.parquet` (robustness check only).
Each file: 1,000,000 rows, 34 columns = 31 original + `id` (row index) + `age_group` (1 if customer_age >= 50) + categoricals cast to category dtype.

### Split (temporal, never shuffled)

| Split | Months | Rows | Share | Fraud rate |
|---|---|---|---|---|
| train | 0–4 | 675,666 | 67.6% | 1.00% |
| val | 5 | 119,323 | 11.9% | 1.18% |
| test | 6–7 | 205,011 | 20.5% | 1.40% |

The fraud rate rises over time. The 5% FPR cutoff is set on honest applicants in val only, so it is unaffected, but precision on test will differ from val.

### Protected attribute

`customer_age` comes in decade bins (10, 20, …, 90). `age_group` = 1 for 50 and above.

| Dataset | Fraud rate, under 50 | Fraud rate, 50+ |
|---|---|---|
| Base | 0.83% | 2.34% |
| Variant II | 0.36% | 1.82% |

Note: the proposal cited 0.9% vs 1.8% for Base. The numbers above are measured and supersede it.

### Negative values

Missing values are coded as negatives in six columns (checked against the BAF datasheet):

| Column | Missing code | Share missing |
|---|---|---|
| prev_address_months_count | -1 | 71.3% |
| intended_balcon_amount | any negative value (not just -1) | 74.3% |
| bank_months_count | -1 | 25.4% |
| current_address_months_count | -1 | 0.4% |
| session_length_in_minutes | -1 | 0.2% |
| device_distinct_emails_8w | -1 | <0.1% |

Genuine negative values (NOT missing): `credit_risk_score` (valid range -191 to 398; 1.4% negative) and `velocity_6h` (a handful of rows).

Tree models (XGBoost): use all values as-is — do NOT impute or drop.
Linear models (e.g. practice logistic regression): for the six missing-coded columns only, add a missing-indicator column, then median-fill. Detect missing with `< 0`, not `== -1`, or intended_balcon_amount will be missed.

### Model inputs

All columns except `id`, `fraud_bool` (label), `month`, `age_group`. `customer_age` stays IN — fairness fixes act on weights and thresholds, never by dropping age. Use `features(df)` from `src/common.py`.
