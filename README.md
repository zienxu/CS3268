# CS3268 Track 1 — Does Fixing Fairness Remove Bias, or Just Move It?

Bank account opening fraud detection on the BAF dataset, with fairness fixes (reweighing, group thresholds, ExponentiatedGradient) audited by TreeSHAP for age-proxy relocation.

Team: Zien Xu (data, baseline, integration) · Chloe (fairness metrics and fixes) · Jianrong (interpretability audit) · Sze Ling (report, figures, slides)

---

## Setup (do this once)

1. Open Google Drive → **Shared with me** → right-click `CS3268` → **Organize → Add shortcut** → choose **My Drive**. Keep the name exactly `CS3268`. Without this, the paths below will not work on your account.
2. In any Colab notebook, run:

```python
from google.colab import drive; drive.mount('/content/drive')
import sys; sys.path.append('/content/drive/MyDrive/CS3268/src')
from common import load_split, features, threshold_at_fpr, save_preds
```

3. Check it works:

```python
tr = load_split("train")
print(len(tr), tr["month"].unique())   # expect 675666, [0 1 2 3 4]
print(features(tr).shape)              # expect (675666, 30)
```

---

## Folder layout

Data and outputs live in Google Drive (`MyDrive/CS3268/`). Code lives in this GitHub repo. Never commit `.csv` or `.parquet` files.

| Folder | Contents | Who writes |
|---|---|---|
| `data/` | `base.parquet`, `variant2.parquet` | Zien Xu only, once |
| `preds/` | `preds_<model>.parquet`, one per model | Whoever owns that model |
| `models/` | Saved models (`<model>.json`) | Whoever owns that model |
| `src/` | `common.py` (shared helpers) | Zien Xu only |
| `notebooks/` | One notebook per person per stream | Owner only |

---

## Data

Source: Bank Account Fraud (BAF) suite, NeurIPS 2022 (Kaggle: `sgpjesus/bank-account-fraud-dataset-neurips-2022`).
Stored in Drive: `data/base.parquet` (primary) and `data/variant2.parquet` (robustness check only).
Each file: 1,000,000 rows, 34 columns = 31 original + `id` (row index) + `age_group` (1 if customer_age >= 50) + categoricals cast to category dtype.

### Split (temporal, never shuffled)

| Split | Months | Rows | Share | Fraud rate |
|---|---|---|---|---|
| train | 0–4 | 675,666 | 67.6% | 1.00% |
| val | 5 | 119,323 | 11.9% | 1.18% |
| test | 6–7 | 205,011 | 20.5% | 1.40% |

The fraud rate rises over time. The 5% FPR cutoff is set on honest applicants in val only, so it is unaffected, but precision on test will differ from val.

Use val for all tuning and threshold choices. Touch test only for final reported numbers.

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

Categorical columns (`payment_type`, `employment_status`, `housing_status`, `source`, `device_os`) are stored as category dtype; use XGBoost with `enable_categorical=True`.

---

## Predictions file (the handoff contract)

Every model produces one file, `preds/preds_<model>.parquet`, written with `save_preds()`. Everything downstream (metrics, SHAP comparisons, plots) reads these files, so nobody needs anyone else's code.

| Column | Type | Meaning |
|---|---|---|
| `id` | int | Row id, matches `data/*.parquet` |
| `month` | int | 5 = val, 6–7 = test |
| `y_true` | int | 1 = fraud |
| `age_group` | int | 1 = 50-plus |
| `score` | float | Model's fraud probability (0–1) |
| `decision` | int | 1 = flagged, after applying the cutoff |

Rows: val and test only (months 5–7).

Model names: `baseline`, `reweigh`, `groupthr`, `expgrad`. Sweep variants add a suffix, e.g. `reweigh_s0.5`.

Cutoff: set with `threshold_at_fpr(val_scores, val_y, fpr=0.05)` so that 5% of honest val applicants are flagged. Group thresholds pass a dict `{0: cutoff_under50, 1: cutoff_50plus}` to `save_preds()`.

---

## Baseline (W1, done 22 Sep)

Notebook: `notebooks/01_baseline.py` · Outputs: `models/baseline.json`, `models/baseline_grid.csv`, `preds/preds_baseline.parquet`

### Model

XGBoost, `tree_method="hist"`, `enable_categorical=True`, seed 42. Tuned with a 12-point grid (max_depth × learning_rate × min_child_weight), early stopping on val PR-AUC, selected by **recall at 5% FPR on val**.

```python
BEST_PARAMS = dict(n_estimators=232, max_depth=4, learning_rate=0.1,
                   min_child_weight=10, subsample=0.8, colsample_bytree=0.8)
```

`n_estimators` is now FIXED at the best iteration. Every fairness fix trains the same-sized model via `train_model()` — no early stopping downstream, so differences come from the intervention and not from model size.

The top five grid configurations spanned recall 0.5443–0.5478, a range of ~5 fraudsters out of ~1,400 in val — well inside sampling noise (±~1.3pp SE). Selection among them does not affect conclusions; state this in the report.

### Overall performance

| Split | FPR | Recall | n |
|---|---|---|---|
| val (month 5) | 0.0500 | 0.5478 | 119,323 |
| test (months 6–7) | 0.0622 | 0.5893 | 205,011 |

Val FPR is 5.00% by construction (the cutoff is set there). **Test FPR is 6.22%** — the fixed cutoff flags ~24% more honest applicants than promised on unseen months. Honest applicants score higher in later months, i.e. distribution shift, not the rising fraud rate (FPR is computed on honest rows only). Test recall rises for the same reason. Report per-group test FPR for every intervention, not just val.

### Fairness gap (the finding the project is built on)

group 0 = under 50, group 1 = 50-plus.

| Split | Group | FPR | Recall | n |
|---|---|---|---|---|
| val | 0 | 0.0368 | 0.4601 | 98,598 |
| val | 1 | 0.1135 | 0.7234 | 20,725 |
| test | 0 | 0.0465 | 0.5109 | 171,232 |
| test | 1 | 0.1431 | 0.7391 | 33,779 |

**FPR ratio (lower/higher) = 0.32 on val, 0.33 on test.** Parity would be 1.0. Honest 50-plus applicants are ~3x more likely to be wrongly rejected than honest under-50s, and the gap is stable across splits.

50-plus applicants are ~17% of applicants but ~39% of honest applicants wrongly rejected (approximate — Chloe's bootstrap figures supersede this).

The model also catches more 50-plus fraud (0.72 vs 0.46 recall): it treats "older" as riskier across the board, which tracks the 2.34% vs 0.83% fraud-rate difference in the data. Higher base rates explain the direction of the gap; they do not justify it, since FPR is measured only on innocent people. Per Chouldechova (2017) and Kleinberg et al. (2016), equal calibration and equal error rates cannot both hold when group base rates differ — our interventions trade the latter for the former, and the trade-off curves quantify the cost.

**Checkpoint (Sun 4 Oct): PASSED.** The gap is large and stable; we stay on Base. Variant II remains a robustness check only.

---

## Rules

- **Seed 42** everywhere (`random_state=42`, `np.random.seed(42)`).
- **Edit only your own notebook.** Shared code changes go through Zien Xu.
- **Never overwrite someone else's file** in `preds/` or `models/`.
- **The proxy set is frozen** in `proxy_set.json` by a dated commit before any fairness fix runs. Do not edit it afterwards.
- **Test data is for final numbers only.** No tuning or threshold choices on months 6–7.
- **Record new dependencies** in `requirements.txt`.
- **Code freeze: Sun 1 Nov.** No new experiments after that date.
- **`BEST_PARAMS` is frozen.** All interventions call `train_model()` with it. Do not re-tune per intervention.
- **Model selection rule was fixed in advance**: highest recall at 5% FPR on val. Do not switch to a different criterion after seeing results.

## Method notes for the report

- The val month (5) is used for three things: early stopping, hyperparameter selection, and cutoff selection. Test months (6–7) are untouched until final numbers. Say this explicitly in Method so it does not read as an oversight.
- Cutoffs are always set on val and applied unchanged to test, mirroring how a bank sets a policy on past data and applies it to new applicants.
- Test FPR drifting above the 5% target is a reportable finding about distribution shift, not a bug.
