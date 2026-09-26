# CS3268 Track 1 — Does Fixing Fairness Remove Bias, or Just Move It?

Bank account opening fraud detection on the BAF dataset, with fairness fixes (reweighing, group thresholds, ExponentiatedGradient) audited by TreeSHAP for age-proxy relocation.

Team: Zien Xu (data, baseline, integration) · Chloe (fairness metrics and fixes) · Jianrong (interpretability audit) · Sze Ling (report, figures, slides)

---

## Setup (do this once)

1. Open Google Drive → **Shared with me** → right-click `CS3268` → **Organize → Add shortcut** → choose **My Drive**. Keep the name exactly `CS3268`. Without this, the paths below will not work on your account.
2. In any Colab notebook, run:

```python
from google.colab import drive; drive.mount('/content/drive')
import sys; sys.path.append('/content/drive/MyDrive/CS3268_Project/src')
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

## Intervention 1: ExponentiatedGradient (W2, run 23 Sep) — PRELIMINARY
 
Notebook: `notebooks/02_expgrad.py` · Outputs: `preds/preds_expgrad.parquet`, `models/expgrad_components/`, `models/expgrad_weights.csv`
 
Status: **one configuration only (`difference_bound=0.01`). Do not write this up as a final result until the bound sweep is done** — see "Open question" below.
 
### Setup
 
Fairlearn's implementation of the reductions approach of Agarwal et al. [3], with a `FalsePositiveRateParity` constraint, `max_iter=40`, on the full training months (no subsampling needed: ~40s per inner fit, ~30 min total).
 
**Base estimator is wrapped (`FixedFPRXGB`).** The reduction enforces the constraint on each inner estimator's `predict()`. XGBoost's `predict()` fires at probability 0.5; at 1.1% prevalence almost nothing crosses 0.5, so every component would predict "not fraud", both groups would show ~0% FPR, and the constraint would be satisfied trivially by a model that flags nobody. The wrapper makes `predict()` fire at the 5% FPR operating point instead, so the constraint is enforced where decisions are actually made. This is a deviation from the textbook reduction and must be stated in Method.
 
**Scoring the mixture.** ExpGrad returns ~40 predictors plus weights. `eg.predict()` samples one at random (not reproducible); `_pmf_predict()` averages 0/1 votes and yields a coarse score unusable at a 5% cutoff. We score with the **weighted average of component probabilities**, which is continuous and comparable to the baseline. Val score spread (min/median/95th/99th/max): 0.000 / 0.008 / 0.120 / 0.282 / 0.912.
 
Note: `ExponentiatedGradient.fit()` forwards extra kwargs to the constraint's `load_data()`, which accepts only `sensitive_features`. Any subsampling correction must be baked into the estimator (`row_weights`), not passed as `sample_weight`.
 
### Results (difference_bound = 0.01)
 
| Split | Group | FPR | Recall |
|---|---|---|---|
| val | 0 (under 50) | 0.0586 | 0.5494 |
| val | 1 (50-plus) | 0.0083 | 0.2085 |
| test | 0 | 0.0708 | 0.6003 |
| test | 1 | 0.0132 | 0.2649 |
 
Compared with the baseline:
 
| Metric (val) | Baseline | ExpGrad |
|---|---|---|
| FPR, under 50 | 0.0368 | 0.0586 |
| FPR, 50-plus | 0.1135 | 0.0083 |
| FPR ratio (lower/higher) | 0.32 | **~0.14** |
| Recall, 50-plus | 0.7234 | 0.2085 |
| Recall, overall | 0.5478 | ~0.42 |
 
**The gap did not close — it reversed and widened.** Under-50s are now wrongly flagged ~7x more often than 50-plus. By our own metric the intervention is worse than doing nothing, and it cost ~13pp of overall recall. 50-plus fraud recall collapsed from 0.72 to 0.21: roughly four in five older fraudsters now pass through.
 
### Diagnosis (likely, not confirmed)
 
The constraint and the deployed decision rule are not the same object. ExpGrad equalised FPR over **each component's own 0/1 decisions on the training months**; we then discarded those decisions, averaged the components' probabilities, and applied **one global cutoff on val**. Parity is not preserved across that change of decision rule. To equalise its own decisions, the reduction pushed 50-plus scores down; under a single shared cutoff those scores land far below it.
 
Secondary suspect: each component computes `cut_` in-sample on the data it just fitted, so its 5% FPR estimate is optimistic.
 
### Open question — resolve before writing this up
 
1. **Sweep `difference_bound`** (0.001, 0.01, 0.05, 0.10), one predictions file each, e.g. `preds_expgrad_db0.05.parquet`. This traces the fairness/recall frontier we need for the trade-off curves anyway, and shows whether a looser bound stops short of overshooting.
2. **Diagnostic only:** check `eg._pmf_predict(Xva)[:, 1] >= 0.5`. If FPRs are near-equal there, the diagnosis above is confirmed. Not usable as a headline result (it cannot hit a 5% FPR target, so it is not comparable to the other models).
3. Optional: compute each component's `cut_` on a held-out month inside the wrapper rather than in-sample.
If the sweep shows a looser bound behaves sensibly, the finding is **"this method is highly sensitive to the gap between the point where fairness is enforced and the point where decisions are made"** — not "the method fails". Frame it that way; the stronger claim is not supported by one configuration.
 
Either way this is reportable and on-topic: the project asks whether fairness interventions deliver what they promise, and this is direct evidence that a published method can satisfy its stated constraint while making the deployed outcome worse. Jianrong should run SHAP on this model too — what an overcorrected model relies on is worth seeing.
 
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
