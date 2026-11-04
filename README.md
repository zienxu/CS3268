# CS3268 Track 1 — Does Fixing Fairness Remove Bias, or Just Move It?

Bank account opening fraud detection on the BAF dataset, with fairness fixes (reweighing, group thresholds, ExponentiatedGradient) audited by TreeSHAP for age-proxy relocation.

Team: Zien Xu (data, baseline, integration) · Chloe (fairness metrics and fixes) · Jianrong (interpretability audit) · Sze Ling (report, figures, slides)

---

## Key results and talking points (slide-ready; updated 4 Nov 2026)

Status: baseline, ExpGrad (canonical file = bound 0.05) and reweighing done. Group thresholds, bootstrap CIs and the TreeSHAP audit are still to run (target: before the 8 Nov code freeze). Fill the TBD rows as results land; every number must match a `preds/` file.

**Headline finding.** Our fraud model wrongly flags honest 50-plus applicants about 3x as often as honest under-50s (FPR ratio 0.32 val, 0.33 test), at 5% overall FPR.

**Results by fix (val FPR ratio, lower/higher; 1.0 = parity).**

| Fix | Where it acts | Val ratio | Overall recall | Takeaway |
|---|---|---|---|---|
| Baseline | n/a | 0.32 | 0.548 | Large, stable gap |
| ExpGrad (bound 0.05) | During training | 0.48 | 0.549 | Partial fix, no recall cost; 50-plus still ~2x FPR |
| ExpGrad (bound 0.01) | During training | 0.14 | 0.436 | Overshoots: under-50s now flagged more, ~11pp recall lost |
| Reweighing | Before training | 0.98 (test 0.90) | 0.527 | Near-parity on FPR; costs ~2pp overall recall (val), ~3pp (test) |
| Group thresholds | After scoring | TBD | TBD | TBD |

**ExpGrad one-liner.** A published method can satisfy its own constraint while the deployed outcome gets worse: fairness was enforced on randomized 0/1 votes, but decisions use one cutoff on averaged probabilities. Sensitive to where fairness is enforced vs where decisions are made, not "the method fails".

**Other talking points.**
- Test FPR is above 5% for every model (6.22% baseline): distribution shift, not a bug.
- Higher base rates (2.34% vs 0.83%) explain the direction of the gap but do not justify it (FPR is measured on innocent people only).
- Point estimates only until Chloe's bootstrap CIs are in; ±1.3pp SE on recall.
- Proxy audit: TreeSHAP on baseline vs fixed models, using the frozen 6-feature proxy set (one strong proxy: date_of_birth_distinct_emails_4w, AUC 0.81). Results TBD.

**Limitations / Future Work.** Variant II robustness check not run (cut for time). Finer ExpGrad bound grid between 0.01 and 0.05 untested. Component cutoffs computed in-sample.

---

## Setup (do this once)

1. Open Google Drive → **Shared with me** → right-click `CS3268_Project` → **Organize → Add shortcut** → choose **My Drive**. Keep the name exactly `CS3268_Project`. Without this, the paths below will not work on your account.
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

Data and outputs live in Google Drive (`MyDrive/CS3268_Project/`). Code lives in this GitHub repo. Never commit `.csv` or `.parquet` files.

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

**Checkpoint (Sun 4 Oct): PASSED.** The gap is large and stable; we stay on Base. Variant II remains a robustness check only and was cut to Future Work (4 Nov).

---

## Intervention 1: ExponentiatedGradient (W2, run 23 Sep; sweep 23 Oct)
 
Notebook: `notebooks/02_expgrad_py.ipynb` (sweep: `notebooks/02b_expgrad_sweep.ipynb`) · Outputs: `preds/preds_expgrad.parquet`, `models/expgrad_components/`, `models/expgrad_weights.csv`
 
Status: **sweep complete (4 bounds, 23 Oct). Winner by the pre-registered rule: `difference_bound=0.05`.** **Done (4 Nov): repro check OK, cell 8 run with `CONFIRM=True`; canonical `preds_expgrad.parquet` is now the 0.05 winner.** The original 23 Sep 0.01 files are kept as `*_prelim_0923`.
 
### Why ExpGrad, and how it works (plain English)

**Why we chose it.** *(Reconstructed from the project design, not from a recorded team decision. Team to confirm before this goes in the report.)* The three fixes sit at three different points in the pipeline: reweighing changes the training data (before), group thresholds change the decision cutoffs (after), and ExpGrad changes how the model is trained (during). Together they let us ask whether the fix's location matters. ExpGrad is also a published method (Agarwal et al. [3]) with a built-in FPR-parity constraint, which is the same metric as our headline gap, so "does the fix deliver what it promises?" is testable directly. Fairlearn provides it off the shelf.

**How it works.** *(Based on the paper; the exact weight and label mechanics in Fairlearn are not yet checked against its docs.)*
1. We give it a base model (our XGBoost wrapper) and a fairness rule: "the groups' FPRs may differ by at most the bound."
2. It trains the model once. Then it checks which group the model is treating worse and builds a "penalty" for that.
3. It turns the penalty into new example weights (and possibly flipped labels), so the next model concentrates on fixing that group's errors. It trains another model on that.
4. It repeats this for `max_iter` rounds. Each round it raises the penalty on whichever group is currently more out of line, and eases off on the other.
5. Its final answer is not one model but a weighted mix of all the models it trained. In our runs only 1–2 of them ended up with non-zero weight.

The key point for us: it makes the *mix of 0/1 decisions* satisfy the rule, which is not the same thing as the single-cutoff score we deploy. See "Diagnosis" below.

### Setup
 
Fairlearn's implementation of the reductions approach of Agarwal et al. [3], with a `FalsePositiveRateParity` constraint, `max_iter=40`, on the full training months (no subsampling needed: ~40s per inner fit, ~30 min total).
 
**Base estimator is wrapped (`FixedFPRXGB`).** The reduction enforces the constraint on each inner estimator's `predict()`. XGBoost's `predict()` fires at probability 0.5; at 1.1% prevalence almost nothing crosses 0.5, so every component would predict "not fraud", both groups would show ~0% FPR, and the constraint would be satisfied trivially by a model that flags nobody. The wrapper makes `predict()` fire at the 5% FPR operating point instead, so the constraint is enforced where decisions are actually made. This is a deviation from the textbook reduction and must be stated in Method.
 
**Scoring the mixture.** ExpGrad returns ~40 predictors plus weights. `eg.predict()` samples one at random (not reproducible); `_pmf_predict()` averages 0/1 votes and yields a coarse score unusable at a 5% cutoff. We score with the **weighted average of component probabilities**, which is continuous and comparable to the baseline. Val score spread (min/median/95th/99th/max): 0.000 / 0.008 / 0.120 / 0.282 / 0.912.
 
Note: `ExponentiatedGradient.fit()` forwards extra kwargs to the constraint's `load_data()`, which accepts only `sensitive_features`. Any subsampling correction must be baked into the estimator (`row_weights`), not passed as `sample_weight`.
 
### Plain-English guide: the bound and the "coarse score"

**What `difference_bound` means.** It is how strict we are about fairness. We tell ExpGrad: "the FPR of the two age groups may differ by at most this much." Think of it as a speed limit.

| Bound | Meaning | What happened |
|---|---|---|
| 0.001 | Groups' FPRs must be within 0.1pp. Very strict | Overshot: gap reversed, under-50s flagged more |
| 0.01 | Within 1pp. Strict | Overshot, same as 23 Sep |
| 0.05 | Within 5pp. Moderate | Useful middle: partial fix, no recall cost |
| 0.1 | Within 10pp. Loose | Baseline's own gap is only ~7.7pp (0.1135 − 0.0368), so the limit is above our current speed and never binds. Result = baseline |

**Why it matters.** The bound is the dial that trades fairness against detection, so sweeping it gives the trade-off curve. It did not behave smoothly: strict settings overcorrected and only one setting was useful. That sensitivity is the finding.

**"Averages 0/1 votes and yields a coarse score."** ExpGrad returns a handful of models with weights. Each makes a hard 0/1 call per applicant; the weighted average of those calls is the score. With 2 models weighted 60/40, every applicant gets one of only four scores (0, 0.4, 0.6, 1): lots of ties, no fine ranking.

**Why that is unusable at a 5% cutoff.** The 5% FPR cutoff needs a fine ranking so we can draw the line at the 95th percentile of honest applicants. With only a few distinct scores the line falls on a step: if 30% score 0 and 20% score 0.4, we can flag 0% or 20%, never 5%. The result would also not be comparable to the other models. So we average each component's *probability* instead, which is continuous.

**The catch.** ExpGrad enforced fairness on the 0/1 votes, but we deploy the smooth probabilities with one global cutoff. These are different decision rules, and parity on one does not carry over to the other. This is the "decision-rule switch" in the diagnosis below.

*Example (two models, weights 0.6 and 0.4; each fires at its own 5%-FPR cutoff, say A at 0.15 and B at 0.25; two applicants).*

| | Model A prob | Model B prob | A's 0/1 call | B's 0/1 call | Score from votes (before) | Score from probabilities (what we use) |
|---|---|---|---|---|---|---|
| Applicant 1 | 0.12 | 0.30 | 0 (0.12 < 0.15) | 1 (0.30 ≥ 0.25) | 0.6 × 0 + 0.4 × 1 = **0.4** | 0.6 × 0.12 + 0.4 × 0.30 = 0.072 + 0.120 = **0.192** |
| Applicant 2 | 0.14 | 0.29 | 0 (0.14 < 0.15) | 1 (0.29 ≥ 0.25) | 0.6 × 0 + 0.4 × 1 = **0.4** | 0.6 × 0.14 + 0.4 × 0.29 = 0.084 + 0.116 = **0.200** |

With votes the two applicants tie at 0.4, and so does everyone else who gets the same A/B calls. With probabilities they are ranked apart (0.192 vs 0.200), so a 5% cutoff can be placed precisely. (Code: `mixture_score` in notebook 02b.)

### Sweep results (run 23 Oct 2026; pre-registered below)

Refit of 0.01 reproduces the 23 Sep run (val FPR 0.0586 / 0.0083, recall 0.5494 / 0.2085; test FPR 0.0708 / 0.0132). Cell-5 repro check output: **confirm it printed OK before finalising.** Fits took 11–21 min each.

| Bound | Non-zero components | Val FPR 0 / 1 | Val ratio | Val recall (overall / 50+) | Test ratio | Test recall (overall) | Flagged more |
|---|---|---|---|---|---|---|---|
| 0.001 | 2 of 5 | 0.0594 / 0.0048 | 0.081 | 0.4274 / 0.1723 | 0.117 | 0.4635 | under-50 |
| 0.01 | 2 of 5 | 0.0586 / 0.0083 | 0.142 | 0.4359 / 0.2085 | 0.187 | 0.4851 | under-50 |
| **0.05** | 2 of 7 | 0.0422 / 0.0874 | **0.483** | 0.5485 / 0.6681 | 0.467 | 0.5823 | 50-plus |
| 0.1 | 1 of 6 | 0.0368 / 0.1135 | 0.325 | 0.5478 / 0.7234 | 0.325 | 0.5893 | 50-plus |
| baseline | – | 0.0368 / 0.1135 | 0.325 | 0.5478 / 0.7234 | 0.325 | 0.5893 | 50-plus |

**"The ExpGrad result" (pre-registered rule: val FPR ratio closest to 1.0) = bound 0.05.** Val ratio 0.32 → 0.48 (test 0.33 → 0.47) with no change in overall recall (0.5485 vs 0.5478, well inside ±1.3pp SE). 50-plus recall falls 0.72 → 0.67, under-50 recall rises 0.46 → 0.49. This is a partial fix only: 50-plus applicants are still wrongly flagged ~2x as often (val FPR 0.087 vs 0.042; test 0.109 vs 0.051). Treat the 0.32 → 0.48 change as a point estimate until Chloe's bootstrap CIs are in.

Observations:
- **Tight bounds (0.001, 0.01) overshoot:** the gap reverses (under-50s worse) and overall recall drops ~11pp.
- **Loosest bound (0.1) is the baseline:** one non-zero component, i.e. the constraint never binds; results are identical to baseline to 4 d.p.
- **The curve is not monotone in the bound:** the ratio crosses from under-50-worse (0.01) to 50-plus-worse (0.05), so parity lies somewhere between. Not tested (outside the pre-registered grid); a finer grid is Future Work, or exploratory if run before code freeze.
- Only 1–2 components carry weight, so the "mixture" is effectively two models.
- Test FPR exceeds 5% for every bound (distribution shift, as for the baseline).

### Diagnosis (supported by the sweep)

The constraint and the deployed decision rule are not the same object. ExpGrad equalised FPR over **each component's own randomized 0/1 decisions on the training months**; we then averaged the components' probabilities and applied **one global cutoff on val**. Parity is not preserved across that change of decision rule.

Evidence (randomized-classifier FPR ratio, lower/higher, vs deployed val ratio):

| Bound | Randomized, train | Randomized, val | Deployed, val |
|---|---|---|---|
| 0.001 | 0.976 | 0.843 | 0.081 |
| 0.01 | 0.795 | 0.968 | 0.142 |
| 0.05 | 0.384 | 0.473 | 0.483 |
| 0.1 | 0.258 | 0.309 | 0.325 |

At 0.001 and 0.01 ExpGrad's own rule is near parity on both train and val while the deployed rule is far from it, so the failure comes from the change of decision rule. The secondary suspect (in-sample `cut_` optimism) is ruled out: the randomized ratio is also near 1 on val. At 0.05 and 0.1 the randomized rule is itself unequal, so the bound is not binding there and the switch is not what limits them.

**Framing for the report:** ExpGrad is highly sensitive to the gap between where fairness is enforced and where decisions are made. Tight bounds overshoot, the loosest does nothing, and only an intermediate bound gives a useful, still partial, improvement. Not "the method fails". This is direct evidence that a published method can satisfy its stated constraint while making the deployed outcome worse. Jianrong should run SHAP on the winner and on an overcorrected model (0.01).

Optional, not done: compute each component's `cut_` on a held-out month rather than in-sample.

### Sweep pre-registration (stated 23 Oct 2026, before any sweep run)

Bounds: `difference_bound` in [0.001, 0.01, 0.05, 0.1], `max_iter=40`, all else unchanged. The 0.01 point is refit (the original runtime is gone) and checked against the 23 Sep file.

- **All four bounds are always reported** as the fairness/recall trade-off curve, whatever the outcome.
- **The single number called "the ExpGrad result"** is the bound whose **val FPR ratio (lower/higher) is closest to 1.0**.
- **Why this differs from the baseline rule** (highest recall at 5% FPR): that rule picks the most accurate model among candidates with no fairness trade-off. Here the candidates sit on a fairness/accuracy frontier, so "most accurate" and "best" are no longer the same thing, and reusing it would mechanically pick the loosest bound.
- **Diagnostic, every bound:** per-group FPR of ExpGrad's own randomized classifier (mean of `_pmf_predict(X)[:,1]` over honest rows) on train and val, next to the deployed single-cutoff FPRs.

---

## Intervention 2: Reweighing (run 4 Nov 2026)

Notebook: `notebooks/04_reweigh.ipynb` · Outputs: `preds/preds_reweigh.parquet`, `models/reweigh.json`

Status: **done 4 Nov (scheme fixed before running).**

**Scheme (as taught in the course; Kamiran & Calders 2012 [cite]).** Each training row gets weight W(y,a) = P_exp(Y=y, A=a) / P_obs(Y=y, A=a), with P_exp = P(y)·P(a) (the joint if label and age group were independent), all computed on train months 0-4. W = 1 means that cell is already independent; W > 1 upweights it, W < 1 downweights it. Applied via `train_model(X, y, sample_weight=w)` with frozen `BEST_PARAMS`, seed 42. Full strength, single run, no sweep, no tuning. Cutoff set at 5% FPR on val, applied unchanged to test.

**Weights (train).** W(under-50, honest) 0.997 · W(under-50, fraud) 1.395 · W(50-plus, honest) 1.012 · W(50-plus, fraud) 0.454; mean 1.000. 50-plus has the higher fraud rate, so its fraud rows are downweighted and under-50 fraud rows upweighted, as expected. The weights target independence of label and age group (demographic parity of the label), not FPR parity directly.

**Results** (cutoff set on val; test uses the same cutoff).

| Split | Group | FPR | Recall | n |
|---|---|---|---|---|
| val | 0 | 0.0502 | 0.5292 | 98,598 |
| val | 1 | 0.0490 | 0.5234 | 20,725 |
| val | all | 0.0500 | 0.5273 | 119,323 |
| test | 0 | 0.0591 | 0.5611 | 171,232 |
| test | 1 | 0.0657 | 0.5521 | 33,779 |
| test | all | 0.0602 | 0.5580 | 205,011 |

**FPR ratio (lower/higher): 0.976 val, 0.90 test** (baseline 0.32 / 0.33). On val the two groups are at parity (the 50-plus group is marginally lower; the difference is noise). On test 50-plus is flagged slightly more (0.066 vs 0.059). Group recall also converges (0.53 vs 0.52 on val; baseline 0.46 vs 0.72).

**Cost.** Overall recall at 5% FPR falls 0.548 → 0.527 on val (-2.1pp) and 0.589 → 0.558 on test (-3.1pp). Against ±1.3pp SE this is a modest, not clearly significant, cost until Chloe's bootstrap CIs are in. Test FPR is 6.02% (baseline 6.22%), again above 5% (distribution shift).

**Reading.** Unlike ExpGrad, which enforced fairness on one decision rule and deployed another, reweighing changes the data the single scored model learns from, so the deployed rule is the one that was fixed. Likely mechanism: with age and fraud decoupled in the weighted data, the model stops treating "older" as riskier, so 50-plus recall falls (0.72 → 0.52) and under-50 recall rises (0.46 → 0.53). Whether the model now leans on age proxies instead is the open question for the TreeSHAP audit (frozen proxy set). Single run, point estimates; no sweep, so the strength of the fix was not tuned.

## Proxy set (frozen 4 Nov 2026)

File: `models/proxy_set.json` · Notebook: `notebooks/03_proxy_set.ipynb`

**What it is.** After a fairness fix, the model may stop using age directly and lean on other features that carry the same information. The TreeSHAP audit checks whether attribution moves onto these "age proxies". The proxy set is the list of features we agree, in advance, count as proxies.

**Why we need it.** The project asks whether a fix removes bias or just moves it. SHAP alone shows which features the model relies on, not which of them are linked to age, so a shift in attribution cannot be called "bias moving to a proxy" without an independent definition of proxy. The proxy set is that definition. It is fixed in advance and independent of every fix, so it targets the "moved it" half of the question and prevents choosing proxies after seeing where attribution went. A descriptive SHAP-only table (top features gaining or losing attribution per fix) is reported alongside as a second view.

**Rule (written before seeing any output).** For each feature (excluding `customer_age`), fit a small decision tree (depth 3) on that feature alone to predict `age_group`, using train months 0-3, and score it on train month 4. AUC measures how well the feature alone separates 50-plus from under-50 (0.5 = no information, 1.0 = perfect; folded so direction does not matter). **Proxy iff AUC >= 0.60.**

**Result.**

| Feature | AUC | In set |
|---|---|---|
| date_of_birth_distinct_emails_4w | 0.806 | yes (strong) |
| housing_status | 0.657 | yes |
| current_address_months_count | 0.628 | yes |
| phone_home_valid | 0.621 | yes |
| employment_status | 0.604 | yes (borderline) |
| device_os | 0.601 | yes (borderline) |
| prev_address_months_count | 0.573 | no (next closest) |

All 29 AUCs are stored in `proxy_set.json`. Most features sit at 0.50-0.57, i.e. carry little age information.

**How to read it.** One feature is a strong age proxy (0.81); five are weak-to-moderate; the rest are not. The 0.60 cutoff is a judgement call: employment_status and device_os are barely over it and prev_address_months_count barely under. We do not move the cutoff after the fact; the report should say so and note the sensitivity.

**Method provenance and defence (for Q&A and the report).**
- *Idea.* A proxy is a feature that leaks information about the protected attribute, so we measure how well each feature predicts `age_group`. This follows the established idea in the proxy-discrimination literature (as recalled, not yet verified against the papers: Feldman et al. 2015, "Certifying and Removing Disparate Impact"; Datta et al. 2017, "Proxy Non-Discrimination in Data-Driven Systems". Check both before citing).
- *Why AUC as the measure.* (1) Robust to class imbalance (50-plus is ~17% of applicants, so accuracy would reward always guessing under-50). (2) Threshold-free: it scores ranking across all cutoffs, so no second arbitrary cutoff is needed to evaluate a feature. (3) With a small tree, it catches nonlinear and categorical relationships that correlation misses. (4) Scored on a held-out month, so overfitting cannot qualify a feature. (5) Interpretable (0.5 = chance, 1.0 = perfect). It shows a feature *could* stand in for age, not that the model *uses* it that way, which is what the SHAP audit tests. Limits: no standard says AUC is the metric for proxy detection (Feldman et al. use balanced error rate, as recalled; closely related, not identical); mutual information or correlation would also be defensible alternatives.
- *Our version is a simplification, not a named standard.* We screen features one at a time (depth-3 tree, held-out month, AUC folded as max(auc, 1-auc)). The published approaches typically use all features jointly.
- *The 0.60 cutoff is a heuristic.* AUC 0.5 is chance; 0.60 is a common rule of thumb for "weak but real". It has no citation. It was fixed before output was seen, and we report sensitivity instead of defending it as correct (sensitivity table to be added as a labelled post-freeze analysis; at the frozen AUCs the set has 12 features at 0.55, 6 at 0.60, 2 at 0.65, 1 at 0.70).
- *Limitation.* Single-feature screening can miss joint proxies, where several individually weak features together predict age. The set may therefore understate proxying. Optional post-freeze extra: one model on all non-age features predicting `age_group`; report its AUC as total leakage.
- *Use.* The set is used only in the TreeSHAP audit (share of attribution on proxy features, compared across models). It does not enter training, thresholds or fairness metrics.
- *Timeline.* The proxy audit was in the design from 21 Sep and `.gitignore` allowed the file from 23 Sep; the actual rule and file date from 4 Nov. ExpGrad (23 Sep, 23 Oct) was trained before the freeze, but its attributions were not examined.

**Timing claim for Method.** Frozen on 4 Nov from train data only, before any attribution analysis of a fixed model and before reweighing and group thresholds were run. ExpGrad had been trained earlier (23 Sep, 23 Oct) but its attributions were not examined.

## Rules

- **Seed 42** everywhere (`random_state=42`, `np.random.seed(42)`).
- **Edit only your own notebook.** Shared code changes go through Zien Xu.
- **Never overwrite someone else's file** in `preds/` or `models/`.
- **The proxy set is frozen** in `proxy_set.json` by a dated commit, written from baseline-only evidence, before any attribution analysis of a fixed model and before reweighing/group thresholds run. Do not edit it afterwards. (ExpGrad was trained before the freeze but not analysed; state this in Method.)
- **Test data is for final numbers only.** No tuning or threshold choices on months 6–7.
- **Record new dependencies** in `requirements.txt`.
- **Code freeze: Sun 8 Nov** (moved from 1 Nov). No new experiments after that date.
- **`BEST_PARAMS` is frozen.** All interventions call `train_model()` with it. Do not re-tune per intervention.
- **Model selection rule was fixed in advance**: highest recall at 5% FPR on val. Do not switch to a different criterion after seeing results.

## Method notes for the report

- The val month (5) is used for three things: early stopping, hyperparameter selection, and cutoff selection. Test months (6–7) are untouched until final numbers. Say this explicitly in Method so it does not read as an oversight.
- Cutoffs are always set on val and applied unchanged to test, mirroring how a bank sets a policy on past data and applies it to new applicants.
- Test FPR drifting above the 5% target is a reportable finding about distribution shift, not a bug.
