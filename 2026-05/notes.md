# 2026-05 Notes - Predicting F1 Pit Stops

Challenge: Binary classification (`PitNextLap`), metric is AUC-ROC.

## Objective

Build a strong, interview-friendly V1 in about 1 hour:
- ETL and data quality checks
- Focused EDA for race strategy intuition
- Baseline model
- Improved model
- Kaggle-ready submission

## V1 Results

- Baseline Logistic Regression (5-fold): 0.83639 mean AUC
- LightGBM (5-fold OOF): 0.94832 AUC
- Improvement over baseline: +0.11193 AUC

Submission files generated:
- `submissions/submission_v1_baseline_logreg.csv`
- `submissions/submission_v1_lgbm.csv`
- `submissions/submission.csv`

## What Worked

1. Domain features helped:
- `TyreLifePct`
- `TyreOverExpected`
- `DegPerLap`
- `TyreAge_x_Deg`
- `Progress_x_TyreLife`

2. LightGBM captured nonlinear pit dynamics much better than linear baseline.

3. Fold stability was good in V1 (tight fold AUC spread around 0.948).

## Issues Encountered and Fix

Problem:
- Notebook failed during scaling with: could not convert string to float (`D109`).

Root cause:
- One or more categorical columns were still non-numeric before `StandardScaler`.

Fix applied:
- Explicitly encode all object/string/category columns.
- Add a final safety pass for any remaining non-numeric columns.
- Coerce numeric with `errors="coerce"` and median-fill NaNs if needed.

## Interview/Presentation Talking Points

1. Why baseline first:
- Validates pipeline and gives an interpretable benchmark.

2. Why LightGBM improved:
- Captures nonlinear interactions between tyre wear, pace trend, and race phase.

3. Why AUC:
- Better metric for imbalanced binary classification than accuracy.

4. Reliability lesson:
- Strong modeling fails if preprocessing is not type-safe.

## V2 Experiment Ladder (Prioritized)

1. Validation robustness:
- Try race-aware splits (GroupKFold by race or race-year) and compare with stratified CV.

2. CatBoost branch:
- Use native categorical handling and compare OOF vs LightGBM.

3. Calibration:
- Evaluate isotonic or Platt scaling for potential leaderboard gains.

4. Small Optuna pass:
- 30-60 trials for LightGBM/CatBoost with early stopping.

## V2 Run (Implemented)

Enhancements implemented in notebook:
- Driver x Race target encoding (OOF-smoothed TE)
- Driver x Race frequency encoding
- Causal time-window features (lag and rolling by Year/Race/Driver):
	- `LapTimeDelta_prev1`
	- `LapTimeDelta_roll3_mean`
	- `DegPerLap_roll3_mean`
	- `PosChange_roll3_sum`
- Optuna tuning for LightGBM (12 trials, 3-fold objective)

### V2 Metrics

- Baseline Logistic Regression (with V2 matrix): **0.84676** mean AUC
- Tuned LightGBM OOF AUC: **0.94787**

Comparison to V1 LightGBM:
- V1 OOF: 0.94832
- V2 OOF: 0.94787
- Delta: **-0.00045**

Interpretation:
- V2 features + compact tuning improved baseline but did not beat V1 LightGBM OOF yet.
- This likely means current additions increased complexity/noise relative to signal under this CV split.

## Next Actions After V2

1. Increase Optuna budget to 40-60 trials while narrowing ranges around best params.
2. Validate Driver x Race TE utility with ablation:
	 - only TE,
	 - only time windows,
	 - both together.
3. Evaluate race-aware validation (GroupKFold by race-year) to check CV realism.
4. Add CatBoost native-categorical branch for a stronger comparison.

## Working Rule

Prefer reliability-first improvements (clean preprocessing, leakage checks, stable CV) before complex ensembling.
