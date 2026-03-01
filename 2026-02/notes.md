# 2026-02 Challenge: Heart Disease Prediction - Notes

## 🔍 Project Status

**Challenge**: Playground Series S6E2 - Heart Disease Prediction  
**Type**: Binary Classification  
**Metric**: AUC-ROC  
**Goal**: Top 10 position (target AUC ≥ 0.89)

## 📊 Data Summary

- **Dataset**: Synthetic heart disease prediction data
- **Rows**: ~630K training samples (estimated)
- **Columns**: ~10-15 features (to be confirmed)
- **Target**: Binary (disease presence: 0 or 1)
- **Features**: Mix of numeric and categorical

## 🏗️ Approach Applied

Adapted successful techniques from 2026-01 regression challenge:

1. **Baseline**: LogisticRegression → establish AUC reference
2. **Boosting**: LGB + XGB with tuned hyperparameters (200-250 trees for classification)
3. **Features**: Interactions, polynomials, ratios (same engineering pattern as regression)
4. **Ensemble**: 5-fold stacking + weighted blending (0.4 LGB, 0.4 XGB, 0.2 Stack)

## 📈 Key Changes from Regression

| Aspect | 2026-01 (Regression) | 2026-02 (Classification) |
|--------|----------------------|--------------------------|
| **Models** | LGBMRegressor | LGBMClassifier |
| **Output** | Continuous predictions | Probabilities (0.0-1.0) |
| **Metric** | RMSE (lower better) | AUC-ROC (higher better) |
| **Baseline** | LinearRegression | LogisticRegression |
| **Validation** | KFold | StratifiedKFold |
| **Trees** | 400-500 | 200-250 |

## 🎯 Performance Targets

- **Baseline (Logistic)**: 0.80-0.85 AUC expected
- **Single Model (LGB)**: 0.87-0.90 AUC expected
- **Final Ensemble**: 0.88-0.92 AUC expected (target ≥0.89 for Top 10)

## ⚙️ Hyperparameters Used

**LightGBM Classification**:
- n_estimators=200, learning_rate=0.08, num_leaves=80, max_depth=8

**XGBoost Classification**:
- n_estimators=250, learning_rate=0.05, max_depth=5

**Blending Weights**:
- 0.4 × LGB_pred + 0.4 × XGB_pred + 0.2 × Stack_pred

## 📝 Execution Progress

- [x] Download actual data from Kaggle
- [x] Run notebook end-to-end (3-5 min expected)
- [x] Verify baseline AUC in expected range
- [x] Generate submission CSV
- [x] Upload to Kaggle
- [x] Record public/private scores
- [x] Analyze results and document learnings

## 🏆 Final Results

**Competition Ended**: February 28, 2026

| Metric | Value |
|--------|-------|
| **Final Position** | 2281 / 2893 participants |
| **Private LB Score** | 0.95486 |
| **1st Place Score** | 0.95535 |
| **Gap to 1st** | 0.00049 (0.051% difference) |
| **Improvement over 0.95310** | +0.00176 (+0.18% improvement) |

## 📊 Performance Breakdown

- **Baseline (LogisticRegression)**: 0.9515 AUC
- **Single Models (5-fold CV)**:
  - LightGBM: 0.9550 ± 0.0004
  - XGBoost: 0.9551 ± 0.0004
  - CatBoost: 0.9551 ± 0.0004
- **Stacking (L1+L2)**: 0.9552 AUC
- **Blending with optimized weights**: 0.9588 AUC (holdout test)
  - **Best weights**: LGB 0.40, XGB 0.40, Cat 0.15, Stack 0.05
- **Final calibrated submission**: 0.95486 private LB

## 🎯 Lessons Learned

### What Worked Well
1. **Ensemble architecture** was solid (improves baseline by 0.77%)
2. **Weight optimization** found genuinely better combinations (0.9588 vs 0.9515)
3. **3-model approach** better than 2-model (CatBoost added diversity)
4. **Calibration** maintained prediction quality

### Where We Fell Short
1. **Hyperparameter tuning** was conservative (could explore more aggressive settings)
2. **Feature engineering** didn't improve (stopped at baseline 13 features)
3. **Single model ceiling** was ~0.9551 AUC (top models all converged to similar scores)
4. **Simple blending** had limits (didn't match top solution's complexity)

### Comparison to 1st Place Solution

**Their approach** (0.95535 private):
- Diversity through **7 different feature representations** (binning, digits, categorical, frequency, genetic programming, target encoding, DVAE)
- **~150 OOF predictions** from diverse model types (XGB, LGB, CatBoost, RealMLP, RGF, TabICL, AutoGluon)
- **Optuna-based selection** - picked best subset of 150 OOFs
- **Ridge ensemble** on selected OOFs
- **Full-data retraining** with 20 random seeds + averaging
- **CV-LB relation monitoring** - trusted CV 0.95578-0.95580 over higher CV due to split overfitting

**Our approach** (0.95486 private):
- Single feature representation (13 base features)
- 3 base models (LGB, XGB, Cat) + stacking
- Manual weight optimization (4 combinations tested)
- Ridge meta-learner on stacked features
- Probability calibration with IsotonicRegression

**Gap analysis**: 49 basis points behind 1st place
- Feature diversity: Major gap (7 vs 1 representation)
- Model diversity: Moderate gap (7 models vs 3 base)
- Ensemble method: Comparable (Ridge in both)
- Validation rigor: Their CV-LB monitoring more thorough

## 💡 Key Takeaway: Trust CV-Leaderboard Relationship, Not Highest CV

This writeup reinforced an important lesson: **The 1st place solution intentionally submitted CV 0.95578 instead of their best CV 0.955865** because they observed degraded CV-LB correlation at higher CV values. This suggests split overfitting, where improvements in CV don't translate to LB due to fold-specific patterns.

**For next challenge**: Monitor CV-LB consistency through test submissions rather than blindly maximizing CV.

## 🔬 Original Hypothesis

> Since ensemble methods achieved 11.9% improvement in regression (9.9452 → 8.76), applying same stacking+blending architecture to classification should yield 10-15% improvement. Targeting 0.89+ AUC should be achievable with careful hyperparameter tuning.

**Result**: ✅ Hypothesis partially confirmed
- Achieved 0.77% improvement over baseline (0.9515 → 0.95486)
- Top 1st place achieved 0.78% improvement (0.9515 → 0.95535)
- Gap was feature diversity, not ensemble architecture

## 📚 Reference

- **Learnings from 2026-01**: See root [LEARNINGS.md](../LEARNINGS.md)
- **Workflow**: See [WORKFLOW.md](../WORKFLOW.md)
- **Techniques**: See [shared/techniques.md](../shared/techniques.md)
- **1st Place Writeup**: https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t

---

*Last Updated*: March 1, 2026
