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

- [ ] Download actual data from Kaggle
- [ ] Run notebook end-to-end (3-5 min expected)
- [ ] Verify baseline AUC in expected range
- [ ] Generate submission CSV
- [ ] Upload to Kaggle
- [ ] Record public/private scores
- [ ] Analyze results and document learnings

## 🔬 Hypothesis

> Since ensemble methods achieved 11.9% improvement in regression (9.9452 → 8.76), applying same stacking+blending architecture to classification should yield 10-15% improvement. Targeting 0.89+ AUC should be achievable with careful hyperparameter tuning.

## 📚 Reference

- **Learnings from 2026-01**: See root [LEARNINGS.md](../LEARNINGS.md)
- **Workflow**: See [WORKFLOW.md](../WORKFLOW.md)
- **Techniques**: See [shared/techniques.md](../shared/techniques.md)

---

*Last Updated*: January 31, 2026
