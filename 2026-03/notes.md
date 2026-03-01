# 2026-03 Challenge: Customer Churn Prediction - Notes

## 🔍 Project Status

**Challenge**: Playground Series S6E3 - Predict Customer Churn  
**Type**: Binary Classification  
**Metric**: AUC-ROC (area under receiver operating characteristic curve)  
**Timeline**: March 1 - March 31, 2026 (1 month)

## 📊 Data Summary

- **Dataset**: Synthetic customer churn data from telecom customer behaviors
- **Training**: 594,194 samples with 19 features
- **Test**: 254,655 samples to predict
- **Features**: Mix of categorical (gender, Partner, InternetService, etc.) and numeric (tenure, MonthlyCharges, TotalCharges)
- **Target**: Binary Churn (0/1)
- **Class Distribution**: (to be analyzed in notebook)

## 🏗️ Approach Strategy

### Phase 1: Initial Ensemble (Current)

Adapted from successful 2026-02 with enhancements:

1. **Baseline**: LogisticRegression → AUC reference
2. **Individual Models**: LGB + XGB + CatBoost (5-fold CV)
3. **Feature Engineering**: Multiple representations (binning, interactions)
   - CRITICAL: Test each on CV before committing (learned from 2026-02 failure)
4. **Stacking**: 3 base models → Ridge L1 meta-learner
5. **Blending**: Optimized weights on 10k holdout sample
6. **Calibration**: IsotonicRegression for probability calibration

### Phase 2: Advanced (If needed)

If public score < 0.92, implement:
- 7+ feature representations (like 1st place solution)
- Optuna-based OOF selection (2500 trials)
- Additional model types (RGF, TabICL, AutoGluon, RealMLP)
- Full-data retraining with 20 random seeds
- CV-LB monitoring across multiple submissions

## 📈 Performance Targets

| Metric | Target |
|--------|--------|
| **Baseline (LogisticRegression)** | 0.78-0.82 AUC |
| **Single Model (LGB)** | 0.85-0.90 AUC |
| **Ensemble (Blended)** | 0.88-0.93 AUC |
| **Kaggle Public LB** | Top 10-15% |
| **Position** | ~200-300 / 2500+ |

**Reference**: 2026-02 achieved 0.9549 AUC (position 2281/2893)

## ⚙️ Hyperparameters

**LightGBM Classification**:
- n_estimators=300, learning_rate=0.06, num_leaves=90, max_depth=8

**XGBoost Classification**:
- n_estimators=350, learning_rate=0.04, max_depth=6

**CatBoost Classification**:
- iterations=300, learning_rate=0.06

**Blending Weights** (optimized on holdout):
- LGB: 0.40, XGB: 0.40, CatBoost: 0.15, Stacking: 0.05

## 🎯 Key Lessons from 2026-02 & 1st Place

1. **Feature diversity matters more than model count**
   - We fell 49 bp short of 1st place mainly due to using only 1 feature representation
   - Solution: Test binning, digits, frequency encoding, target encoding, categorical strings

2. **Trust CV-LB relationship, not absolute CV**
   - 1st place submitted CV 0.95578 instead of best CV 0.955865 they found
   - Rationale: Beyond 0.95578, CV-LB correlation degraded (split overfitting)
   - Result: Correct decision for final score

3. **Feature engineering: Test incrementally**
   - 2026-02: Added 13 engineered features → AUC crashed from 0.9549 to 0.4993
   - Action: Test each representation on 3-fold CV before committing

4. **Simple ensembles beat complex ones**
   - Ridge meta-learner more stable than nonlinear stacking
   - Aggressive selection (only ~10% of OOFs) prevents overfitting

## 📝 Execution Progress

- [ ] Run notebook baseline to understand data
- [ ] Execute model comparison (5-fold CV)
- [ ] Test feature engineering incrementally
- [ ] Run complete ensemble + stacking + blending
- [ ] Generate submission
- [ ] Upload to Kaggle and monitor leaderboard
- [ ] After Day 2-3: Check public score + CV-LB relationship
- [ ] If score < 0.92: Implement Phase 2 improvements
- [ ] Document final results and learnings

## 🔬 Hypothesis

> Customer churn is similar in structure to heart disease (binary classification, ~600k samples, mix of features). Using the same ensemble architecture with improved feature diversity should achieve 0.88-0.92+ AUC. Gap to 1st place was feature representation diversity, not ensemble method. Applying 3-4 different feature representations should narrow that gap significantly.

## 📚 References

- **2026-01 & 2026-02 Learnings**: See root [LEARNINGS.md](../LEARNINGS.md)
- **Workflow**: See [WORKFLOW.md](../WORKFLOW.md)
- **1st Place S6E2 Writeup**: https://www.kaggle.com/competitions/playground-series-s6e2/writeups/1st-place-solution-diversity-selection-and-t
- **Challenge Overview**: https://www.kaggle.com/competitions/playground-series-s6e3/overview

---

*Created*: March 1, 2026  
*Status*: Notebook structure complete, ready for execution