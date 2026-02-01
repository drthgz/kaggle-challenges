# Master Learnings Document

Cumulative insights, patterns, and techniques discovered across challenges. For challenge-specific details, see individual `notes.md` files.

## 2026-01: Student Test Score Prediction (Regression)

**Challenge Type**: Regression | **Metric**: RMSE | **Result**: 2243 private leaderboard

### Key Learnings

#### 1. Feature Engineering ROI
- **Finding**: Adding 8 engineered features (interactions, polynomials, ratios) improved RMSE by only 0.0102 (0.12%)
- **Lesson**: Feature engineering has diminishing returns - don't over-engineer
- **What worked**: Interaction terms between dominant features (study_hours × other features)
- **What helped less**: Polynomial features and ratio features contributed but secondary benefit
- **Action**: Test features on CV set before committing to them

#### 2. Feature Importance Hierarchy
- **Finding**: study_hours accounted for 52.95% of predictive power alone
- **Lesson**: Identify dominant features early; weak features don't hurt tree models (keep them)
- **Data insight**: Age (r=-0.010), gender (r=0.001), course (r=0.002) had negligible correlation but tree models robust to noise
- **Action**: Focus engineering on top 3-5 features, not all features

#### 3. Model Selection Strategy  
- **Finding**: LightGBM (8.7720) and XGBoost (8.7713) were nearly identical in 5-fold CV
- **Lesson**: Both LGB and XGB almost always perform similarly; choose based on stability/speed
- **Observation**: CatBoost slightly worse (8.7916) but still competitive
- **Action**: Use ensemble of both LGB+XGB; they provide complementary perspectives despite similar CV scores

#### 4. Ensemble Architecture Effectiveness
- **Finding**: Stacking (5-fold meta-features) + Blending achieved 11.9% improvement over baseline (9.9452 → 8.76)
  - Baseline → Boosting: 11.82% improvement
  - Engineering → Ensemble: additional 0.5% improvement
- **Lesson**: Ensemble methods are the biggest lever for improvement
- **Structure**: Ridge meta-learner learned to weight LGB (0.8864) over XGB (0.1149) for this dataset
- **Action**: Always implement stacking + blending as final step

#### 5. Cross-Validation Stability
- **Finding**: 5-fold CV variance was very low:
  - LGB: ±0.0120 (0.14% std)
  - XGB: ±0.0128 (0.15% std)
- **Lesson**: Low variance indicates good generalization to test set
- **Action**: Use 5-fold for production; 3-fold sufficient for quick feature testing

#### 6. Hyperparameter Tuning
- **LightGBM**: n_estimators=400, learning_rate=0.08, num_leaves=100, max_depth=10
- **XGBoost**: n_estimators=500, learning_rate=0.05, max_depth=6
- **Observation**: Smaller learning rates (0.05-0.08) preferred; early stopping not needed with fixed n_estimators
- **Action**: Conservative regularization (alpha/lambda=0.1-0.4) helps generalization

### Regression-Specific Insights
- **Output**: Continuous values (not bounded)
- **Metric**: RMSE (same scale as target, easy to interpret)
- **Baseline**: Linear Regression provides good reference (~9.9452)
- **Feature scaling**: Important for linear models; less critical for tree models but applied anyway
- **Validation**: Simple KFold sufficient; no stratification needed

---

## 2026-02: Heart Disease Prediction (Classification)

**Challenge Type**: Classification | **Metric**: AUC-ROC | **Status**: In Progress

### Key Differences from Regression (2026-01)

#### Transferred Successfully
- ✅ 5-Fold stacking with LGB + XGB base models
- ✅ Weighted blending with Ridge meta-learner
- ✅ Feature engineering (interactions, polynomials, ratios)
- ✅ Preprocessing pipeline (categorical encoding + numeric scaling)
- ✅ Runtime optimization (no visualizations)
- ✅ Cross-validation strategy

#### What Changed
- **Model classes**: Classifier instead of Regressor
- **Output**: Probabilities (0.0-1.0) via `predict_proba()`, not raw predictions
- **Metric**: AUC-ROC (threshold-independent) vs RMSE (threshold-sensitive)
- **Baseline**: LogisticRegression vs LinearRegression
- **Tree counts**: Reduced to 200-250 (classification converges faster)
- **Blend weights**: 0.4 LGB, 0.4 XGB, 0.2 Stack (equal LGB/XGB due to similarity)
- **Validation**: Should use StratifiedKFold to maintain class distribution

---

## 🧠 General Patterns Discovered

### Model Comparison
| Model | Pros | Cons | Best For |
|-------|------|------|----------|
| LightGBM | Fast, stable | Sometimes overfits | Primary choice |
| XGBoost | Robust, diverse | Slower | Ensemble diversity |
| CatBoost | Handles cats well | Less tested | Datasets with many categoricals |
| Logistic Reg | Interpretable | Limited power | Baseline/benchmark |
| Ridge | Meta-learning | Single model weak | Meta-learner in stacking |

### Feature Engineering Strategy
1. **Domain interactions**: study_hours × other features (high ROI)
2. **Polynomials**: On top 1-3 features (moderate ROI)
3. **Ratios**: Secondary features divided (low ROI)
4. **Binary indicators**: Binned features (low ROI, test last)
5. **Validation**: Always test on CV before using

### Preprocessing Best Practices
1. **Categorical**: LabelEncoder for tree models, OneHotEncoder optional
2. **Numeric**: StandardScaler for linear models; optional for trees
3. **Order**: Fit on train, apply to test (prevent leakage)
4. **Validation**: Apply same preprocessors to all folds

### Ensemble Architecture
```
Base Model 1 (LGB) ─┬─ 5-Fold Stacking ─ Ridge Meta-Learner ─┬─ Blend
                   └─────────────────────────────────────────┤
Base Model 2 (XGB) ─┬───────────────────────────────────────┐│
                   └─────────────────────────────────────────┤│
                                                              ││
Base Models → Blending (0.4 LGB + 0.4 XGB + 0.2 Stack) ─────┴┴→ Final
```
**Why effective**: Each component reduces different error types; stacking learns optimal combination

### Validation Strategy
- **5-fold KFold**: Standard for stable CV estimates
- **StratifiedKFold**: Required for classification (maintain class ratio)
- **Quick testing**: 3-fold for feature experiments (3x faster)
- **Holdout test**: 80/20 split for baseline comparison

---

## 📋 Quick Reference

**Model Stack**:
1. Baseline: Linear/Logistic Regression (reference)
2. Single best: LightGBM with tuned hyperparameters
3. Ensemble base: LGB + XGBoost
4. Final: Stacking + Blending

**Feature Engineering Checklist**:
- [ ] Identify top 3-5 correlated features
- [ ] Create interactions with top features
- [ ] Test polynomial features on top 1-3
- [ ] Validate on 3-fold CV before keeping
- [ ] Stop if improvement < 0.1%

**Preprocessing Checklist**:
- [ ] LabelEncode categorical features
- [ ] StandardScale numeric features
- [ ] Fit on train only, apply to all splits
- [ ] Handle missing values (none so far)
- [ ] Check for duplicates

**Ensemble Checklist**:
- [ ] Build baseline model
- [ ] Compare 3+ algorithms (LGB, XGB, Cat)
- [ ] Select 2 best performers
- [ ] Implement 5-fold stacking
- [ ] Blend with empirical weights (test 0.3/0.3/0.4 range)

---

## 🔄 Challenge-by-Challenge Comparison

| Aspect | 2026-01 (Regression) | 2026-02 (Classification) |
|--------|----------------------|--------------------------|
| **Baseline** | 9.9452 (Linear Reg) | 0.80-0.85 (Logistic) |
| **Best Single** | 8.7720 (LGB) | ~0.90 (est) |
| **Final** | ~8.76 (Ensemble) | ~0.90+ (est) |
| **Improvement** | 11.9% | ~10-15% (est) |
| **Features** | 19 (11+8 eng) | ~20 (est) |
| **Key Feature** | study_hours (53%) | TBD |
| **Models Used** | LGB, XGB, Cat | LGB, XGB, Cat |
| **Ensemble** | Stacking + Blend | Stacking + Blend |

---

*Last Updated*: January 31, 2026
*Next Challenge*: 2026-03 (TBD)
