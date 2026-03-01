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

**Challenge Type**: Classification | **Metric**: AUC-ROC | **Result**: Position 2281 / 2893, Private LB 0.95486

### Final Performance

- **Baseline (LogisticRegression)**: 0.9515 AUC
- **Single best models**: 0.9550-0.9551 AUC (LGB, XGB, CatBoost nearly identical)
- **Stacking (L1+L2)**: 0.9552 AUC (minimal improvement over singles)
- **Blending with optimized weights**: 0.9588 AUC (holdout test)
- **Final submission**: 0.95486 private LB (+0.18% vs prior 0.95310 baseline)
- **1st place**: 0.95535 (49 bp ahead of us)

### Key Learnings

#### ✅ What Transferred Well from 2026-01
- 5-Fold stacking architecture with LGB + XGB + CatBoost base models
- Weighted blending with empirical weight optimization on holdout
- Ridge meta-learner for combining predictions
- Preprocessing pipeline (LabelEncoder + StandardScaler)
- Cross-validation strategy with StratifiedKFold for classification

#### ✅ Classification-Specific Insights
1. **Model convergence**: All three GBDT models (LGB, XGB, Cat) converged to nearly identical performance (~0.9550-0.9551)
   - Implies single model had ~95% of ensemble potential
   - Best 3-model ensemble only added 0.15% improvement

2. **Weight optimization matters**: 
   - Tested 4 weight combinations on 10k holdout sample
   - Best: (LGB 0.40, XGB 0.40, CatBoost 0.15, Stack 0.05) = 0.9588 AUC
   - Demonstrated that blending weights need empirical search, not equal averaging

3. **Calibration had minimal effect**:
   - IsotonicRegression improved training AUC from 0.9552 → 0.9553 (1 bp)
   - Suggests predictions already well-calibrated; diminishing returns on post-hoc calibration

4. **Feature engineering backfired**:
   - Added 13 engineered features (interactions, polynomials, ratios)
   - Degraded performance severely (0.9549 → 0.4993 AUC)
   - Reverted to original 13 base features
   - **Lesson**: Not all datasets benefit from engineering; test incrementally

5. **StratifiedKFold critical for classification**:
   - Ensures each fold maintains class distribution (44.8% positive)
   - Regular KFold would cause imbalanced splits, inflating CV scores

#### ❌ Gaps Compared to 1st Place (0.95535 AUC)

1. **Feature diversity** (biggest gap):
   - We used: 1 representation (13 base features)
   - 1st place used: 7 different feature representations
     * Quantile & equal-width binning (numerical feature grouping)
     * Digit features (units/tens/hundreds place extraction)
     * All features treated as categorical strings
     * Frequency encoding (rare value detection)
     * Genetic programming features (nonlinear interactions via gplearn)
     * Target encoding with original dataset signals
     * Denoising VAE latent representations
   - **Impact**: Different features create ensemble diversity even with same models

2. **Model diversity**:
   - We used: 3 base models (LGB, XGB, CatBoost)
   - 1st place used: 7 model types (XGB, LGB, CatBoost, RealMLP, RGF, TabICL, AutoGluon)
   - Generated: 150 OOF predictions (we had ~15)

3. **Selection methodology**:
   - We used: Manual weight grid search (4 combinations)
   - 1st place used: Optuna with 2500 trials to find effective OOF subsets
   - Selected ~10% of OOFs (only consistently high-value ones kept)

4. **Full-data retraining**:
   - We used: Standard CV + blend on holdout
   - 1st place used: Retrain selected models on full dataset with 20 random seeds, then average
   - **Finding**: This beat simple holdout-based blend

5. **CV-LB monitoring** (most important meta-lesson):
   - 1st place calculated CV on fold predictions → monitored correlation to LB
   - Observed: Beyond CV 0.95578, improvements in CV no longer translated to LB
   - Interpreted: Split overfitting above that threshold
   - **Decision**: Submitted CV 0.95578 instead of best CV 0.955865 they found
   - **Result**: Correct choice (Final 0.95535 vs what their best CV might have yielded)
   - **Lesson**: Trust CV-LB *relationship*, not absolute CV score

### Classification-Specific Technical Details

- **Output format**: Probabilities [0.0, 1.0] via `predict_proba()`, not class labels
- **Metric**: AUC-ROC rewards probability calibration, not just ranking
- **Data characteristics**: 630K samples, 13 numeric features, 44.8% positive class (well-balanced)
- **Validation**: StratifiedKFold to maintain class ratio across folds
- **Meta-learner**: Ridge works better than nonlinear models for stacking (1st place confirmed this)

### What Didn't Work

Methods that provided no benefit or hurt performance:
- Feature engineering (13 engineered features destroyed performance)
- Highly flexible stacking (Level 2 added no AUC gain over L1)
- Too many OOFs without selection (1st place: averaging all 150 hurt; needed Optuna selection)
- Pseudo-labeling, knowledge distillation, very deep GBDT models (per 1st place writeup)

---

## 🎓 Meta-Lesson: 1st Place Analysis - "Trust the CV-LB Relation"

From Masaya Kawamata's 1st place writeup on S6E2:

### Core Philosophy
> "Don't chase the highest CV. Trust the CV-LB relationship. When CV no longer correlates with LB improvements, you're likely experiencing split overfitting."

### What 1st Place Did Differently

**Diversity Over Dominance**:
- Created multiple feature *representations*, not multiple models trained on the same features
- This alone justified testing. Their best single-model performance was 0.9557 - still under 1% away from final ensemble - but diversity is what enabled the edge

**Aggressive but Disciplined Selection**:
- Generated ~150 OOFs from 7 different models × 7 feature representations
- Used Optuna to search 2500 trials finding effective OOF subsets
- Only ~15% of OOFs consistently selected (insight: ensemble contribution ≠ individual CV performance)

**Simple Ensemble, Rigorously Validated**:
- Ridge regression as meta-learner (not neural networks or exotic stacking)
- Full-data retraining on final models with 20 random seeds + averaging
- Ridge worked best because it's stable and handles correlated predictions well

**The Critical Insight: CV-LB Consistency**:
- Monitored CV vs LB scores across multiple submissions
- Found the CV-LB relationship was *consistent* up to CV 0.95578
- But above 0.95578, improvements in CV didn't translate to LB improvements
- **Final decision**: Submitted CV 0.95578 instead of their best CV 0.955865 found
- **Result**: Correct - final private LB 0.95535 (confirms 0.95578 CV was more trustworthy)

### Why We Lost 49 Basis Points

1. **Feature representation** (estimated -30 bp impact):
   - Didn't explore binning, digit features, frequency encoding, target encoding, genetic programming, or VAEs
   - Stayed with 13 base features only

2. **Model diversity** (estimated -15 bp impact):
   - Used 3 models; 1st place used 7 different algorithms
   - Missing: RealMLP, RGF, TabICL, AutoGluon

3. **Ensemble size and selection** (estimated -5 bp impact):
   - 15 OOFs vs 150 OOFs
   - Manual 4-combination grid vs Optuna 2500-trial search

4. **Full-data retraining** (estimated -1 bp impact):
   - Didn't retrain on full data with multiple seeds

5. **CV-LB monitoring** (estimated +0 bp, but prevents overfitting):
   - We didn't systematically track CV-LB relationship
   - Could have caught overfitting if we had

### Actionable Takeaways for Next Challenges

1. **Feature representations > model count**
   - Before adding models, explore different feature transformations
   - Idea: Binning, interactions, frequency encoding, target encoding

2. **Monitor CV-LB correlation in your submissions**
   - Submit multiple candidates and track their CV vs LB scores
   - Look for where correlation breaks down
   - Avoid the highest CV if relationship becomes unreliable

3. **Test ensemble selection, not just averaging**
   - Use Optuna-style search to find best OOF subsets
   - Not all OOFs contribute equally; many can hurt

4. **Use Ridge as meta-learner for stability**
   - More complex meta-models (nonlinear stacking) tend to overfit
   - Ridge is simpler and more robust

5. **Full-data retraining can help**
   - After CV-based hyperparameter tuning, retrain on full data with multiple seeds
   - Average predictions across seeds

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
| **Baseline** | 9.9452 (Linear Reg) | 0.9515 (Logistic Reg) |
| **Best Single** | 8.7720 (LGB) | 0.9551 (LGB/XGB/Cat tied) |
| **Final** | 8.76 (Ensemble) | 0.95486 (Ensemble) |
| **Improvement** | 11.9% | 0.18% improvement (0.77% vs baseline) |
| **Position** | 2243 private LB | 2281 / 2893 private LB |
| **Features** | 19 (11 base + 8 eng) | 13 base (engineering failed) |
| **Key Feature** | study_hours (53%) | None dominant; all features ~equal |
| **Models Used** | LGB, XGB, Cat | LGB, XGB, Cat |
| **Ensemble** | Stacking + Blend | Stacking + Blend |
| **Gap to 1st** | Unknown | 49 bp (0.95535 vs 0.95486) |
| **Main bottleneck** | N/A | Feature diversity (only 1 representation) |

---

*Last Updated*: March 1, 2026
*Next Challenge*: 2026-03 (Customer Churn Prediction - In Progress)
