# 2026-02: Playground Series - Predicting Heart Disease

## Challenge Overview

**Competition**: Kaggle Playground Series - Season 6 Episode 2  
**Task**: Binary classification - Predict likelihood of heart disease  
**Evaluation Metric**: AUC-ROC (Area Under Curve - Receiver Operating Characteristic)  
**Duration**: February 1 - February 28, 2026  
**Participants**: 240+ entrants, top 10% target  

## Competition Details

- **Objective**: Predict probability of heart disease (0.0 - 1.0)
- **Submission Format**: `id,Heart Disease` with probabilities
- **Data Type**: Synthetically generated tabular data
- **Difficulty**: Beginner-friendly playground challenge

## Learnings from 2026-01

Key insights to apply:
1. ✅ **Feature engineering is secondary** - dominant features matter most
2. ✅ **Ensemble methods** - Stacking + Blending effective for combining models
3. ✅ **Hyperparameter tuning** - LightGBM and XGBoost almost identical in CV
4. ✅ **Cross-validation stability** - 5-fold CV provides consistent results
5. ✅ **Runtime optimization** - Remove visualizations, keep computations

## Modifications for Classification

Since this is binary classification (not regression):
- **Primary metric**: AUC-ROC instead of RMSE
- **Output**: Probabilities (0.0-1.0) instead of continuous values
- **Models**: Probabilistic classifiers (LGB, XGB with probability outputs)
- **Feature engineering**: Different features (correlations with binary target)
- **Baseline**: Logistic Regression instead of Linear Regression

## Target Leaderboard Position

**2026-01 Result**: 2243 private leaderboard  
**2026-02 Target**: Top 10 (~24 position)  
**Strategy**: Apply all optimizations + adapt for classification task

## Timeline

1. **Phase 1**: Data exploration & feature analysis
2. **Phase 2**: Baseline classification model
3. **Phase 3**: Feature engineering (classification-specific)
4. **Phase 4**: Model selection & hyperparameter tuning
5. **Phase 5**: Ensemble methods (stacking + blending)
6. **Phase 6**: Optimization & final submission

## Files

- `notebook.ipynb` - Main pipeline (to be created)
- `data/` - Train/test CSV files
- `submissions/` - Output directory
- `LEARNING_LOG.md` - Progress tracking
