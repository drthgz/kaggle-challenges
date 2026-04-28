# S6E4 - Predicting Irrigation Need

## Challenge Info
- **URL**: https://www.kaggle.com/competitions/playground-series-s6e4
- **Task**: Multiclass classification — predict `Irrigation_Need` (Low / Medium / High)
- **Metric**: Balanced Accuracy Score (higher is better)
- **Deadline**: April 30, 2026
- **Source dataset**: [Irrigation Water Requirement Prediction Dataset](https://www.kaggle.com/datasets/miadul/irrigation-water-requirement-prediction-dataset/data) (20 cols → synthetic expansion to 43 cols)

## Data Overview
- ~43 columns total (id + ~41 features + target)
- Features: crop type, soil type, temperature, rainfall, evapotranspiration parameters
- Target classes: Low, Medium, High

## Metric Notes
- `balanced_accuracy_score` — accounts for class imbalance by computing accuracy per class and averaging
- Prediction format: class label string (Low/Medium/High), NOT probability
- Best public scores seen: ~0.978 (pairwise target encoding + XGB+CAT)

## First Iteration Plan
1. Load data, EDA (shape, target distribution, missing values, feature types)
2. Encode target: Low=0, Medium=1, High=2
3. LabelEncode categorical, StandardScale numeric
4. 5-fold StratifiedKFold CV with LGB / XGB / CatBoost
5. OOF-safe blend with Ridge meta-learner
6. Generate submission

## Results Log
| Iteration | Model | CV Balanced Acc | LB Score | Notes |
|-----------|-------|-----------------|----------|-------|
| 1 | — | — | — | First submission |

## What Worked / Didn't
- Baseline stack (no pairwise target encoding) outperformed the pairwise-TE run.
- Pairwise-TE run showed an unusual XGBoost instability: one CV fold dropped sharply (Fold 4 around 0.898), while other folds stayed near the expected range.
- This behavior likely indicates interaction between high-cardinality encoded features and model/early-stopping dynamics, not a broad data issue.

## Study Note For Next Iteration
- Investigate why XGBoost can collapse on a single fold when pairwise target encoding is added.
- Check per-fold class distribution drift and per-fold category coverage for key columns (Water_Source, Irrigation_Type, moisture_bucket).
- Compare XGB with and without scaling on tree inputs, and tune regularization/early-stopping under high-dimensional encoded features.
- Validate whether OOF encoding noise or sparse category mappings are amplifying fold variance.

## Key Differences from S6E3
- Multiclass (3 classes) vs binary
- Metric: balanced accuracy vs AUC-ROC
- Submission: class label string, not probability
- Use `predict_proba` → `argmax` → decode label
