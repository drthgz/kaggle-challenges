# Challenge Template

Template for starting new Kaggle monthly challenges. Copy to new challenge folder.

---

## How to Use This Template

1. Create new folder: `mkdir 2026-XX`
2. Copy this file's structure to `2026-XX/README.md` (rename to match challenge)
3. Read [WORKFLOW.md](../WORKFLOW.md) for execution steps
4. Refer to [LEARNINGS.md](../LEARNINGS.md) for learnings from past challenges

---

## Challenge Overview

**Competition**: [Challenge Name]  
**URL**: [Kaggle Link]  
**Type**: Regression / Classification  
**Metric**: RMSE / AUC-ROC / Other  
**Timeline**: [Dates]  
**Goal**: Top [X] position

## Data Summary

- **Train size**: [rows] × [cols]
- **Test size**: [rows] × [cols]
- **Target**: [name] (distribution, range, etc.)
- **Features**: [numeric count] numeric, [categorical count] categorical
- **Missing values**: [describe]
- **Key observations**: [describe]

## Approach

### Phase 1: Baseline (Single Model)
- Logistic Regression (classification) or Linear Regression (regression)
- Expected score: [estimate]

### Phase 2: Boosting (Multiple Models)
- LightGBM, XGBoost, CatBoost
- Expected improvement: 5-15%

### Phase 3: Features
- Interactions between top features
- Polynomials on dominant features
- Domain-specific engineering
- Expected improvement: 1-3%

### Phase 4: Ensemble
- 5-fold stacking (LGB + XGB)
- Weighted blending
- Expected improvement: 2-5%

## Results Log

| Approach | Score | Position | Notes |
|----------|-------|----------|-------|
| Baseline | — | — | [Details] |
| + Engineering | — | — | [Details] |
| + Ensemble | — | — | [Final result] |

## Key Learnings

- **What worked**: [Describe]
- **What didn't**: [Describe]
- **Next time**: [For LEARNINGS.md]
- **Code reuse**: [What to extract to shared/]

## Files

- `notebook.ipynb` - Main ML pipeline (30 cells typical)
- `data/train.csv` - Training data
- `data/test.csv` - Test data
- `data/sample_submission.csv` - Submission format reference
- `submissions/submission.csv` - Final submission
- `notes.md` - Project-specific insights (challenge-specific only; no duplicates in LEARNINGS.md)

---

*Last Updated*: [Date]
