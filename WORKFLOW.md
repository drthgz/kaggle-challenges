# Challenge Execution Workflow

Step-by-step guide for running any Kaggle monthly challenge in this repository.

## Phase 1: Setup (5 minutes)

### 1.1 Create Challenge Directory
```bash
mkdir -p 2026-XX/{data,submissions}
cd 2026-XX/
```

### 1.2 Download Data
- Go to challenge page on Kaggle
- Download `train.csv`, `test.csv`, `sample_submission.csv`
- Save to `data/` folder

### 1.3 Verify Structure
```bash
ls -la  # Should show: notebook.ipynb, data/, submissions/, notes.md
```

---

## Phase 2: Development (varies)

### 2.1 Start Jupyter Notebook
```bash
jupyter notebook notebook.ipynb
```

### 2.2 Development Cycle
1. **Cells 1-5**: Load data, basic EDA, verify shape/types
2. **Cells 6-8**: Preprocessing pipeline (encode, scale)
3. **Cells 9-10**: Baseline model (Linear/Logistic Regression)
4. **Cells 11-13**: Model comparison (LGB, XGB, CatBoost)
5. **Cells 14-16**: Feature engineering (interactions, polynomials)
6. **Cells 17-19**: Feature validation (3-fold CV)
7. **Cells 20-22**: Stacking (5-fold meta-features)
8. **Cells 23-25**: Blending (empirical weights)
9. **Cells 26-30**: Final submission generation

### 2.3 Testing Features
- Use 3-fold CV for quick testing (3x faster than 5-fold)
- Test each feature type separately
- Only keep improvements > 0.1%

### 2.4 Iterative Improvement
- Adjust hyperparameters in isolation
- Test one change at a time
- Record scores in `notes.md`

---

## Phase 3: Submission (5 minutes)

### 3.1 Generate Submission CSV
- Final cell should output `submission.csv` to `submissions/` folder
- Verify format: `id, prediction` columns
- Check no missing values

### 3.2 Upload to Kaggle
- Log into Kaggle.com
- Go to competition's "Submit Predictions" tab
- Upload CSV file
- Wait for public score, then private score

### 3.3 Record Results
Update `notes.md` with:
- Public score
- Private score (if available)
- Leaderboard position
- What worked best
- What to try next

---

## Phase 4: Learning (varies)

### 4.1 Analyze Results
In `notes.md`, record:
- **Model Performance**: Which algorithm worked best?
- **Feature Insights**: Which engineered features helped most?
- **Hyperparameters**: What tuning mattered?
- **Surprises**: What didn't work as expected?
- **Time Investment**: Was it worth it?

### 4.2 Update Root-Level LEARNINGS.md
Extract generalizable insights:
- Patterns that apply to future challenges
- New techniques to try next time
- Model comparison findings
- Best preprocessing approaches
- Feature engineering strategies that worked

### 4.3 Code Reusability
- Extract utility functions to `shared/utils.py` if reusable
- Add new preprocessing templates if novel approach
- Document new techniques in `shared/techniques.md`

---

## Regression (RMSE) vs Classification (AUC-ROC)

### Regression Workflow (2026-01 Style)
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Baseline
baseline = LinearRegression().fit(X_train, y_train)
pred = baseline.predict(X_val)
rmse = np.sqrt(mean_squared_error(y_val, pred))
```

**Key differences**:
- Output: Continuous values (unbounded)
- Metric: RMSE (lower is better, same scale as target)
- Validation: KFold (no stratification needed)
- Models: LGBMRegressor, XGBRegressor

### Classification Workflow (2026-02 Style)
```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Baseline
baseline = LogisticRegression().fit(X_train, y_train)
pred_proba = baseline.predict_proba(X_val)[:, 1]  # Probability of class 1
auc = roc_auc_score(y_val, pred_proba)
```

**Key differences**:
- Output: Probabilities (0.0-1.0 range via `.predict_proba()`)
- Metric: AUC-ROC (higher is better, 0.0-1.0 scale)
- Validation: StratifiedKFold (maintain class distribution)
- Models: LGBMClassifier, XGBClassifier

---

## Hyperparameter Templates

### LightGBM
```python
# Regression
LGBMRegressor(
    n_estimators=400, learning_rate=0.08,
    num_leaves=100, max_depth=10,
    lambda_l1=0.1, lambda_l2=0.1
)

# Classification (fewer trees, smaller leaves)
LGBMClassifier(
    n_estimators=200, learning_rate=0.08,
    num_leaves=80, max_depth=8,
    lambda_l1=0.1, lambda_l2=0.1
)
```

### XGBoost
```python
# Regression
XGBRegressor(
    n_estimators=500, learning_rate=0.05,
    max_depth=6, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1
)

# Classification (fewer trees)
XGBClassifier(
    n_estimators=250, learning_rate=0.05,
    max_depth=5, subsample=0.8,
    colsample_bytree=0.8, reg_alpha=0.1
)
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| **Overfitting in stacking** | Increase lambda_l1/l2 regularization |
| **Slow CV** | Use fewer folds (3 instead of 5) for testing |
| **Leakage in preprocessing** | Always fit StandardScaler on train only |
| **Class imbalance** | Use StratifiedKFold, test scale_pos_weight |
| **Submission blank rows** | Check for NaN in predict_proba or predictions |
| **Score difference (public vs private)** | Normal if test set has different distribution |

---

## Performance Targets

### Realistic Improvements
1. **Baseline → Single Model**: 5-15% improvement
2. **Single → Feature Engineering**: 1-3% improvement  
3. **Features → Ensemble**: 2-5% improvement
4. **Total**: 8-20% typical improvement

### Leaderboard Positioning
- **Public: Top 50%**: 0.8× baseline performance
- **Public: Top 25%**: 0.9× baseline performance
- **Public: Top 10%**: 0.95× baseline performance (requires ensemble + tuning)
- **Public: Top 1%**: 0.98× baseline performance (requires advanced techniques)

### Time Allocation
- **Data Understanding**: 20%
- **Baseline**: 10%
- **Preprocessing**: 15%
- **Feature Engineering**: 25%
- **Ensemble**: 20%
- **Hyperparameter Tuning**: 10%

---

## Next Challenge Checklist

Before starting a new challenge:
- [ ] Read competition description (objective, metric, data)
- [ ] Check LEARNINGS.md for relevant past patterns
- [ ] Review shared/utils.py for reusable preprocessing
- [ ] Create new 2026-XX/ directory with standard structure
- [ ] Download data to 2026-XX/data/
- [ ] Copy notebook template or start fresh with standard cells

---

*Last Updated*: January 31, 2026
