# Learning Log: From Regression to Classification

## Challenge Comparison

### 2026-01: Student Test Score Prediction
- **Type**: Regression (continuous output)
- **Metric**: RMSE (lower is better)
- **Output**: Continuous exam scores (0-100)
- **Result**: 2243 private leaderboard

### 2026-02: Heart Disease Prediction  
- **Type**: Binary classification (probability output)
- **Metric**: AUC-ROC (higher is better, 0.0-1.0)
- **Output**: Disease probability (0.0-1.0)
- **Target**: Top 10 ranking

---

## Key Differences in Approach

### 1. **Model Selection**

**Regression (2026-01)**:
- LGBMRegressor, XGBRegressor
- Output: continuous values
- Loss: squared error minimization

**Classification (2026-02)**:
- LGBMClassifier, XGBClassifier  
- Output: probability (predict_proba)
- Loss: log loss / cross-entropy

**Code Change**:
```python
# 2026-01 (Regression)
model = LGBMRegressor(...)
preds = model.predict(X_test)  # continuous values

# 2026-02 (Classification)
model = LGBMClassifier(...)
preds = model.predict_proba(X_test)[:, 1]  # probabilities for positive class
```

### 2. **Baseline Model**

**Regression**: Linear Regression
- Output: continuous values
- Error metric: MSE/RMSE

**Classification**: Logistic Regression
- Output: probabilities
- Uses sigmoid function
- Natural fit for binary classification

### 3. **Evaluation Metric**

**Regression**: RMSE (Root Mean Squared Error)
- Lower is better
- Same scale as target variable
- Easy to interpret

**Classification**: AUC-ROC (Area Under Curve)
- Higher is better (0.5 = random, 1.0 = perfect)
- Scale-independent
- Threshold-independent
- **Why better for imbalanced data**: Doesn't require optimizing a threshold

### 4. **Preprocessing Impact**

**Regression** (2026-01):
- StandardScaler critical for linear models
- Feature scaling affects gradient descent

**Classification** (2026-02):
- Tree-based models less sensitive to scaling
- But still apply for logistic regression component
- Categorical encoding same (LabelEncoder)

### 5. **Feature Engineering Strategy**

**Regression** (2026-01):
- Focus: Continuous relationships (study_hours dominating)
- Features: interactions, polynomials with continuous targets

**Classification (2026-02)**:
- Focus: Separability between classes
- Similar techniques but different interpretations:
  - Interactions: may amplify disease risk patterns
  - Polynomials: capture non-linear decision boundaries
  - Ratios: normalize features for better class separation

### 6. **Cross-Validation Strategy**

**Regression**:
- Regular KFold sufficient
- Metric: RMSE

**Classification** (Important!):
- Should use StratifiedKFold to maintain class distribution
- Applied in train/test split but not in full CV loop
- Metric: AUC-ROC

**Recommended Enhancement**:
```python
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=5, shuffle=False, random_state=42)
```

### 7. **Ensemble Weighting**

**Regression** (2026-01):
- Weights: 0.5 LGB, 0.3 XGB, 0.2 Stacking
- Based on RMSE CV performance

**Classification** (2026-02)**:
- Weights: 0.4 LGB, 0.4 XGB, 0.2 Stacking
- Adjusted because both LGB and XGB have similar AUC
- Equal weight on base models to leverage diversity

**Why this change?**
- LGB vs XGB almost identical in regression (8.7720 vs 8.7713)
- Same here too, so give equal weight
- Stacking learns non-linear relationships between probabilities

---

## What Transfers from 2026-01

✅ **Techniques that work for both**:
1. Feature engineering (interactions, polynomials, ratios)
2. Stacking with 5-fold CV for meta-features
3. Blending with weighted ensemble
4. Preprocessing pipeline (encode + scale)
5. Runtime optimization (no visualizations)
6. Cross-validation for model validation

✅ **Hyperparameter tuning approach**:
- Similar learning rates (0.05-0.08)
- Similar tree depths (5-8 for XGB, 8-10 for LGB)
- Same regularization strategies (alpha, lambda)

---

## New Techniques for Classification

🆕 **Specific to classification**:

1. **Predict probabilities, not raw predictions**
   - Always use `.predict_proba()[:, 1]` for positive class
   - Never use `.predict()` for submissions

2. **Threshold optimization** (optional future improvement)
   - Different thresholds affect precision/recall
   - AUC-ROC doesn't depend on threshold
   - Could optimize for F1/precision at decision boundary

3. **Class imbalance handling** (check data)
   - If imbalanced: use class_weight='balanced'
   - Or use threshold adjustment
   - Not applied yet; will test if needed

4. **Probability calibration** (optional)
   - Ensure predicted probabilities are realistic
   - CalibratedClassifierCV for calibration
   - Future enhancement if probabilities seem off

---

## Metric Deep Dive: AUC-ROC

### What is AUC-ROC?
- **ROC Curve**: True Positive Rate vs False Positive Rate at different thresholds
- **AUC**: Area under that curve
- **Range**: 0.0 - 1.0
  - 0.5 = random classifier
  - 1.0 = perfect classifier
  - <0.5 = worse than random (inverted predictions)

### Why AUC-ROC for this challenge?
1. **Threshold-independent**: Doesn't require optimizing cutoff
2. **Class imbalance robust**: Works well with imbalanced datasets
3. **Discrimination ability**: Measures how well model ranks positive examples higher

### Code Implementation
```python
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(y_true, y_pred_proba)  # y_pred_proba must be probabilities
```

---

## Expected Performance Comparison

### 2026-01 (Regression, RMSE)
- Baseline: 9.9452
- Final: ~8.76
- Improvement: 11.9%

### 2026-02 (Classification, AUC-ROC) 
- Baseline (Logistic Reg): ~0.80-0.85 (expected)
- Final (Ensemble): ~0.88-0.92 (targeting top 10)
- Improvement: ~10-15% in AUC

---

## Hyperparameter Adjustments for Classification

**2026-01 LightGBM**:
```python
LGBMRegressor(
    n_estimators=400, learning_rate=0.08, num_leaves=100,
    max_depth=10, subsample=0.85, colsample_bytree=0.8,
    reg_alpha=0.15, reg_lambda=0.4
)
```

**2026-02 LightGBM** (reduced for classification):
```python
LGBMClassifier(
    n_estimators=200,  # Reduced: fewer trees needed for classification
    learning_rate=0.08,
    num_leaves=80,     # Reduced: classification simpler
    max_depth=8,       # Reduced
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,     # Reduced regularization
    reg_lambda=0.3
)
```

**Why reductions?**
- Classification typically converges faster
- Fewer parameters needed for probability estimation
- Risk of overfitting with too many trees

---

## Potential Improvements for Top 10

1. **Stratified K-Fold** (immediate)
   - Maintain class distribution in folds
   - More representative validation

2. **Probability calibration** (if predictions seem miscalibrated)
   - Use CalibratedClassifierCV
   - Ensure model probabilities reflect actual frequencies

3. **Threshold optimization** (post-stacking)
   - Find optimal decision threshold
   - Maximize F1 or custom metric

4. **More ensemble diversity** (if time permits)
   - Add neural network as base model
   - Use different algorithms (LogisticRegression, SVM)
   - Combine with statistical models

5. **Hyperparameter grid search** (systematic tuning)
   - Optimize learning_rate, num_leaves per algorithm
   - Validate with stratified CV

6. **Feature selection** (if overfitting)
   - Use feature importance to select top features
   - Reduce model complexity
   - Faster inference

---

## Model Architecture Comparison

### 2026-01 Architecture
```
Input Features (11)
    ↓
Feature Engineering (+8 features) → 19 features
    ↓
Train/Val Split (80/20)
    ↓
5-Fold Stacking (LGB + XGB base models)
    ↓
Ridge Meta-Learner
    ↓
Blending: LGB 0.5 + XGB 0.3 + Stack 0.2
    ↓
RMSE ~8.76 (11.9% improvement)
```

### 2026-02 Architecture (Same pattern, different models)
```
Input Features (~10-15 expected)
    ↓
Feature Engineering (interactions, polynomials)
    ↓
Train/Val Split (80/20, stratified)
    ↓
5-Fold Stacking (LGBClassifier + XGBClassifier)
    ↓
Ridge Meta-Learner
    ↓
Blending: LGB 0.4 + XGB 0.4 + Stack 0.2
    ↓
AUC-ROC ~0.88-0.92 (target: top 10)
```

---

## Key Takeaway: Regression vs Classification

The **methodology transfers perfectly** - only model classes and output interpretation change:

| Aspect | Regression | Classification |
|--------|-----------|-----------------|
| Model Class | Regressor | Classifier |
| Output | Continuous | Probability (0-1) |
| Metric | RMSE | AUC-ROC |
| Baseline | LinearRegression | LogisticRegression |
| Ensemble Meta-Learner | Ridge | Ridge (works for both!) |
| Feature Engineering | Same techniques | Same techniques |
| Validation | KFold | StratifiedKFold |

**Core insight**: The ensemble architecture is algorithm-agnostic. Whether predicting scores or probabilities, stacking + blending provides consistent improvements.

---

*Generated for 2026-02 challenge - Learning from 2026-01 (RMSE 9.9452 → 8.76)*
