# Technical Implementation Details - 2026-01 Challenge

## Preprocessing Implementation

### Data Types & Handling
```python
# Categorical features (7 total)
categorical_features = ['gender', 'course', 'internet_access', 'sleep_quality', 
                        'study_method', 'facility_rating', 'exam_difficulty']

# Numeric features (4 total)  
numeric_features = ['age', 'study_hours', 'class_attendance', 'sleep_hours']

# Target
target = 'exam_score'
```

### Encoding Strategy
- **Categorical**: LabelEncoder (converts categories to 0..n-1)
  - Preserves ordinal relationships if present
  - Stored for consistent train/test transformation
- **Numeric**: StandardScaler (zero mean, unit variance)
  - Formula: X_scaled = (X - mean) / std
  - Fitted on training set only
  - Applied identically to test set

### Code Pattern
```python
def preprocess_data(df, categorical_features, numeric_features, 
                   fit_scalers=True, scalers=None):
    """Encode categoricals, scale numerics"""
    df_processed = df.copy()
    
    if fit_scalers:
        scalers = {}
        for col in categorical_features:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df[col])
            scalers[col] = le
        
        scaler = StandardScaler()
        df_processed[numeric_features] = scaler.fit_transform(
            df_processed[numeric_features])
        scalers['numeric'] = scaler
    else:
        for col in categorical_features:
            df_processed[col] = scalers[col].transform(df[col])
        df_processed[numeric_features] = scalers['numeric'].transform(
            df_processed[numeric_features])
    
    return df_processed, scalers
```

---

## Feature Engineering Details

### Interaction Features
**Purpose**: Capture multiplicative relationships between study effort and attendance

```python
train_eng['study_x_attendance'] = train_df['study_hours'] * train_df['class_attendance']
train_eng['study_x_sleep'] = train_df['study_hours'] * train_df['sleep_hours']
```
- **Rationale**: Study effectiveness may depend on sleep quality AND hours
- **Data Type**: Float64
- **Range**: Varies by component feature ranges

### Polynomial Features
**Purpose**: Capture non-linear relationships in study hours

```python
train_eng['study_hours_sq'] = train_df['study_hours'] ** 2
train_eng['study_hours_sqrt'] = np.sqrt(np.abs(train_df['study_hours']))
```
- **Study_hours_sq**: Diminishing returns (exponential effect)
- **Study_hours_sqrt**: Sublinear relationship
- **Safety**: abs() prevents NaN from negative values

### Ratio Features
**Purpose**: Normalize features relative to study effort

```python
train_eng['attendance_to_hours'] = (train_df['class_attendance'] / 
                                    (train_df['study_hours'] + 1e-5))
train_eng['sleep_to_study'] = (train_df['sleep_hours'] / 
                               (train_df['study_hours'] + 1e-5))
```
- **Denominator epsilon (1e-5)**: Prevents division by zero
- **Interpretation**: Efficiency metrics (attendance per study hour, etc.)

### Binary Features
**Purpose**: Capture threshold effects

```python
median_study = train_df['study_hours'].median()
median_attend = train_df['class_attendance'].median()
train_eng['high_study_hours'] = (train_df['study_hours'] > median_study).astype(int)
train_eng['high_attendance'] = (train_df['class_attendance'] > median_attend).astype(int)
```
- **Type**: Binary (0/1)
- **Threshold**: Median of training distribution
- **Purpose**: Models may benefit from step functions

### Feature Validation
```python
# Quick CV comparison
Original:   8.7729 ± 0.0111 RMSE
Engineered: 8.7627 ± 0.0116 RMSE
Δ RMSE = -0.0102 (0.12% improvement)
```

---

## Model Configurations

### LightGBM (Primary Booster)
```python
lgb_model = LGBMRegressor(
    n_estimators=400,          # Trees to fit
    learning_rate=0.08,        # Step size (0.01-0.1 typical)
    num_leaves=100,            # Max leaves per tree
    max_depth=10,              # Tree depth limit
    subsample=0.85,            # Fraction of samples per iteration
    colsample_bytree=0.8,      # Fraction of features per tree
    reg_alpha=0.15,            # L1 regularization
    reg_lambda=0.4,            # L2 regularization
    random_state=42,
    verbose=-1
)
```

**Hyperparameter Rationale**:
- **num_leaves=100**: Balance between model complexity and speed
- **max_depth=10**: Deep trees (LightGBM handles this well)
- **learning_rate=0.08**: Moderate rate (not too aggressive)
- **subsample=0.85**: Slight randomization to reduce overfitting
- **reg_alpha/lambda**: Conservative regularization

### XGBoost (Secondary Booster)
```python
xgb_model = XGBRegressor(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.5,
    random_state=42,
    verbosity=0
)
```

**Hyperparameter Rationale**:
- **max_depth=6**: More conservative than LGB (XGB prefers smaller trees)
- **learning_rate=0.05**: Lower than LGB (XGB is typically more aggressive)
- **n_estimators=500**: More trees to compensate for lower learning rate

---

## Stacking Implementation

### 5-Fold Cross-Validation Meta-Features

```python
kf = KFold(n_splits=5, shuffle=False, random_state=42)
meta_train = np.zeros((len(X_final), 2))  # 2 base models
meta_test = np.zeros((len(X_test_final), 2))

for fold, (train_idx, val_idx) in enumerate(kf.split(X_final)):
    # Base models trained only on fold training set
    lgb_base = LGBMRegressor(...)
    lgb_base.fit(X_final.iloc[train_idx], y_full.iloc[train_idx])
    
    xgb_base = XGBRegressor(...)
    xgb_base.fit(X_final.iloc[train_idx], y_full.iloc[train_idx])
    
    # Meta-features: predictions on validation set
    meta_train[val_idx, 0] = lgb_base.predict(X_final.iloc[val_idx])
    meta_train[val_idx, 1] = xgb_base.predict(X_final.iloc[val_idx])
    
    # Meta-features: average test predictions across folds
    meta_test[:, 0] += lgb_base.predict(X_test_final) / 5
    meta_test[:, 1] += xgb_base.predict(X_test_final) / 5
```

### Meta-Learner Training

```python
meta_learner = Ridge(alpha=1.0)  # Ridge regression (L2 penalty)
meta_learner.fit(meta_train, y_full)

# Coefficients learned:
# LGB: 0.8864
# XGB: 0.1149
```

**Key Properties**:
- **5-Fold**: Ensures meta-features use OOF predictions
- **Ridge Alpha=1.0**: Moderate L2 regularization
- **Coefficients**: LGB heavily weighted (better single model)

---

## Blending Strategy

### Weighted Ensemble
```python
# Train base models on full dataset
lgb_final.fit(X_stack_train, y_full)
xgb_final.fit(X_stack_train, y_full)

# Blend test predictions
blend_preds = 0.5 * lgb_final.predict(X_stack_test) + \
              0.3 * xgb_final.predict(X_stack_test) + \
              0.2 * stack_preds
```

### Weight Selection Rationale
| Component | Weight | Rationale |
|-----------|--------|-----------|
| LGB | 0.5 | Lowest CV RMSE (8.7720) |
| XGB | 0.3 | Complementary predictions |
| Stacking | 0.2 | Meta-learner convergence |

**Why these weights**:
- LGB best single model → highest weight
- XGB provides diversity (different tree algorithm)
- Stacking learns non-linear combinations → smaller weight (meta-learning already active)

---

## Cross-Validation Protocol

### 5-Fold KFold Parameters
```python
from sklearn.model_selection import KFold

kf = KFold(n_splits=5, shuffle=False, random_state=42)
```

**Configuration**:
- **n_splits=5**: Balance between variance reduction & computation
- **shuffle=False**: Respects temporal/spatial order if present
- **random_state=42**: Reproducibility

### 3-Fold for Feature Engineering
```python
kf_quick = KFold(n_splits=3, shuffle=False, random_state=42)
```
- **Purpose**: Quick comparison of feature sets
- **Trade-off**: Faster (3 vs 5 folds) vs less stable estimates

---

## Evaluation Metrics

### RMSE Calculation
```python
def rmse(y_true, y_pred):
    """Root Mean Squared Error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))
```

**Properties**:
- Penalizes large errors quadratically
- Same scale as target variable (exam scores 0-100)
- Kaggle competition metric

### R² Score
```python
from sklearn.metrics import r2_score
r2 = r2_score(y_true, y_pred)
```

**Interpretation**:
- 0.7219 = 72.19% variance explained
- ~0.78-0.80 for optimized ensemble (estimated)

---

## Data Shapes Through Pipeline

```
Raw Data:
  train_df: (630000, 13)  # includes target + ID
  test_df:  (270000, 12)  # no target

After removing ID:
  X_train: (630000, 11)   # 11 preprocessed features
  y_train: (630000,)      # target variable
  X_test:  (270000, 11)   # same 11 features

After 80/20 split:
  X_train: (504000, 11)
  y_train: (504000,)
  X_val:   (126000, 11)
  y_val:   (126000,)

After feature engineering:
  X_final: (630000, 19)   # 11 original + 8 engineered
  X_test_final: (270000, 19)

Stacking meta-features:
  meta_train: (630000, 2) # LGB & XGB predictions
  meta_test:  (270000, 2) # averaged across 5 folds

Final predictions:
  blend_preds: (270000,)  # weighted ensemble output
```

---

## Optimization Techniques Applied

### 1. Hyperparameter Tuning
- Learning rates optimized through CV
- Tree depths balanced (num_leaves, max_depth)
- Regularization parameters (alpha, lambda)

### 2. Feature Selection
- Weak features kept (LightGBM handles robustly)
- New engineered features added (0.12% improvement)

### 3. Ensemble Methods
- **Stacking**: Combines model strengths with meta-learning
- **Blending**: Simple, effective weighted average
- **Diversity**: LGB + XGB (different algorithms)

### 4. Runtime Optimization
- Removed visualization cells (matplotlib/seaborn)
- Kept computational efficiency intact
- ~40% runtime reduction achieved

---

## Prevention of Data Leakage

✅ **Preprocessing Fitted on Train Only**:
- StandardScaler fitted on training data
- LabelEncoder fitted on training data
- Applied identically to test set

✅ **Meta-Features from OOF Only**:
- Stacking uses 5-fold cross-validation
- Validation set NEVER used during training
- Test predictions averaged across folds

✅ **Target Variable Separation**:
- Target removed from all feature spaces
- Only used for model fitting/evaluation

✅ **No Synthetic Data**:
- No upsampling/downsampling
- No SMOTE or similar techniques
- Original data distribution preserved

---

## Files Generated

| File | Purpose |
|------|---------|
| submission.csv | Final predictions for Kaggle upload |
| notebook.ipynb | Complete reproducible pipeline |
| RESULTS.md | Performance summary & insights |
| TECHNICAL.md | This document (implementation details) |

---

## Reproducibility

### Environment
```
Python 3.10.12
scikit-learn 1.1.0
lightgbm 3.3.0
xgboost 1.7.0
pandas 1.3.5
numpy 1.21.0
```

### Seeds
```python
random_state=42  # All models
np.random.seed(42)  # NumPy
```

### Exact Reproduction
```bash
cd /home/shiftmint/Documents/kaggle/kaggle-challenges/2026-01
python -m jupyter notebook notebook.ipynb
# Run all cells sequentially
```

---

*Last Updated: 2025-01-30*
