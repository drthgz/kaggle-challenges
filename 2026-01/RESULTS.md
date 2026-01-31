# 2026-01 Student Test Score Prediction - Results & Summary

## 🎯 Challenge Overview
**Dataset**: Student Test Score Prediction  
**Task**: Predict exam scores for 270,000 students based on 12 features  
**Data Size**: 630,000 training samples + 270,000 test samples  
**Target Variable**: `exam_score` (continuous, range 19.6-100.0)

---

## 📊 Final Results

### Model Performance Comparison

| Model | CV RMSE | Std Dev | Improvement vs Baseline |
|-------|---------|---------|------------------------|
| **Baseline (Linear Regression)** | **9.9452** | - | **-** |
| XGBoost (5-fold CV) | **8.7713** | ±0.0128 | **11.82%** ↓ |
| LightGBM (5-fold CV) | **8.7720** | ±0.0120 | **11.82%** ↓ |
| CatBoost (5-fold CV) | **8.7916** | ±0.0120 | **11.56%** ↓ |
| **Final Ensemble (Stacking + Blending)** | **~8.76** | (estimated) | **~12%** ↓ |

### Feature Engineering Impact
- **Original features**: 8.7729 ± 0.0111 RMSE
- **Engineered features**: 8.7627 ± 0.0116 RMSE  
- **Improvement**: 0.0102 RMSE (0.12% better)

### Submission Statistics
- **File**: `submission.csv` (6.5 MB)
- **Records**: 270,000
- **Mean prediction**: 62.52
- **Std deviation**: 16.74
- **Min/Max**: 15.06 / 102.77
- **Format**: ✓ Valid (id, exam_score columns)

---

## 🔧 Pipeline Architecture

### 1. Data Preprocessing
```
Raw Data (630k × 13 features)
    ↓
[Encoding] LabelEncoder for 7 categorical features
[Scaling] StandardScaler for 4 numeric features
    ↓
Processed Data (630k × 11 final features)
```

**Features Used**:
- **Numeric (4)**: `age`, `study_hours`, `class_attendance`, `sleep_hours`
- **Categorical (7)**: `gender`, `course`, `internet_access`, `sleep_quality`, `study_method`, `facility_rating`, `exam_difficulty`

### 2. Feature Engineering
**8 New Features Created**:
1. `study_x_attendance`: Interaction term (study_hours × class_attendance)
2. `study_x_sleep`: Interaction term (study_hours × sleep_hours)
3. `study_hours_sq`: Polynomial (study_hours²)
4. `study_hours_sqrt`: Polynomial (√study_hours)
5. `attendance_to_hours`: Ratio feature (class_attendance / study_hours)
6. `sleep_to_study`: Ratio feature (sleep_hours / study_hours)
7. `high_study_hours`: Binary indicator (study_hours > median)
8. `high_attendance`: Binary indicator (class_attendance > median)

**Result**: 11 features → 19 total features  
**Validation**: 3-fold CV confirmed 0.12% improvement

### 3. Base Models

#### LightGBM Configuration
```python
n_estimators=400
learning_rate=0.08
num_leaves=100
max_depth=10
subsample=0.85
colsample_bytree=0.8
reg_alpha=0.15
reg_lambda=0.4
```
**CV Performance**: 8.7720 ± 0.0120 RMSE

#### XGBoost Configuration
```python
n_estimators=500
learning_rate=0.05
max_depth=6
subsample=0.8
colsample_bytree=0.8
reg_alpha=0.1
reg_lambda=0.5
```
**CV Performance**: 8.7713 ± 0.0128 RMSE

### 4. Ensemble Methods

#### Stacking (5-Fold Cross-Validation)
- **Base Models**: LightGBM + XGBoost (2 models)
- **Meta-Features**: 5-fold CV generates 2 meta-features per sample
- **Meta-Learner**: Ridge Regression (α=1.0)
- **Meta-Learner Weights**:
  - LGB: 0.8864
  - XGB: 0.1149
- **Fold-by-Fold Performance**:
  - Fold 1: LGB=8.7450, XGB=8.7691
  - Fold 2: LGB=8.7481, XGB=8.7798
  - Fold 3: LGB=8.7392, XGB=8.7725
  - Fold 4: LGB=8.7558, XGB=8.7912
  - Fold 5: LGB=8.7728, XGB=8.8051

#### Blending (Weighted Ensemble)
```python
blend_predictions = 0.5 × LGB + 0.3 × XGB + 0.2 × Stacking
```
- **Rationale**: 
  - LGB (0.5): Highest average CV score
  - XGB (0.3): Complementary predictions
  - Stacking (0.2): Learns relationships from base model predictions
- **Prediction Range**: [15.06, 102.77]

---

## 📈 Key Insights

### Feature Importance (Top 5)
Based on LightGBM feature importance analysis:

1. **study_hours** - 0.5295 (52.95% importance)
2. **sleep_quality** - 0.1268 (12.68% importance)
3. **study_method** - 0.0937 (9.37% importance)
4. **class_attendance** - 0.0687 (6.87% importance)
5. **facility_rating** - 0.0475 (4.75% importance)

**Weak Features** (low/no correlation):
- age (0.010)
- gender (0.001)
- course (0.002)
- exam_difficulty (-0.001)
- internet_access (0.006)

*Decision*: Kept all features as gradient boosting models handle weak features robustly

### Model Selection Process
1. Baseline Linear Regression: 9.9452 RMSE (reference)
2. Single boosting models tested (LGB, XGB, CatBoost)
3. XGBoost slightly edge (8.7713 vs 8.7720), but similar
4. Ensemble approach chosen to maximize diversity & robustness

### Runtime Optimization
- **Original notebook**: Multiple visualization cells (heatmaps, distributions)
- **Optimized notebook**: Commented out matplotlib/seaborn, retained numerical outputs
- **Runtime reduction**: ~40% faster

---

## 🚀 Improvements Applied

### ✅ Completed Optimizations
1. **Feature Engineering**: Interactions, polynomials, ratios, binaries (+0.12% RMSE)
2. **Ensemble Stacking**: 5-fold CV meta-features with Ridge meta-learner
3. **Weighted Blending**: Combined LGB + XGB + Stacking with empirical weights
4. **Hyperparameter Tuning**: Optimized learning rates, tree depths, regularization
5. **Runtime Efficiency**: Removed redundant visualizations

### 💡 Future Improvements (Not Implemented)
- Bayesian optimization for blend weights
- Neural network as base model (diversity boost)
- 10-fold CV instead of 5-fold (robustness)
- Advanced meta-learners (GradBoost, Neural Network)
- Domain-specific feature engineering

---

## 📁 File Structure

```
2026-01/
├── notebook.ipynb                      # Complete ML pipeline
├── RESULTS.md                          # This file
├── notes.md                            # Challenge notes
├── README.md                           # Challenge description
├── data/
│   ├── train.csv                       # 630,000 training samples
│   ├── test.csv                        # 270,000 test samples
│   └── sample_submission.csv           # Format reference
├── submissions/
│   ├── submission.csv                  # Final submission (current)
│   └── baseline_submission.csv         # Baseline for reference
```

---

## 🎓 Pipeline Execution Summary

**Total Cells**: 30 (20 code, 10 markdown)  
**Key Execution Order**:
1. Imports & environment setup
2. Data loading & EDA
3. Preprocessing (encoding + scaling)
4. Baseline Linear Regression (reference)
5. 5-Fold cross-validation (LGB, XGB, CatBoost)
6. Feature engineering (8 new features)
7. Feature set comparison (3-fold CV)
8. Stacking ensemble (5-fold meta-features + Ridge)
9. Blending (weighted average)
10. Final submission generation

**Total Runtime**: ~3-4 minutes (optimized)

---

## 📋 Submission Validation

✅ **File Format**: Valid  
✅ **Row Count**: 270,000 (matches test set)  
✅ **Column Structure**: [id, exam_score]  
✅ **Value Range**: [15.06, 102.77] (reasonable predictions)  
✅ **Distribution**: Mean 62.52 (matches training target ~62.5)  
✅ **No Missing Values**: All 270,000 predictions present  

---

## 🔍 Technical Details

### Data Split Strategy
- **Training**: 504,000 samples (80%)
- **Validation**: 126,000 samples (20%)
- **Cross-Validation**: 5-fold KFold

### Metrics Used
- **Primary**: Root Mean Squared Error (RMSE)
- **Secondary**: R² Score

### Validation Strategy
- 5-fold cross-validation for stacking meta-features
- 80/20 train/val split for baseline
- 3-fold CV for feature engineering comparison

### Preventing Overfitting
- Cross-validation on training data
- Regularization (L1/L2 penalties)
- Conservative ensemble weights
- Feature engineering validated on CV set

---

## 🎁 Deliverables

| Item | Status | Details |
|------|--------|---------|
| Training Pipeline | ✅ Complete | All preprocessing & modeling steps |
| Model Evaluation | ✅ Complete | 5-fold CV on 630K samples |
| Feature Engineering | ✅ Complete | 8 new features, 0.12% improvement |
| Ensemble Method | ✅ Complete | Stacking + Blending |
| Submission File | ✅ Ready | 270,000 predictions, valid format |
| Documentation | ✅ Complete | This results document |

---

## 📊 Performance Summary

```
BASELINE vs FINAL ENSEMBLE
===========================

Baseline (Linear Regression):     RMSE = 9.9452
Final Ensemble:                   RMSE ≈ 8.76

Improvement: ~1.18 RMSE reduction (11.9% better)
Expected Kaggle Ranking: Top 10-15% (estimated)
```

---

*Last Updated: 2025-01-30*  
*Challenge: Kaggle Monthly Challenge - 2026-01 Student Test Score Prediction*  
*Status: ✅ Submission Ready*
