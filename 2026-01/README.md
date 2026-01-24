# 2026-01: Playground Series S6E1 - Predicting Student Test Scores

## 📋 Challenge Overview

**Kaggle Link**: https://www.kaggle.com/competitions/playground-series-s6e1

**Objective**: Predict student exam scores based on available features

**Dataset**: 
- Training set: 630,000 samples with 12 features (+ target)
- Test set: 270,000 samples with 11 features
- Target variable: `exam_score`
- ID column: `id`

**Evaluation Metric**: Root Mean Squared Error (RMSE)

## 🎯 Approach

### 1. Exploratory Data Analysis (EDA)
- [x] Load and inspect data
- [x] Check for missing values
- [x] Analyze feature distributions
- [x] Identify correlations with target
- [ ] Detect outliers

### 2. Data Preprocessing
- [x] Handle missing values
- [x] Encode categorical variables
- [x] Scale/normalize features
- [ ] Remove/engineer features

### 3. Feature Engineering
- [ ] Create interaction features
- [ ] Polynomial features
- [ ] Domain-specific features
- [ ] Feature selection

### 4. Model Development
- [x] Baseline model
- [ ] Try multiple algorithms
- [ ] Hyperparameter tuning
- [ ] Cross-validation

### 5. Ensemble & Optimization
- [ ] Combine models
- [ ] Final hyperparameter optimization
- [ ] Generate predictions

## 📊 Results

### Baseline Model: Linear Regression
- **Training RMSE**: 9.9596
- **Validation RMSE**: 9.9452
- **Training R²**: 0.7232
- **Validation R²**: 0.7219

### Performance Progression
| Attempt | Model | Validation RMSE | Notes |
|---------|-------|-----------------|-------|
| 1 | Linear Regression (Baseline) | 9.9452 | All features, simple scaling |
| 2 | [Pending] | - | [To be updated] |
| 3 | [Pending] | - | [To be updated] |

## 💡 Key Learnings

### Feature Importance
**Numeric Features**:
- `study_hours`: Strong positive correlation (0.762) with exam_score
- `class_attendance`: Moderate positive correlation (0.361)
- `sleep_hours`: Weak positive correlation (0.167)
- `age`: Negligible correlation (0.010)

**Categorical Features**: 7 features (gender, course, internet_access, sleep_quality, study_method, facility_rating, exam_difficulty)

### Data Characteristics
- No duplicate rows
- No missing values
- Target (exam_score) distribution is approximately normal, centered at 62.5
- Range: 19.6 to 100.0 (scale of 0-100)
- Large dataset: 630,000 training samples

## 📂 File Structure

- `notebook.ipynb` - Working notebook with full analysis and modeling
- `submissions/` - Final submission versions
- `data/` - Dataset files
- `notes.md` - Key insights and techniques used

---

**Status**: Baseline Established ✅
**Last Updated**: January 24, 2026
