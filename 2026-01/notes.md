# 2026-01 Challenge: Key Learnings & Notes

## 🔍 What Worked

### Baseline Approach
- **Linear Regression with standardized features** achieves RMSE of 9.95 on validation
- Simple label encoding for categorical features is effective
- StandardScaler works well for numeric features
- Strong performance suggests linear relationships are important

## ❌ What Didn't Work

*To be updated as challenge progresses*

## 💭 Important Insights

### Dataset Insights
1. **Large, clean dataset**: 630K training samples with no missing values or duplicates
2. **Numeric features dominate**: Only 4 numeric features but they have strong predictive power
3. **study_hours is dominant**: Correlation of 0.76 with target - single strongest predictor
4. **Target distribution**: Nearly normal, which suggests linear models are appropriate

### Model Insights
1. **Linear assumptions hold**: R² of 0.72 with simple linear regression suggests data has strong linear patterns
2. **No overfitting**: Training and validation RMSE nearly identical (9.96 vs 9.95)
3. **Feature encoding simple**: Label encoding categorical features works well, no need for complex encoding yet

## 🛠 Techniques & Methods Used

### Data Preprocessing
- **Label Encoding**: Applied to 7 categorical features (gender, course, internet_access, sleep_quality, study_method, facility_rating, exam_difficulty)
- **StandardScaler**: Applied to 4 numeric features (age, study_hours, class_attendance, sleep_hours)
- **Train-test split**: 80-20 split with random_state=42 for reproducibility

### Modeling Strategy
- **Baseline Model**: Linear Regression with all features
- **Evaluation**: RMSE on held-out validation set (20% of training data)

## 📈 Performance Analysis

- **Best single model**: Linear Regression, Validation RMSE = 9.9452, R² = 0.7219
- **Training-validation gap**: Minimal (0.0144 RMSE difference) indicates good generalization
- **Improvement potential**: R² of 0.72 suggests ~28% of variance unexplained - room for improvement

## 🚀 Next Steps / Future Improvements

- [ ] Try ensemble methods (Random Forest, XGBoost, LightGBM)
- [ ] Feature engineering: interaction features between study_hours and other features
- [ ] Hyperparameter tuning for tree-based models
- [ ] Polynomial features (especially for study_hours given its importance)
- [ ] Cross-validation to ensure robustness
- [ ] Outlier detection and handling
- [ ] Domain-specific features (e.g., study_hours × class_attendance interaction)

## 🔗 References & Resources

- **Challenge**: Playground Series S6E1
- **Metric**: Root Mean Squared Error (RMSE)
- **Baseline Code**: `notebook.ipynb` cells 1-12

---

**Last Updated**: January 24, 2026
