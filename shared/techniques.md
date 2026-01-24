# Common ML Techniques Reference

This document serves as a quick reference for ML techniques, when to use them, and key considerations.

## 📊 Feature Engineering

### Numerical Features
- **Scaling/Normalization**: Use StandardScaler for normally distributed features, MinMaxScaler for bounded ranges
- **Log Transform**: Apply to skewed distributions (e.g., income, population)
- **Binning/Discretization**: Convert continuous to categorical for decision trees
- **Polynomial Features**: $x, x^2, x^3$ for capturing non-linear relationships
- **Interaction Features**: $x_1 \times x_2$ when features interact synergistically

### Categorical Features
- **One-Hot Encoding**: Use for tree models and when categories are unordered
- **Label Encoding**: Use when ordinal relationship exists
- **Target Encoding**: Encode with target mean (use cross-validation to avoid leakage)
- **Frequency Encoding**: Encode by occurrence frequency

### Advanced Techniques
- **Domain-Specific Features**: Create features based on domain knowledge
- **Temporal Features**: Extract year, month, day, day-of-week from dates
- **Text Features**: TF-IDF, embeddings for text data
- **Statistical Features**: Mean, std, skew, kurtosis in windows

## 🤖 Model Selection

### Regression
- **Linear Regression**: Fast baseline for linear relationships
- **Decision Trees**: Non-linear, interpretable, prone to overfitting
- **Random Forest**: Robust ensemble, good for mixed feature types
- **XGBoost/LightGBM**: High performance, requires tuning
- **Neural Networks**: Complex non-linear relationships, needs more data

### Classification
- **Logistic Regression**: Linear, interpretable, fast
- **Decision Trees**: Non-linear, interpretable
- **Random Forest**: Good general-purpose classifier
- **XGBoost/LightGBM**: State-of-the-art performance
- **Neural Networks**: For complex patterns, images, text

## 🔄 Validation Strategy

### Cross-Validation
- **K-Fold (k=5 or 10)**: Standard approach for most problems
- **Stratified K-Fold**: For imbalanced classification
- **Time Series Split**: For temporal data (train on past, validate on future)
- **GroupKFold**: When data has natural groups

### Avoiding Common Pitfalls
- **Data Leakage**: Never fit scaler/encoder on full data including test
- **Target Leakage**: Don't include features computed from target
- **Time Leakage**: Don't use future data to predict past in time series

## 📈 Ensemble Methods

### Averaging Ensembles
- Simple averaging of predictions
- Weighted averaging based on CV scores
- Works well when models are diverse and independent

### Stacking
- Train meta-model on predictions from base models
- Captures correlations between base model predictions
- More complex but often more powerful

### Blending
- Similar to stacking but uses holdout validation set
- Faster to compute than stacking
- Less data-efficient

## 🎯 Hyperparameter Tuning

### Grid Search
- Exhaustive search over specified parameter grid
- Good for small search spaces
- Computationally expensive for large spaces

### Random Search
- Random sampling from parameter distributions
- Often finds good parameters faster than grid search
- Better for high-dimensional spaces

### Bayesian Optimization
- Uses probability model to guide search
- Most efficient for small number of trials
- Good for expensive objective functions (e.g., model training)

## ⚖️ Handling Class Imbalance

- **Resampling**: Oversample minority or undersample majority class
- **Class Weights**: Penalize misclassification of minority class
- **Threshold Tuning**: Adjust decision threshold for classification
- **Synthetic Data**: SMOTE for generating synthetic minority samples
- **Stratified Sampling**: Maintain class distribution in train/val/test splits

---

**To add**: Specific techniques, parameters, and results as challenges progress.
