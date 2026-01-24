"""
Shared utility functions for Kaggle challenges.

Reusable functions extracted from various challenges to support
preprocessing, modeling, and evaluation across projects.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor


# ============================================================================
# DATA LOADING & INSPECTION
# ============================================================================


def load_data(train_path, test_path=None):
    """
    Load training and optional test data from CSV files.

    Parameters:
    -----------
    train_path : str
        Path to training CSV file
    test_path : str, optional
        Path to test CSV file

    Returns:
    --------
    tuple : (train_df, test_df) or just train_df
    """
    train_df = pd.read_csv(train_path)

    if test_path:
        test_df = pd.read_csv(test_path)
        return train_df, test_df

    return train_df


def inspect_data(df, max_rows=5):
    """
    Provide comprehensive data inspection summary.

    Parameters:
    -----------
    df : pd.DataFrame
        DataFrame to inspect
    max_rows : int
        Maximum rows to display for head/tail
    """
    print(f"Dataset Shape: {df.shape}")
    print(f"\nFirst {max_rows} rows:")
    print(df.head(max_rows))
    print(f"\nLast {max_rows} rows:")
    print(df.tail(max_rows))
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nBasic Statistics:\n{df.describe()}")


# ============================================================================
# DATA PREPROCESSING
# ============================================================================


def handle_missing_values(df, strategy="mean", columns=None):
    """
    Handle missing values in DataFrame.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    strategy : str
        Strategy for filling: 'mean', 'median', 'drop', 'forward_fill'
    columns : list, optional
        Specific columns to apply strategy to

    Returns:
    --------
    pd.DataFrame : DataFrame with missing values handled
    """
    df_copy = df.copy()

    if columns is None:
        columns = df_copy.columns[df_copy.isnull().any()]

    if strategy == "mean":
        df_copy[columns] = df_copy[columns].fillna(df_copy[columns].mean())
    elif strategy == "median":
        df_copy[columns] = df_copy[columns].fillna(df_copy[columns].median())
    elif strategy == "drop":
        df_copy = df_copy.dropna(subset=columns)
    elif strategy == "forward_fill":
        df_copy[columns] = df_copy[columns].fillna(method="ffill")

    return df_copy


def scale_features(df, feature_columns, scaler_type="standard"):
    """
    Scale numerical features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    feature_columns : list
        Columns to scale
    scaler_type : str
        Type of scaler: 'standard' or 'minmax'

    Returns:
    --------
    pd.DataFrame : DataFrame with scaled features
    """
    df_copy = df.copy()

    if scaler_type == "standard":
        scaler = StandardScaler()
    elif scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")

    df_copy[feature_columns] = scaler.fit_transform(df_copy[feature_columns])
    return df_copy


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================


def create_interaction_features(df, feature_pairs):
    """
    Create interaction features between specified pairs.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    feature_pairs : list of tuples
        Pairs of feature names to create interactions for

    Returns:
    --------
    pd.DataFrame : DataFrame with new interaction features
    """
    df_copy = df.copy()

    for feat1, feat2 in feature_pairs:
        interaction_name = f"{feat1}_x_{feat2}"
        df_copy[interaction_name] = df_copy[feat1] * df_copy[feat2]

    return df_copy


def create_polynomial_features(df, feature_columns, degree=2):
    """
    Create polynomial features.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame
    feature_columns : list
        Columns to create polynomial features for
    degree : int
        Degree of polynomial

    Returns:
    --------
    pd.DataFrame : DataFrame with polynomial features
    """
    df_copy = df.copy()

    for col in feature_columns:
        for d in range(2, degree + 1):
            df_copy[f"{col}^{d}"] = df_copy[col] ** d

    return df_copy


# ============================================================================
# MODEL EVALUATION
# ============================================================================


def evaluate_model(model, X, y, cv=5, scoring="r2"):
    """
    Evaluate model using cross-validation.

    Parameters:
    -----------
    model : sklearn model
        Trained or untrained model
    X : array-like
        Features
    y : array-like
        Target variable
    cv : int
        Number of cross-validation folds
    scoring : str
        Scoring metric

    Returns:
    --------
    dict : Dictionary with mean and std of CV scores
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)

    return {
        "mean_score": scores.mean(),
        "std_score": scores.std(),
        "all_scores": scores,
    }


def feature_importance(model, feature_names):
    """
    Extract feature importance from tree-based models.

    Parameters:
    -----------
    model : sklearn tree-based model
        Trained model with feature_importances_ attribute
    feature_names : list
        Names of features

    Returns:
    --------
    pd.DataFrame : Features ranked by importance
    """
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Model does not have feature_importances_ attribute")

    importance_df = pd.DataFrame(
        {"feature": feature_names, "importance": model.feature_importances_}
    ).sort_values("importance", ascending=False)

    return importance_df


# ============================================================================
# UTILITIES
# ============================================================================


def split_train_val(X, y, test_size=0.2, random_state=42):
    """
    Split data into training and validation sets.

    Parameters:
    -----------
    X : array-like
        Features
    y : array-like
        Target variable
    test_size : float
        Proportion for validation set
    random_state : int
        Random seed

    Returns:
    --------
    tuple : (X_train, X_val, y_train, y_val)
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


# Add more utility functions as they emerge from challenges!
