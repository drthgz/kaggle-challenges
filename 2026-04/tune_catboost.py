"""
Optuna hyperparameter search for CatBoost on S6E4 Irrigation Need.

Strategy
--------
- 3-fold CV inside each Optuna trial (fast; 630k rows is heavy)
- Metric: balanced_accuracy_score (matches competition)
- After search, re-run best params with full 5-fold CV to get true OOF
- Copy winning params into build_cat() in run_pipeline.py

Usage
-----
    python3 tune_catboost.py [--trials 60] [--timeout 3600]

Defaults: 60 trials, 1-hour wall-clock timeout.
"""

import argparse
import json
import os
import warnings

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)  # suppress per-trial noise

# ── Reproducibility ────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)

# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR = "./data"
RESULTS_PATH = "./optuna_catboost_results.json"


# ── Data loading & feature engineering ────────────────────────────────────
def is_categorical(series):
    return series.dtype == "object" or (
        hasattr(series.dtype, "name") and series.dtype.name in ("string", "str")
    )


def load_data():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
    TARGET = "Irrigation_Need"
    CLASS_ORDER = ["Low", "Medium", "High"]
    label_map = {c: i for i, c in enumerate(CLASS_ORDER)}
    y = train_df[TARGET].map(label_map).astype(int)

    train_fe = train_df.drop(columns=[TARGET]).copy()

    for df in [train_fe]:
        df["rainfall_x_moisture"] = df["Rainfall_mm"] * df["Soil_Moisture"]
        df["rainfall_per_prev_irrig"] = df["Rainfall_mm"] / (
            df["Previous_Irrigation_mm"] + 1.0
        )
        df["moisture_minus_prev_irrig"] = (
            df["Soil_Moisture"] - df["Previous_Irrigation_mm"]
        )
        df["water_source__irrigation_type"] = (
            df["Water_Source"].astype(str) + "__" + df["Irrigation_Type"].astype(str)
        )
        df["crop__season"] = (
            df["Crop_Type"].astype(str) + "__" + df["Season"].astype(str)
        )
        df["moisture_bucket"] = pd.qcut(
            df["Soil_Moisture"], q=8, duplicates="drop"
        ).astype(str)
        df["water_source__moisture_bucket"] = (
            df["Water_Source"].astype(str) + "__" + df["moisture_bucket"].astype(str)
        )

    all_features = [c for c in train_fe.columns if c != "id"]
    X_df = train_fe[all_features].copy()

    cat_features = [c for c in all_features if is_categorical(X_df[c])]
    num_features = [c for c in all_features if c not in cat_features]

    le_map = {}
    for col in cat_features:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
        le_map[col] = le

    scaler = StandardScaler()
    X_df[num_features] = scaler.fit_transform(X_df[num_features])

    X = X_df.values.astype(np.float32)
    print(f"Data loaded: X={X.shape}, classes={CLASS_ORDER}")
    return X, y


# ── Optuna objective ───────────────────────────────────────────────────────
def make_objective(X, y, n_folds=3):
    """Return an Optuna objective closure over the dataset."""

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    def objective(trial):
        params = {
            "iterations": trial.suggest_int("iterations", 500, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "depth": trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
            "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
            "random_strength": trial.suggest_float("random_strength", 0.0, 10.0),
            "border_count": trial.suggest_int("border_count", 32, 255),
            "grow_policy": trial.suggest_categorical(
                "grow_policy", ["SymmetricTree", "Lossguide"]
            ),
            "auto_class_weights": trial.suggest_categorical(
                "auto_class_weights", ["Balanced", "None"]
            ),
            # Fixed
            "loss_function": "MultiClass",
            "eval_metric": "Accuracy",
            "random_seed": SEED,
            "verbose": 0,
            "use_best_model": True,
        }
        # CatBoost expects Python None, not the string "None"
        if params["auto_class_weights"] == "None":
            params["auto_class_weights"] = None

        scores = []
        for tr_idx, val_idx in folds:
            X_tr, X_val = X[tr_idx], X[val_idx]
            y_tr, y_val = y.values[tr_idx], y.values[val_idx]

            model = CatBoostClassifier(**params)
            model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=80)

            preds = np.argmax(model.predict_proba(X_val), axis=1)
            scores.append(balanced_accuracy_score(y_val, preds))

        mean_score = float(np.mean(scores))
        return mean_score

    return objective


# ── Final 5-fold verification with best params ─────────────────────────────
def run_final_cv(best_params, X, y, n_folds=5):
    print("\n── Final 5-fold CV with best params ──────────────────────────────")
    params = best_params.copy()
    params["loss_function"] = "MultiClass"
    params["eval_metric"] = "Accuracy"
    params["random_seed"] = SEED
    params["verbose"] = 0
    params["use_best_model"] = True
    if params.get("auto_class_weights") == "None":
        params["auto_class_weights"] = None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    oof_proba = np.zeros((len(X), 3))
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y.values[tr_idx], y.values[val_idx]

        model = CatBoostClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=80)

        proba = model.predict_proba(X_val)
        oof_proba[val_idx] = proba
        fold_score = balanced_accuracy_score(y_val, np.argmax(proba, axis=1))
        scores.append(fold_score)
        print(f"  Fold {fold + 1}: {fold_score:.5f}")

    oof_score = balanced_accuracy_score(y.values, np.argmax(oof_proba, axis=1))
    print(f"\n  Mean fold score : {np.mean(scores):.5f} ± {np.std(scores):.5f}")
    print(f"  True OOF score  : {oof_score:.5f}  (vs current baseline 0.96842)")
    return oof_score


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials", type=int, default=60, help="Number of Optuna trials"
    )
    parser.add_argument(
        "--timeout", type=int, default=3600, help="Wall-clock timeout in seconds"
    )
    args = parser.parse_args()

    X, y = load_data()

    study = optuna.create_study(
        direction="maximize",
        study_name="catboost_s6e4",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0),
    )

    objective = make_objective(X, y, n_folds=3)

    print(f"Starting Optuna search: {args.trials} trials, timeout={args.timeout}s ...")
    print(
        "  (Each trial = 3-fold CatBoost CV on 630k rows; expect ~30-60s per trial)\n"
    )

    study.optimize(
        objective,
        n_trials=args.trials,
        timeout=args.timeout,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\nBest trial #{best.number}:")
    print(f"  3-fold score : {best.value:.5f}")
    print(f"  Params:")
    for k, v in best.params.items():
        print(f"    {k}: {v}")

    # Save results to JSON for later reference
    results = {
        "best_3fold_score": best.value,
        "best_params": best.params,
        "n_trials_completed": len(study.trials),
    }
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")

    # Run full 5-fold with best params to get true OOF
    final_oof = run_final_cv(best.params, X, y, n_folds=5)
    results["final_5fold_oof"] = final_oof
    with open(RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=2)

    print("\n── Copy these params into build_cat() in run_pipeline.py ──────────")
    clean_params = best.params.copy()
    if clean_params.get("auto_class_weights") == "None":
        clean_params["auto_class_weights"] = None
    print(json.dumps(clean_params, indent=4))


if __name__ == "__main__":
    main()
