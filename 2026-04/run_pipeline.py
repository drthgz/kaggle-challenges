"""First iteration pipeline for S6E4 — Predicting Irrigation Need."""

import os, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── Load Data ──────────────────────────────────────────────────────────────
train_df = pd.read_csv("./data/train.csv")
test_df = pd.read_csv("./data/test.csv")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

TARGET = "Irrigation_Need"
CLASS_ORDER = ["Low", "Medium", "High"]
label_map = {c: i for i, c in enumerate(CLASS_ORDER)}
inv_label_map = {i: c for c, i in label_map.items()}

y = train_df[TARGET].map(label_map).astype(int)


# ── Preprocessing ──────────────────────────────────────────────────────────
def is_categorical(series):
    return series.dtype == "object" or pd.api.types.is_string_dtype(series)


drop_cols = [TARGET, "id"]
all_features = [c for c in train_df.columns if c not in drop_cols]

X_train_raw = train_df[all_features].copy()
X_test_raw = test_df[[c for c in all_features if c in test_df.columns]].copy()

cat_features = [c for c in all_features if is_categorical(X_train_raw[c])]
num_features = [c for c in all_features if not is_categorical(X_train_raw[c])]
print(f"Cat: {cat_features}")
print(f"Num: {num_features}")

le_dict = {}
for col in cat_features:
    le = LabelEncoder()
    combined = pd.concat([X_train_raw[col], X_test_raw[col]], axis=0).astype(str)
    le.fit(combined)
    X_train_raw[col] = le.transform(X_train_raw[col].astype(str))
    X_test_raw[col] = le.transform(X_test_raw[col].astype(str))
    le_dict[col] = le

scaler = StandardScaler()
X_train_raw[num_features] = scaler.fit_transform(X_train_raw[num_features])
X_test_raw[num_features] = scaler.transform(X_test_raw[num_features])

X_train = X_train_raw.values.astype(np.float32)
X_test = X_test_raw.values.astype(np.float32)
n_classes = len(CLASS_ORDER)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")


# ── Model Builders ─────────────────────────────────────────────────────────
def build_lgb(n_classes):
    return lgb.LGBMClassifier(
        n_estimators=800,
        learning_rate=0.05,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        objective="multiclass",
        num_class=n_classes,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )


def build_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=800,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def build_cat(n_classes):
    return CatBoostClassifier(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        eval_metric="Accuracy",
        auto_class_weights="Balanced",
        random_seed=42,
        verbose=0,
    )


# ── CV helper ──────────────────────────────────────────────────────────────
def run_cv(model_fn, X, y, X_test, n_classes, n_folds=5, name="model"):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X), n_classes))
    test_proba = np.zeros((len(X_test), n_classes))
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y.values[tr_idx], y.values[val_idx]
        model = model_fn()
        model.fit(X_tr, y_tr)
        val_pred = model.predict_proba(X_val)
        oof_proba[val_idx] = val_pred
        test_proba += model.predict_proba(X_test) / n_folds
        score = balanced_accuracy_score(y_val, np.argmax(val_pred, axis=1))
        scores.append(score)
        print(f"  Fold {fold + 1}: {score:.5f}")
    overall = balanced_accuracy_score(y.values, np.argmax(oof_proba, axis=1))
    print(f"  [{name}] OOF: {overall:.5f}  (mean folds: {np.mean(scores):.5f})")
    return oof_proba, test_proba, overall


# ── Train models ───────────────────────────────────────────────────────────
print("\n=== LightGBM ===")
lgb_oof, lgb_test, lgb_score = run_cv(
    lambda: build_lgb(n_classes), X_train, y, X_test, n_classes, name="LGB"
)

print("\n=== XGBoost ===")
xgb_oof, xgb_test, xgb_score = run_cv(
    lambda: build_xgb(n_classes), X_train, y, X_test, n_classes, name="XGB"
)

print("\n=== CatBoost ===")
cat_oof, cat_test, cat_score = run_cv(
    lambda: build_cat(n_classes), X_train, y, X_test, n_classes, name="CAT"
)

print(f"\nSummary: LGB={lgb_score:.5f}, XGB={xgb_score:.5f}, CAT={cat_score:.5f}")

# ── Ensemble ───────────────────────────────────────────────────────────────
total_score = lgb_score + xgb_score + cat_score
w_lgb, w_xgb, w_cat = (
    lgb_score / total_score,
    xgb_score / total_score,
    cat_score / total_score,
)
blend_oof = w_lgb * lgb_oof + w_xgb * xgb_oof + w_cat * cat_oof
blend_test = w_lgb * lgb_test + w_xgb * xgb_test + w_cat * cat_test
blend_score = balanced_accuracy_score(y.values, np.argmax(blend_oof, axis=1))
print(f"Weights → LGB:{w_lgb:.3f}, XGB:{w_xgb:.3f}, CAT:{w_cat:.3f}")
print(f"Blend OOF: {blend_score:.5f}")

# Meta-LR
meta_train = np.hstack([lgb_oof, xgb_oof, cat_oof])
meta_test = np.hstack([lgb_test, xgb_test, cat_test])
meta_model = LogisticRegression(
    C=1.0, max_iter=1000, random_state=42, class_weight="balanced"
)
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_oof_pred = np.zeros(len(y))
for tr_idx, val_idx in skf2.split(meta_train, y):
    meta_model.fit(meta_train[tr_idx], y.values[tr_idx])
    meta_oof_pred[val_idx] = meta_model.predict(meta_train[val_idx])
meta_score = balanced_accuracy_score(y.values, meta_oof_pred)
print(f"Meta-LR OOF: {meta_score:.5f}")

meta_model.fit(meta_train, y)
meta_test_pred = meta_model.predict(meta_test)

# ── Submission ─────────────────────────────────────────────────────────────
os.makedirs("./submissions", exist_ok=True)
if meta_score >= blend_score:
    final_labels = [inv_label_map[p] for p in meta_test_pred]
    chosen = "meta-LR"
else:
    final_labels = [inv_label_map[p] for p in np.argmax(blend_test, axis=1)]
    chosen = "weighted blend"

print(f"\nUsing: {chosen}  (meta={meta_score:.5f}, blend={blend_score:.5f})")
sub = pd.DataFrame({"id": test_df["id"], "Irrigation_Need": final_labels})
sub.to_csv("./submissions/submission.csv", index=False)
print("Saved: ./submissions/submission.csv")
print(sub["Irrigation_Need"].value_counts())
