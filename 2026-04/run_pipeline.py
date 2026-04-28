"""
S6E4 — Predicting Irrigation Need
Full pipeline: interactions + OOF-safe pairwise target encoding + LGB/XGB/CAT + meta-LR stack
"""
import os, warnings
from itertools import combinations
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

# ── Helpers ────────────────────────────────────────────────────────────────
def is_categorical(series):
    return series.dtype == "object" or (
        hasattr(series.dtype, "name") and series.dtype.name in ("string", "str")
    )

# ── Load Data ──────────────────────────────────────────────────────────────
train_df = pd.read_csv("./data/train.csv")
test_df  = pd.read_csv("./data/test.csv")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

TARGET = "Irrigation_Need"
CLASS_ORDER   = ["Low", "Medium", "High"]
label_map     = {c: i for i, c in enumerate(CLASS_ORDER)}
inv_label_map = {i: c for c, i in label_map.items()}
y = train_df[TARGET].map(label_map).astype(int)

# ── Feature Engineering ────────────────────────────────────────────────────
train_fe = train_df.drop(columns=[TARGET]).copy()
test_fe  = test_df.copy()

for df in [train_fe, test_fe]:
    # Numeric interactions
    df["rainfall_x_moisture"]     = df["Rainfall_mm"] * df["Soil_Moisture"]
    df["rainfall_per_prev_irrig"] = df["Rainfall_mm"] / (df["Previous_Irrigation_mm"] + 1.0)
    df["moisture_minus_prev_irrig"] = df["Soil_Moisture"] - df["Previous_Irrigation_mm"]
    # Categorical combos
    df["water_source__irrigation_type"] = (
        df["Water_Source"].astype(str) + "__" + df["Irrigation_Type"].astype(str)
    )
    df["crop__season"] = df["Crop_Type"].astype(str) + "__" + df["Season"].astype(str)
    # Bucketed interaction
    df["moisture_bucket"] = pd.qcut(df["Soil_Moisture"], q=8, duplicates="drop").astype(str)
    df["water_source__moisture_bucket"] = (
        df["Water_Source"].astype(str) + "__" + df["moisture_bucket"].astype(str)
    )

all_features = [c for c in train_fe.columns if c != "id"]
X_train_base = train_fe[all_features].copy()
X_test_base  = test_fe[all_features].copy()

cat_features = [c for c in all_features if is_categorical(X_train_base[c])]
num_features = [c for c in all_features if c not in cat_features]
print(f"Cat: {len(cat_features)}, Num: {len(num_features)}")

# ── OOF-Safe Pairwise Target Encoding ─────────────────────────────────────
def make_pairwise_te(train_cat, test_cat, y_series, cat_cols, n_splits=5, random_state=42):
    skf   = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pairs = list(combinations(sorted(y_series.unique()), 2))
    te_train = pd.DataFrame(index=train_cat.index)
    te_test  = pd.DataFrame(index=test_cat.index)

    for col in cat_cols:
        col_train = train_cat[col].astype(str)
        col_test  = test_cat[col].astype(str)
        for a, b in pairs:
            feat     = f"pte_{col}_{a}vs{b}"
            oof_vals = np.zeros(len(train_cat), dtype=float)
            for tr_idx, val_idx in skf.split(train_cat, y_series):
                y_tr     = y_series.iloc[tr_idx]
                x_tr     = col_train.iloc[tr_idx]
                x_val    = col_train.iloc[val_idx]
                pm       = y_tr.isin([a, b])
                prior    = (y_tr[pm] == a).mean()
                mapping  = (pd.DataFrame({"x": x_tr[pm], "ya": (y_tr[pm] == a).astype(int)})
                              .groupby("x")["ya"].mean())
                oof_vals[val_idx] = x_val.map(mapping).fillna(prior).values
            pm_full  = y_series.isin([a, b])
            prior_f  = (y_series[pm_full] == a).mean()
            full_map = (pd.DataFrame({"x": col_train[pm_full],
                                      "ya": (y_series[pm_full] == a).astype(int)})
                          .groupby("x")["ya"].mean())
            te_train[feat] = oof_vals
            te_test[feat]  = col_test.map(full_map).fillna(prior_f).values
    return te_train, te_test

print("Building pairwise target encoding...")
te_train_df, te_test_df = make_pairwise_te(
    X_train_base[cat_features], X_test_base[cat_features],
    y, cat_features, n_splits=5, random_state=42
)
print(f"Pairwise TE features: {te_train_df.shape[1]}")

# ── Label-encode categoricals & scale numerics ─────────────────────────────
for col in cat_features:
    le = LabelEncoder()
    combo = pd.concat([X_train_base[col], X_test_base[col]], axis=0).astype(str)
    le.fit(combo)
    X_train_base[col] = le.transform(X_train_base[col].astype(str))
    X_test_base[col]  = le.transform(X_test_base[col].astype(str))

scaler = StandardScaler()
X_train_base[num_features] = scaler.fit_transform(X_train_base[num_features])
X_test_base[num_features]  = scaler.transform(X_test_base[num_features])

X_train_df = pd.concat([X_train_base, te_train_df], axis=1)
X_test_df  = pd.concat([X_test_base,  te_test_df],  axis=1)

X_train = X_train_df.values.astype(np.float32)
X_test  = X_test_df.values.astype(np.float32)
n_classes = len(CLASS_ORDER)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ── Model Builders ─────────────────────────────────────────────────────────
def build_lgb(n_classes):
    return lgb.LGBMClassifier(
        n_estimators=1200, learning_rate=0.04, num_leaves=63,
        min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        objective="multiclass", num_class=n_classes,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
    )

def build_xgb(n_classes):
    return xgb.XGBClassifier(
        n_estimators=450, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=n_classes,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
    )

def build_cat(n_classes):
    return CatBoostClassifier(
        iterations=900, learning_rate=0.04, depth=7,
        loss_function="MultiClass", eval_metric="Accuracy",
        auto_class_weights="Balanced", random_seed=42, verbose=0,
    )

# ── CV helper ──────────────────────────────────────────────────────────────
def run_cv(model_fn, X, y, X_test, n_classes, n_folds=5, name="model"):
    skf       = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_proba = np.zeros((len(X), n_classes))
    test_proba= np.zeros((len(X_test), n_classes))
    scores    = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y.values[tr_idx], y.values[val_idx]
        model = model_fn()
        try:
            if name == "LGB":
                model.fit(X_tr, y_tr,
                          eval_set=[(X_val, y_val)],
                          eval_metric="multi_logloss",
                          callbacks=[lgb.early_stopping(100, verbose=False)])
            elif name == "XGB":
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            elif name == "CAT":
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
            else:
                model.fit(X_tr, y_tr)
        except TypeError:
            model.fit(X_tr, y_tr)
        val_pred  = model.predict_proba(X_val)
        oof_proba[val_idx] = val_pred
        test_proba += model.predict_proba(X_test) / n_folds
        score = balanced_accuracy_score(y_val, np.argmax(val_pred, axis=1))
        scores.append(score)
        print(f"  Fold {fold+1}: {score:.5f}")
    overall = balanced_accuracy_score(y.values, np.argmax(oof_proba, axis=1))
    print(f"  [{name}] OOF: {overall:.5f}  (mean: {np.mean(scores):.5f})")
    return oof_proba, test_proba, overall

# ── Train ──────────────────────────────────────────────────────────────────
print("\n=== LightGBM ===")
lgb_oof, lgb_test, lgb_score = run_cv(lambda: build_lgb(n_classes), X_train, y, X_test, n_classes, name="LGB")

print("\n=== XGBoost ===")
xgb_oof, xgb_test, xgb_score = run_cv(lambda: build_xgb(n_classes), X_train, y, X_test, n_classes, name="XGB")

print("\n=== CatBoost ===")
cat_oof, cat_test, cat_score = run_cv(lambda: build_cat(n_classes), X_train, y, X_test, n_classes, name="CAT")

print(f"\nSummary: LGB={lgb_score:.5f}, XGB={xgb_score:.5f}, CAT={cat_score:.5f}")

# ── Ensemble ───────────────────────────────────────────────────────────────
total = lgb_score + xgb_score + cat_score
w_lgb, w_xgb, w_cat = lgb_score/total, xgb_score/total, cat_score/total
blend_oof  = w_lgb*lgb_oof  + w_xgb*xgb_oof  + w_cat*cat_oof
blend_test = w_lgb*lgb_test + w_xgb*xgb_test + w_cat*cat_test
blend_score = balanced_accuracy_score(y.values, np.argmax(blend_oof, axis=1))
print(f"Weights → LGB:{w_lgb:.3f}, XGB:{w_xgb:.3f}, CAT:{w_cat:.3f}")
print(f"Blend OOF: {blend_score:.5f}")

meta_train = np.hstack([lgb_oof, xgb_oof, cat_oof])
meta_test  = np.hstack([lgb_test, xgb_test, cat_test])
meta_model = LogisticRegression(C=1.0, max_iter=1000, random_state=42, class_weight="balanced")
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
    final_labels, chosen = [inv_label_map[p] for p in meta_test_pred], "meta-LR"
else:
    final_labels, chosen = [inv_label_map[p] for p in np.argmax(blend_test, axis=1)], "blend"

print(f"\nUsing: {chosen}  (meta={meta_score:.5f}, blend={blend_score:.5f})")
sub = pd.DataFrame({"id": test_df["id"], "Irrigation_Need": final_labels})
sub.to_csv("./submissions/submission_v2.csv", index=False)
print("Saved: ./submissions/submission_v2.csv")
print(sub["Irrigation_Need"].value_counts())
