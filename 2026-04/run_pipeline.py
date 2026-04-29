"""
S6E4 — Predicting Irrigation Need
Interaction-only pipeline (no pairwise target encoding)
with High-class threshold optimization on OOF predictions.
"""

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.isotonic import IsotonicRegression
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
test_df = pd.read_csv("./data/test.csv")
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

TARGET = "Irrigation_Need"
CLASS_ORDER = ["Low", "Medium", "High"]
label_map = {c: i for i, c in enumerate(CLASS_ORDER)}
inv_label_map = {i: c for c, i in label_map.items()}
y = train_df[TARGET].map(label_map).astype(int)

# ── Feature Engineering ────────────────────────────────────────────────────
train_fe = train_df.drop(columns=[TARGET]).copy()
test_fe = test_df.copy()

for df in [train_fe, test_fe]:
    # Numeric interactions
    df["rainfall_x_moisture"] = df["Rainfall_mm"] * df["Soil_Moisture"]
    df["rainfall_per_prev_irrig"] = df["Rainfall_mm"] / (
        df["Previous_Irrigation_mm"] + 1.0
    )
    df["moisture_minus_prev_irrig"] = df["Soil_Moisture"] - df["Previous_Irrigation_mm"]
    # Categorical combos
    df["water_source__irrigation_type"] = (
        df["Water_Source"].astype(str) + "__" + df["Irrigation_Type"].astype(str)
    )
    df["crop__season"] = df["Crop_Type"].astype(str) + "__" + df["Season"].astype(str)
    # Bucketed interaction
    df["moisture_bucket"] = pd.qcut(df["Soil_Moisture"], q=8, duplicates="drop").astype(
        str
    )
    df["water_source__moisture_bucket"] = (
        df["Water_Source"].astype(str) + "__" + df["moisture_bucket"].astype(str)
    )

all_features = [c for c in train_fe.columns if c != "id"]
X_train_base = train_fe[all_features].copy()
X_test_base = test_fe[all_features].copy()

cat_features = [c for c in all_features if is_categorical(X_train_base[c])]
num_features = [c for c in all_features if c not in cat_features]
print(f"Cat: {len(cat_features)}, Num: {len(num_features)}")

# ── Label-encode categoricals & scale numerics ─────────────────────────────
for col in cat_features:
    le = LabelEncoder()
    combo = pd.concat([X_train_base[col], X_test_base[col]], axis=0).astype(str)
    le.fit(combo)
    X_train_base[col] = le.transform(X_train_base[col].astype(str))
    X_test_base[col] = le.transform(X_test_base[col].astype(str))

scaler = StandardScaler()
X_train_base[num_features] = scaler.fit_transform(X_train_base[num_features])
X_test_base[num_features] = scaler.transform(X_test_base[num_features])

X_train_df = X_train_base.copy()
X_test_df = X_test_base.copy()

X_train = X_train_df.values.astype(np.float32)
X_test = X_test_df.values.astype(np.float32)
n_classes = len(CLASS_ORDER)
print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")


# ── Model Builders ─────────────────────────────────────────────────────────
def build_lgb(n_classes):
    return lgb.LGBMClassifier(
        n_estimators=1200,
        learning_rate=0.04,
        num_leaves=63,
        min_child_samples=20,
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
        n_estimators=450,
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
        iterations=900,
        learning_rate=0.04,
        depth=7,
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
        try:
            if name == "LGB":
                model.fit(
                    X_tr,
                    y_tr,
                    eval_set=[(X_val, y_val)],
                    eval_metric="multi_logloss",
                    callbacks=[lgb.early_stopping(100, verbose=False)],
                )
            elif name == "XGB":
                model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
            elif name == "CAT":
                model.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
            else:
                model.fit(X_tr, y_tr)
        except TypeError:
            model.fit(X_tr, y_tr)
        val_pred = model.predict_proba(X_val)
        oof_proba[val_idx] = val_pred
        test_proba += model.predict_proba(X_test) / n_folds
        score = balanced_accuracy_score(y_val, np.argmax(val_pred, axis=1))
        scores.append(score)
        print(f"  Fold {fold + 1}: {score:.5f}")
    overall = balanced_accuracy_score(y.values, np.argmax(oof_proba, axis=1))
    print(f"  [{name}] OOF: {overall:.5f}  (mean: {np.mean(scores):.5f})")
    return oof_proba, test_proba, overall


def apply_high_threshold(proba, high_idx, threshold):
    preds = np.argmax(proba, axis=1)
    preds[proba[:, high_idx] >= threshold] = high_idx
    return preds


def find_best_high_threshold(y_true, oof_proba, high_idx, grid=None):
    if grid is None:
        grid = np.linspace(0.15, 0.85, 71)

    base_preds = np.argmax(oof_proba, axis=1)
    base_score = balanced_accuracy_score(y_true, base_preds)

    best_t = 0.5
    best_score = base_score
    for t in grid:
        tuned_preds = apply_high_threshold(oof_proba, high_idx, float(t))
        score = balanced_accuracy_score(y_true, tuned_preds)
        if score > best_score:
            best_score = score
            best_t = float(t)
    return best_t, best_score, base_score


def fit_ovr_isotonic_calibrators(proba, y_true, n_classes):
    calibrators = []
    for cls in range(n_classes):
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(proba[:, cls], (y_true == cls).astype(int))
        calibrators.append(iso)
    return calibrators


def apply_ovr_isotonic_calibrators(proba, calibrators, eps=1e-12):
    calibrated = np.column_stack(
        [calibrators[cls].transform(proba[:, cls]) for cls in range(len(calibrators))]
    )
    calibrated = np.clip(calibrated, eps, 1.0)
    row_sum = calibrated.sum(axis=1, keepdims=True)
    return calibrated / np.clip(row_sum, eps, None)


def crossfit_multiclass_calibration(
    oof_proba, y_true, n_classes, n_splits=5, random_state=42
):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    calibrated_oof = np.zeros_like(oof_proba)

    for tr_idx, val_idx in skf.split(oof_proba, y_true):
        fold_calibrators = fit_ovr_isotonic_calibrators(
            oof_proba[tr_idx], y_true[tr_idx], n_classes
        )
        calibrated_oof[val_idx] = apply_ovr_isotonic_calibrators(
            oof_proba[val_idx], fold_calibrators
        )

    full_calibrators = fit_ovr_isotonic_calibrators(oof_proba, y_true, n_classes)
    return calibrated_oof, full_calibrators


# ── Train ──────────────────────────────────────────────────────────────────
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
total = lgb_score + xgb_score + cat_score
w_lgb, w_xgb, w_cat = lgb_score / total, xgb_score / total, cat_score / total
blend_oof = w_lgb * lgb_oof + w_xgb * xgb_oof + w_cat * cat_oof
blend_test = w_lgb * lgb_test + w_xgb * xgb_test + w_cat * cat_test
blend_score = balanced_accuracy_score(y.values, np.argmax(blend_oof, axis=1))
print(f"Weights → LGB:{w_lgb:.3f}, XGB:{w_xgb:.3f}, CAT:{w_cat:.3f}")
print(f"Blend OOF: {blend_score:.5f}")

# A fallback blend that ignores XGB if it is unstable.
lc_total = lgb_score + cat_score
w_lgb_lc, w_cat_lc = lgb_score / lc_total, cat_score / lc_total
blend_lc_oof = w_lgb_lc * lgb_oof + w_cat_lc * cat_oof
blend_lc_test = w_lgb_lc * lgb_test + w_cat_lc * cat_test
blend_lc_score = balanced_accuracy_score(y.values, np.argmax(blend_lc_oof, axis=1))
print(f"LGB+CAT Weights → LGB:{w_lgb_lc:.3f}, CAT:{w_cat_lc:.3f}")
print(f"LGB+CAT Blend OOF: {blend_lc_score:.5f}")

meta_train = np.hstack([lgb_oof, xgb_oof, cat_oof])
meta_test = np.hstack([lgb_test, xgb_test, cat_test])
meta_model = LogisticRegression(
    C=1.0, max_iter=1000, random_state=42, class_weight="balanced"
)
skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
meta_oof_pred = np.zeros(len(y), dtype=int)
meta_oof_proba = np.zeros((len(y), n_classes), dtype=float)
for tr_idx, val_idx in skf2.split(meta_train, y):
    meta_model.fit(meta_train[tr_idx], y.values[tr_idx])
    meta_oof_proba[val_idx] = meta_model.predict_proba(meta_train[val_idx])
    meta_oof_pred[val_idx] = np.argmax(meta_oof_proba[val_idx], axis=1)
meta_score = balanced_accuracy_score(y.values, meta_oof_pred)
print(f"Meta-LR OOF: {meta_score:.5f}")

meta_model.fit(meta_train, y)
meta_test_proba = meta_model.predict_proba(meta_test)

# ── High-class threshold optimization ──────────────────────────────────────
HIGH_CLASS_IDX = label_map["High"]

blend_t, blend_t_score, blend_base = find_best_high_threshold(
    y.values, blend_oof, high_idx=HIGH_CLASS_IDX
)
blend_lc_t, blend_lc_t_score, blend_lc_base = find_best_high_threshold(
    y.values, blend_lc_oof, high_idx=HIGH_CLASS_IDX
)
meta_t, meta_t_score, meta_base = find_best_high_threshold(
    y.values, meta_oof_proba, high_idx=HIGH_CLASS_IDX
)

print("\nHigh threshold search (class=High):")
print(
    f"  blend(all3): base={blend_base:.5f}, tuned={blend_t_score:.5f}, t={blend_t:.3f}"
)
print(
    f"  blend(LGB+CAT): base={blend_lc_base:.5f}, tuned={blend_lc_t_score:.5f}, t={blend_lc_t:.3f}"
)
print(f"  meta-LR: base={meta_base:.5f}, tuned={meta_t_score:.5f}, t={meta_t:.3f}")

# ── OOF-safe calibration + threshold optimization ─────────────────────────
blend_cal_oof, blend_calibrators = crossfit_multiclass_calibration(
    blend_oof, y.values, n_classes
)
blend_cal_test = apply_ovr_isotonic_calibrators(blend_test, blend_calibrators)

blend_lc_cal_oof, blend_lc_calibrators = crossfit_multiclass_calibration(
    blend_lc_oof, y.values, n_classes
)
blend_lc_cal_test = apply_ovr_isotonic_calibrators(blend_lc_test, blend_lc_calibrators)

meta_cal_oof, meta_calibrators = crossfit_multiclass_calibration(
    meta_oof_proba, y.values, n_classes
)
meta_cal_test = apply_ovr_isotonic_calibrators(meta_test_proba, meta_calibrators)

blend_cal_score = balanced_accuracy_score(y.values, np.argmax(blend_cal_oof, axis=1))
blend_lc_cal_score = balanced_accuracy_score(
    y.values, np.argmax(blend_lc_cal_oof, axis=1)
)
meta_cal_score = balanced_accuracy_score(y.values, np.argmax(meta_cal_oof, axis=1))

blend_cal_t, blend_cal_t_score, _ = find_best_high_threshold(
    y.values, blend_cal_oof, high_idx=HIGH_CLASS_IDX
)
blend_lc_cal_t, blend_lc_cal_t_score, _ = find_best_high_threshold(
    y.values, blend_lc_cal_oof, high_idx=HIGH_CLASS_IDX
)
meta_cal_t, meta_cal_t_score, _ = find_best_high_threshold(
    y.values, meta_cal_oof, high_idx=HIGH_CLASS_IDX
)

print("\nCalibration + high threshold search (class=High):")
print(
    f"  blend(all3): cal_argmax={blend_cal_score:.5f}, cal_tuned={blend_cal_t_score:.5f}, t={blend_cal_t:.3f}"
)
print(
    f"  blend(LGB+CAT): cal_argmax={blend_lc_cal_score:.5f}, cal_tuned={blend_lc_cal_t_score:.5f}, t={blend_lc_cal_t:.3f}"
)
print(
    f"  meta-LR: cal_argmax={meta_cal_score:.5f}, cal_tuned={meta_cal_t_score:.5f}, t={meta_cal_t:.3f}"
)

# ── Candidate selection ────────────────────────────────────────────────────
candidate_scores = {
    "lgb_argmax": lgb_score,
    "xgb_argmax": xgb_score,
    "cat_argmax": cat_score,
    "blend_all_argmax": blend_score,
    "blend_lgb_cat_argmax": blend_lc_score,
    "blend_all_high_threshold": blend_t_score,
    "blend_lgb_cat_high_threshold": blend_lc_t_score,
    "meta_argmax": meta_score,
    "meta_high_threshold": meta_t_score,
    "blend_all_calibrated_argmax": blend_cal_score,
    "blend_lgb_cat_calibrated_argmax": blend_lc_cal_score,
    "meta_calibrated_argmax": meta_cal_score,
    "blend_all_calibrated_high_threshold": blend_cal_t_score,
    "blend_lgb_cat_calibrated_high_threshold": blend_lc_cal_t_score,
    "meta_calibrated_high_threshold": meta_cal_t_score,
}

best_name = max(candidate_scores, key=candidate_scores.get)
best_oof = candidate_scores[best_name]

if best_name == "lgb_argmax":
    final_pred = np.argmax(lgb_test, axis=1)
elif best_name == "xgb_argmax":
    final_pred = np.argmax(xgb_test, axis=1)
elif best_name == "cat_argmax":
    final_pred = np.argmax(cat_test, axis=1)
elif best_name == "blend_all_argmax":
    final_pred = np.argmax(blend_test, axis=1)
elif best_name == "blend_lgb_cat_argmax":
    final_pred = np.argmax(blend_lc_test, axis=1)
elif best_name == "blend_all_high_threshold":
    final_pred = apply_high_threshold(blend_test, HIGH_CLASS_IDX, blend_t)
elif best_name == "blend_lgb_cat_high_threshold":
    final_pred = apply_high_threshold(blend_lc_test, HIGH_CLASS_IDX, blend_lc_t)
elif best_name == "meta_high_threshold":
    final_pred = apply_high_threshold(meta_test_proba, HIGH_CLASS_IDX, meta_t)
elif best_name == "blend_all_calibrated_argmax":
    final_pred = np.argmax(blend_cal_test, axis=1)
elif best_name == "blend_lgb_cat_calibrated_argmax":
    final_pred = np.argmax(blend_lc_cal_test, axis=1)
elif best_name == "meta_calibrated_argmax":
    final_pred = np.argmax(meta_cal_test, axis=1)
elif best_name == "blend_all_calibrated_high_threshold":
    final_pred = apply_high_threshold(blend_cal_test, HIGH_CLASS_IDX, blend_cal_t)
elif best_name == "blend_lgb_cat_calibrated_high_threshold":
    final_pred = apply_high_threshold(blend_lc_cal_test, HIGH_CLASS_IDX, blend_lc_cal_t)
elif best_name == "meta_calibrated_high_threshold":
    final_pred = apply_high_threshold(meta_cal_test, HIGH_CLASS_IDX, meta_cal_t)
else:
    final_pred = np.argmax(meta_test_proba, axis=1)

# ── Submission ─────────────────────────────────────────────────────────────
os.makedirs("./submissions", exist_ok=True)
final_labels = [inv_label_map[p] for p in final_pred]

print("\nCandidate OOF summary:")
for k, v in sorted(candidate_scores.items(), key=lambda kv: kv[1], reverse=True):
    print(f"  {k:<30} {v:.5f}")

print(f"\nUsing: {best_name} (OOF={best_oof:.5f})")
sub = pd.DataFrame({"id": test_df["id"], "Irrigation_Need": final_labels})
sub.to_csv("./submissions/submission_v4_threshold_calibration.csv", index=False)
print("Saved: ./submissions/submission_v4_threshold_calibration.csv")
print(sub["Irrigation_Need"].value_counts())
