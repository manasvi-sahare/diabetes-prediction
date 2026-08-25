import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, brier_score_loss

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

results = []

# ---------------- Logistic Regression ----------------
print("=== Logistic Regression: 5-fold CV ===")
t0 = time.time()
aucs, briers = [], []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    fold_scaler = StandardScaler()
    Xtr_s = fold_scaler.fit_transform(Xtr)
    Xval_s = fold_scaler.transform(Xval)

    m = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    m.fit(Xtr_s, ytr)
    proba = m.predict_proba(Xval_s)[:, 1]
    auc, brier = roc_auc_score(yval, proba), brier_score_loss(yval, proba)
    aucs.append(auc); briers.append(brier)
    print(f"  Fold {fold+1}: AUC={auc:.4f}  Brier={brier:.4f}")
print(f"  Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  ({time.time()-t0:.1f}s)")
results.append({"Model": "Logistic Regression", "CV_AUC_mean": np.mean(aucs), "CV_AUC_std": np.std(aucs),
                 "CV_Brier_mean": np.mean(briers), "CV_Brier_std": np.std(briers)})

# ---------------- Random Forest ----------------
print("\n=== Random Forest: 5-fold CV ===")
t0 = time.time()
aucs, briers = [], []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    m = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=20,
                                max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xval)[:, 1]
    auc, brier = roc_auc_score(yval, proba), brier_score_loss(yval, proba)
    aucs.append(auc); briers.append(brier)
    print(f"  Fold {fold+1}: AUC={auc:.4f}  Brier={brier:.4f}")
print(f"  Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  ({time.time()-t0:.1f}s)")
results.append({"Model": "Random Forest", "CV_AUC_mean": np.mean(aucs), "CV_AUC_std": np.std(aucs),
                 "CV_Brier_mean": np.mean(briers), "CV_Brier_std": np.std(briers)})

# ---------------- XGBoost ----------------
print("\n=== XGBoost: 5-fold CV ===")
t0 = time.time()
aucs, briers = [], []
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
    spw = neg / pos

    m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                       scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1, random_state=42)
    m.fit(Xtr, ytr)
    proba = m.predict_proba(Xval)[:, 1]
    auc, brier = roc_auc_score(yval, proba), brier_score_loss(yval, proba)
    aucs.append(auc); briers.append(brier)
    print(f"  Fold {fold+1}: AUC={auc:.4f}  Brier={brier:.4f}")
print(f"  Mean AUC: {np.mean(aucs):.4f} +/- {np.std(aucs):.4f}  ({time.time()-t0:.1f}s)")
results.append({"Model": "XGBoost", "CV_AUC_mean": np.mean(aucs), "CV_AUC_std": np.std(aucs),
                 "CV_Brier_mean": np.mean(briers), "CV_Brier_std": np.std(briers)})

# ---------------- Save ----------------
summary = pd.DataFrame(results).round(4)
print("\n" + "=" * 60)
print(summary.to_string(index=False))
summary.to_csv("figures/cv_all_models_summary.csv", index=False)
print("\nSaved figures/cv_all_models_summary.csv")