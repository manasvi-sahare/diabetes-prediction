import time
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from scipy.stats import spearmanr
import shap

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")
feature_names = X_train.columns.tolist()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def run_stability(model_name, fit_and_explain_fn):
    print(f"\n{'='*70}\n{model_name}: SHAP Stability Across 5 CV Folds\n{'='*70}")
    fold_importances, fold_top10, fold_top5 = [], [], []

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        t0 = time.time()
        Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
        ytr = y_train.iloc[tr_idx]

        mean_abs_shap = fit_and_explain_fn(Xtr, ytr, Xval)
        importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
        fold_importances.append(importance)
        fold_top10.append(set(importance.head(10).index))
        fold_top5.append(set(importance.head(5).index))
        print(f"  Fold {fold+1} done ({time.time()-t0:.1f}s) — top 3: {list(importance.head(3).index)}")

    importance_df = pd.DataFrame(fold_importances).T
    importance_df.columns = [f"Fold{i+1}" for i in range(5)]
    rank_df = importance_df.rank(ascending=False)

    correlations = []
    for i in range(5):
        for j in range(i + 1, 5):
            rho, _ = spearmanr(rank_df[f"Fold{i+1}"], rank_df[f"Fold{j+1}"])
            correlations.append(rho)

    def mean_jaccard(sets):
        js = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                js.append(len(sets[i] & sets[j]) / len(sets[i] | sets[j]))
        return np.mean(js)

    result = {
        "Model": model_name,
        "mean_spearman_rho": round(np.mean(correlations), 4),
        "top5_jaccard": round(mean_jaccard(fold_top5), 4),
        "top10_jaccard": round(mean_jaccard(fold_top10), 4),
    }
    print(f"  Mean Spearman rho: {result['mean_spearman_rho']}  |  "
          f"Top-5 Jaccard: {result['top5_jaccard']}  |  Top-10 Jaccard: {result['top10_jaccard']}")
    return result


# ---------------- Logistic Regression ----------------
def lr_fit_explain(Xtr, ytr, Xval):
    fold_scaler = StandardScaler()
    Xtr_s = fold_scaler.fit_transform(Xtr)
    Xval_s = fold_scaler.transform(Xval)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    model.fit(Xtr_s, ytr)
    sample_idx = np.random.RandomState(42).choice(len(Xval_s), size=min(3000, len(Xval_s)), replace=False)
    explainer = shap.LinearExplainer(model, Xtr_s)
    shap_values = explainer(Xval_s[sample_idx])
    return np.abs(shap_values.values).mean(axis=0)

# ---------------- Random Forest ----------------
def rf_fit_explain(Xtr, ytr, Xval):
    model = RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=20,
                                    max_features="sqrt", class_weight="balanced", n_jobs=-1, random_state=42)
    model.fit(Xtr, ytr)
    sample_idx = np.random.RandomState(42).choice(Xval.index, size=min(3000, len(Xval)), replace=False)
    X_sample = Xval.loc[sample_idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    vals = shap_values.values
    if vals.ndim == 3:  # RF binary classifiers sometimes return (n, features, 2)
        vals = vals[:, :, 1]
    return np.abs(vals).mean(axis=0)

# ---------------- LightGBM ----------------
def lgbm_fit_explain(Xtr, ytr, Xval):
    neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
    model = LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                            scale_pos_weight=neg / pos, n_jobs=-1, random_state=42, verbose=-1)
    model.fit(Xtr, ytr)
    sample_idx = np.random.RandomState(42).choice(Xval.index, size=min(3000, len(Xval)), replace=False)
    X_sample = Xval.loc[sample_idx]
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    return np.abs(shap_values.values).mean(axis=0)


results = []
results.append(run_stability("Logistic Regression", lr_fit_explain))
results.append(run_stability("Random Forest", rf_fit_explain))
results.append(run_stability("LightGBM", lgbm_fit_explain))

summary = pd.DataFrame(results)
print("\n" + "=" * 70)
print(summary.to_string(index=False))
summary.to_csv("figures/shap_stability_lr_rf_lgbm.csv", index=False)
print("\nSaved figures/shap_stability_lr_rf_lgbm.csv")
print("(XGBoost result already saved in figures/shap_stability_summary.csv from script 18)")