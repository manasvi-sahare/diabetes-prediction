import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from scipy.stats import spearmanr
import shap

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")
feature_names = X_train.columns.tolist()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_importances = []  # list of pd.Series, one per fold: feature -> mean|SHAP|
fold_top10 = []
fold_top5 = []

print("=== SHAP Stability Across 5 CV Folds ===\n")

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
    t0 = time.time()
    Xtr, Xval = X_train.iloc[tr_idx], X_train.iloc[val_idx]
    ytr, yval = y_train.iloc[tr_idx], y_train.iloc[val_idx]

    neg, pos = (ytr == 0).sum(), (ytr == 1).sum()
    spw = neg / pos

    model = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                           scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1, random_state=42)
    model.fit(Xtr, ytr)

    # Sample validation fold for SHAP (same sample-size convention as 11_shap_explainability.py)
    sample_idx = np.random.RandomState(42).choice(Xval.index, size=min(3000, len(Xval)), replace=False)
    X_sample = Xval.loc[sample_idx]

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_sample)
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)

    importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    fold_importances.append(importance)
    fold_top10.append(set(importance.head(10).index))
    fold_top5.append(set(importance.head(5).index))

    print(f"Fold {fold+1} ({time.time()-t0:.1f}s) — Top 10 features:")
    for rank, (feat, val) in enumerate(importance.head(10).items(), 1):
        print(f"  {rank:2d}. {feat:<22} {val:.4f}")
    print()

# ---------------- Aggregate: mean SHAP importance across folds ----------------
importance_df = pd.DataFrame(fold_importances).T  # rows=features, cols=fold0..fold4
importance_df.columns = [f"Fold{i+1}" for i in range(5)]
importance_df["Mean"] = importance_df.mean(axis=1)
importance_df["Std"] = importance_df[[f"Fold{i+1}" for i in range(5)]].std(axis=1)
importance_df = importance_df.sort_values("Mean", ascending=False)

print("=" * 70)
print("Mean SHAP importance across all 5 folds (sorted):")
print(importance_df[["Mean", "Std"]].round(4).to_string())
importance_df.round(4).to_csv("figures/shap_stability_by_fold.csv")

# ---------------- Rank correlation between folds ----------------
rank_df = importance_df[[f"Fold{i+1}" for i in range(5)]].rank(ascending=False)
print("\n" + "=" * 70)
print("Pairwise Spearman rank correlation between folds:")
correlations = []
for i in range(5):
    for j in range(i + 1, 5):
        rho, _ = spearmanr(rank_df[f"Fold{i+1}"], rank_df[f"Fold{j+1}"])
        correlations.append(rho)
        print(f"  Fold {i+1} vs Fold {j+1}: rho = {rho:.4f}")
print(f"\nMean pairwise Spearman rho: {np.mean(correlations):.4f} (+/- {np.std(correlations):.4f})")

# ---------------- Top-5 / Top-10 consistency (Jaccard overlap) ----------------
def mean_jaccard(sets):
    jaccards = []
    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            inter = len(sets[i] & sets[j])
            union = len(sets[i] | sets[j])
            jaccards.append(inter / union)
    return np.mean(jaccards), np.std(jaccards)

top5_mean, top5_std = mean_jaccard(fold_top5)
top10_mean, top10_std = mean_jaccard(fold_top10)

print(f"\nTop-5 feature-set consistency (mean pairwise Jaccard):  {top5_mean:.4f} (+/- {top5_std:.4f})")
print(f"Top-10 feature-set consistency (mean pairwise Jaccard): {top10_mean:.4f} (+/- {top10_std:.4f})")

# Features that appear in ALL 5 folds' top 10
always_top10 = set.intersection(*fold_top10)
print(f"\nFeatures in top-10 across ALL 5 folds ({len(always_top10)}): {sorted(always_top10)}")

summary = pd.DataFrame([{
    "mean_spearman_rho": round(np.mean(correlations), 4),
    "std_spearman_rho": round(np.std(correlations), 4),
    "top5_jaccard_mean": round(top5_mean, 4),
    "top10_jaccard_mean": round(top10_mean, 4),
    "n_features_always_in_top10": len(always_top10),
}])
summary.to_csv("figures/shap_stability_summary.csv", index=False)

# ---------------- Plot: mean SHAP importance with std error bars across folds ----------------
fig, ax = plt.subplots(figsize=(8, 6))
top10_features = importance_df.head(10).iloc[::-1]
ax.barh(top10_features.index, top10_features["Mean"], xerr=top10_features["Std"],
        color="#028090", capsize=4)
ax.set_xlabel("Mean |SHAP value| across 5 folds (error bars = std across folds)")
ax.set_title("SHAP Feature Importance Stability Across CV Folds")
plt.tight_layout()
plt.savefig("figures/shap_stability_plot.png", dpi=150)
plt.close()

print("\nSaved figures/shap_stability_by_fold.csv, shap_stability_summary.csv, shap_stability_plot.png")