import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")

models = {
    "Logistic Regression": (joblib.load("models/logreg.pkl"), True),
    "Random Forest": (joblib.load("models/random_forest.pkl"), False),
    "XGBoost": (joblib.load("models/xgboost.pkl"), False),
    "XGBoost (calibrated)": (joblib.load("models/xgboost_calibrated.pkl"), False),
    "LightGBM": (joblib.load("models/lgbm.pkl"), False),
}

N_BOOTSTRAP = 1000
rng = np.random.RandomState(42)
y_test_arr = y_test.values
n = len(y_test_arr)

# Pre-compute predicted probabilities once per model (fast), then bootstrap indices
proba_by_model = {}
for name, (model, needs_scaling) in models.items():
    X_eval = scaler.transform(X_test) if needs_scaling else X_test
    proba_by_model[name] = model.predict_proba(X_eval)[:, 1]

# Bootstrap resampling: same resampled indices reused across models for paired comparison
boot_indices = [rng.choice(n, size=n, replace=True) for _ in range(N_BOOTSTRAP)]

results = []
auc_boot_by_model = {}
brier_boot_by_model = {}

for name, proba in proba_by_model.items():
    aucs, briers = [], []
    for idx in boot_indices:
        y_b, p_b = y_test_arr[idx], proba[idx]
        if len(np.unique(y_b)) < 2:
            continue  # skip degenerate resamples with only one class
        aucs.append(roc_auc_score(y_b, p_b))
        briers.append(brier_score_loss(y_b, p_b))
    aucs, briers = np.array(aucs), np.array(briers)
    auc_boot_by_model[name] = aucs
    brier_boot_by_model[name] = briers

    auc_point = roc_auc_score(y_test_arr, proba)
    brier_point = brier_score_loss(y_test_arr, proba)
    auc_ci = np.percentile(aucs, [2.5, 97.5])
    brier_ci = np.percentile(briers, [2.5, 97.5])

    results.append({
        "Model": name,
        "ROC_AUC": round(auc_point, 4),
        "ROC_AUC_CI_low": round(auc_ci[0], 4),
        "ROC_AUC_CI_high": round(auc_ci[1], 4),
        "Brier": round(brier_point, 4),
        "Brier_CI_low": round(brier_ci[0], 4),
        "Brier_CI_high": round(brier_ci[1], 4),
    })

summary = pd.DataFrame(results)
print("\n=== Bootstrap 95% Confidence Intervals (1000 resamples) ===")
print(summary.to_string(index=False))
summary.to_csv("figures/bootstrap_ci_summary.csv", index=False)

# --- Pairwise significance test: is the AUC difference between models real? ---
print("\n=== Pairwise AUC Difference (paired bootstrap, same resamples) ===")
pairs = [
    ("Logistic Regression", "Random Forest"),
    ("Logistic Regression", "XGBoost"),
    ("Random Forest", "XGBoost"),
    ("XGBoost", "XGBoost (calibrated)"),
    ("XGBoost", "LightGBM"),
    ("Random Forest", "LightGBM"),
]
pairwise_results = []
for a, b in pairs:
    diff = auc_boot_by_model[a] - auc_boot_by_model[b]
    ci = np.percentile(diff, [2.5, 97.5])
    significant = not (ci[0] <= 0 <= ci[1])
    point_diff = summary.loc[summary["Model"] == a, "ROC_AUC"].values[0] - \
                 summary.loc[summary["Model"] == b, "ROC_AUC"].values[0]
    print(f"{a} vs {b}: diff={point_diff:+.4f}, 95% CI=[{ci[0]:+.4f}, {ci[1]:+.4f}], "
          f"{'SIGNIFICANT' if significant else 'not significant'} (CI {'excludes' if significant else 'includes'} 0)")
    pairwise_results.append({
        "Comparison": f"{a} vs {b}", "AUC_diff": round(point_diff, 4),
        "CI_low": round(ci[0], 4), "CI_high": round(ci[1], 4), "Significant": significant
    })
pd.DataFrame(pairwise_results).to_csv("figures/pairwise_significance.csv", index=False)

# --- Plot: ROC-AUC with 95% CI error bars ---
fig, ax = plt.subplots(figsize=(8, 5))
names = summary["Model"].tolist()
points = summary["ROC_AUC"].values
lower_err = points - summary["ROC_AUC_CI_low"].values
upper_err = summary["ROC_AUC_CI_high"].values - points
ax.errorbar(names, points, yerr=[lower_err, upper_err], fmt="o", capsize=6,
            color="#028090", markersize=8, linewidth=2)
ax.set_ylabel("ROC-AUC")
ax.set_title("ROC-AUC with 95% Bootstrap Confidence Intervals")
ax.set_ylim(0.78, 0.86)
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig("figures/bootstrap_ci_plot.png", dpi=150)
plt.close()
print("\nSaved figures/bootstrap_ci_summary.csv, pairwise_significance.csv, bootstrap_ci_plot.png")