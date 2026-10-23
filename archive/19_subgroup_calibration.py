import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_score, recall_score, f1_score
)
from sklearn.calibration import calibration_curve

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")
xgb_cal = joblib.load("models/xgboost_calibrated.pkl")

proba_cal = xgb_cal.predict_proba(X_test)[:, 1]
TUNED_THRESHOLD = 0.245


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Weighted average |predicted - observed| across equal-width probability bins."""
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bin_edges[1:-1])
    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_indices == b
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(bin_conf - bin_acc)
    return ece


def subgroup_metrics(mask, group_name, group_label):
    y_g = y_test[mask].values
    p_g = proba_cal[mask.values]
    pred_g = (p_g >= TUNED_THRESHOLD).astype(int)

    if len(np.unique(y_g)) < 2:
        return None

    return {
        "Group": group_name,
        "Subgroup": group_label,
        "N": int(mask.sum()),
        "Positive_rate": round(y_g.mean(), 4),
        "ROC_AUC": round(roc_auc_score(y_g, p_g), 4),
        "PR_AUC": round(average_precision_score(y_g, p_g), 4),
        "Brier": round(brier_score_loss(y_g, p_g), 4),
        "ECE": round(expected_calibration_error(y_g, p_g, n_bins=10), 4),
        "Precision": round(precision_score(y_g, pred_g, zero_division=0), 4),
        "Recall": round(recall_score(y_g, pred_g, zero_division=0), 4),
        "F1": round(f1_score(y_g, pred_g, zero_division=0), 4),
    }


results = []

for val, label in [(0, "Female"), (1, "Male")]:
    r = subgroup_metrics(X_test["Sex"] == val, "Sex", label)
    if r:
        results.append(r)

age_buckets = {
    "18-39": X_test["Age"].between(1, 4),
    "40-59": X_test["Age"].between(5, 8),
    "60+": X_test["Age"].between(9, 13),
}
for label, mask in age_buckets.items():
    r = subgroup_metrics(mask, "Age", label)
    if r:
        results.append(r)

income_buckets = {
    "Lower income (1-4)": X_test["Income"].between(1, 4),
    "Higher income (5-8)": X_test["Income"].between(5, 8),
}
for label, mask in income_buckets.items():
    r = subgroup_metrics(mask, "Income", label)
    if r:
        results.append(r)

summary = pd.DataFrame(results)
print(summary.to_string(index=False))
summary.to_csv("figures/subgroup_calibration_full.csv", index=False)
print("\nSaved figures/subgroup_calibration_full.csv")

# ---------------- Reliability diagrams: Age and Income ----------------
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

ax = axes[0]
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
colors_age = {"18-39": "#02C39A", "40-59": "#028090", "60+": "#E86A5C"}
for label, mask in age_buckets.items():
    y_g, p_g = y_test.values[mask.values], proba_cal[mask.values]
    frac_pos, mean_pred = calibration_curve(y_g, p_g, n_bins=10, strategy="quantile")
    ece = expected_calibration_error(y_g, p_g)
    ax.plot(mean_pred, frac_pos, "o-", color=colors_age[label], label=f"{label} (ECE={ece:.3f})")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Calibration by Age Group")
ax.legend()

ax = axes[1]
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
colors_income = {"Lower income (1-4)": "#E86A5C", "Higher income (5-8)": "#028090"}
for label, mask in income_buckets.items():
    y_g, p_g = y_test.values[mask.values], proba_cal[mask.values]
    frac_pos, mean_pred = calibration_curve(y_g, p_g, n_bins=10, strategy="quantile")
    ece = expected_calibration_error(y_g, p_g)
    ax.plot(mean_pred, frac_pos, "o-", color=colors_income[label], label=f"{label} (ECE={ece:.3f})")
ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Calibration by Income Group")
ax.legend()

plt.tight_layout()
plt.savefig("figures/subgroup_calibration_age_income.png", dpi=150)
plt.close()
print("Saved figures/subgroup_calibration_age_income.png")
