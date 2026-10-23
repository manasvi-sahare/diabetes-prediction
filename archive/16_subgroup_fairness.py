import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score, f1_score
from sklearn.calibration import calibration_curve

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")
xgb_raw = joblib.load("models/xgboost.pkl")
xgb_cal = joblib.load("models/xgboost_calibrated.pkl")

proba_raw = xgb_raw.predict_proba(X_test)[:, 1]
proba_cal = xgb_cal.predict_proba(X_test)[:, 1]

TUNED_THRESHOLD = 0.245  # from 10_threshold_tuning.py

def subgroup_metrics(mask, group_name, group_label):
    y_g = y_test[mask]
    p_raw_g = proba_raw[mask.values]
    p_cal_g = proba_cal[mask.values]
    pred_g = (p_cal_g >= TUNED_THRESHOLD).astype(int)

    if len(np.unique(y_g)) < 2:
        return None  # can't compute AUC with only one class present

    return {
        "Group": group_name,
        "Subgroup": group_label,
        "N": int(mask.sum()),
        "Positive_rate": round(y_g.mean(), 4),
        "ROC_AUC": round(roc_auc_score(y_g, p_cal_g), 4),
        "Brier_raw": round(brier_score_loss(y_g, p_raw_g), 4),
        "Brier_calibrated": round(brier_score_loss(y_g, p_cal_g), 4),
        "Precision_tuned_thresh": round(precision_score(y_g, pred_g, zero_division=0), 4),
        "Recall_tuned_thresh": round(recall_score(y_g, pred_g, zero_division=0), 4),
        "F1_tuned_thresh": round(f1_score(y_g, pred_g, zero_division=0), 4),
    }

results = []

# ---------------- Sex ----------------
# BRFSS coding: 0 = Female, 1 = Male
for val, label in [(0, "Female"), (1, "Male")]:
    r = subgroup_metrics(X_test["Sex"] == val, "Sex", label)
    if r:
        results.append(r)

# ---------------- Age (bucketed from 13-level BRFSS scale) ----------------
# 1-4 = 18-39, 5-8 = 40-59, 9-13 = 60+
age_buckets = {
    "18-39": X_test["Age"].between(1, 4),
    "40-59": X_test["Age"].between(5, 8),
    "60+": X_test["Age"].between(9, 13),
}
for label, mask in age_buckets.items():
    r = subgroup_metrics(mask, "Age", label)
    if r:
        results.append(r)

# ---------------- Income (bucketed from 8-level BRFSS scale) ----------------
# 1-4 = lower income, 5-8 = higher income
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
summary.to_csv("figures/subgroup_fairness_summary.csv", index=False)
print("\nSaved figures/subgroup_fairness_summary.csv")

# ---------------- Calibration curve overlay: Male vs. Female ----------------
fig, ax = plt.subplots(figsize=(7, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")

for val, label, color in [(0, "Female", "#E86A5C"), (1, "Male", "#028090")]:
    mask = (X_test["Sex"] == val).values
    y_g, p_g = y_test.values[mask], proba_cal[mask]
    frac_pos, mean_pred = calibration_curve(y_g, p_g, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y_g, p_g)
    ax.plot(mean_pred, frac_pos, "o-", color=color, label=f"{label} (Brier={brier:.4f}, n={mask.sum()})")

ax.set_xlabel("Mean predicted probability")
ax.set_ylabel("Observed frequency")
ax.set_title("Calibration by Sex (XGBoost, Platt/isotonic-calibrated)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/subgroup_calibration_by_sex.png", dpi=150)
plt.close()
print("Saved figures/subgroup_calibration_by_sex.png")