import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import precision_score, recall_score, f1_score

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")
calibrated = joblib.load("models/xgboost_calibrated.pkl")

proba = calibrated.predict_proba(X_test)[:, 1]
y_true = y_test.values

thresholds = np.arange(0.10, 0.91, 0.05)
results = []

print("=== Threshold Sensitivity Sweep ===\n")
print(f"{'Threshold':<12}{'Precision':<12}{'Recall':<12}{'F1':<12}{'PosPredRate':<12}")

for t in thresholds:
    pred = (proba >= t).astype(int)
    prec = precision_score(y_true, pred, zero_division=0)
    rec = recall_score(y_true, pred, zero_division=0)
    f1 = f1_score(y_true, pred, zero_division=0)
    pos_rate = pred.mean()

    print(f"{t:<12.2f}{prec:<12.4f}{rec:<12.4f}{f1:<12.4f}{pos_rate:<12.4f}")
    results.append({"Threshold": round(t, 2), "Precision": round(prec, 4),
                     "Recall": round(rec, 4), "F1": round(f1, 4),
                     "Positive_Pred_Rate": round(pos_rate, 4)})

summary = pd.DataFrame(results)
summary.to_csv("figures/threshold_sensitivity.csv", index=False)

# ---------------- Identify the F1-optimal threshold and plateau width ----------------
best_idx = summary["F1"].idxmax()
best_t = summary.loc[best_idx, "Threshold"]
best_f1 = summary.loc[best_idx, "F1"]

# "Near-optimal" = within 2% of best F1 (relative), to characterize robustness
near_optimal = summary[summary["F1"] >= best_f1 * 0.98]
plateau_low, plateau_high = near_optimal["Threshold"].min(), near_optimal["Threshold"].max()

print(f"\nF1-optimal threshold (from this coarser 0.05-step sweep): {best_t:.2f} (F1={best_f1:.4f})")
print(f"Near-optimal plateau (F1 within 2% of best): thresholds {plateau_low:.2f} to {plateau_high:.2f}")
print(f"Plateau width: {plateau_high - plateau_low:.2f}")

# ---------------- Plot ----------------
fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(summary["Threshold"], summary["Precision"], "o-", label="Precision", color="#028090")
ax.plot(summary["Threshold"], summary["Recall"], "o-", label="Recall", color="#E86A5C")
ax.plot(summary["Threshold"], summary["F1"], "o-", label="F1", color="#02C39A", linewidth=2.5)
ax.plot(summary["Threshold"], summary["Positive_Pred_Rate"], "o--", label="Positive Prediction Rate",
        color="#7A7A7A", alpha=0.7)
ax.axvline(0.245, color="black", linestyle=":", alpha=0.6, label="Previously-tuned threshold (0.245)")
ax.axvspan(plateau_low, plateau_high, alpha=0.1, color="#02C39A", label="Near-optimal F1 plateau")
ax.set_xlabel("Decision Threshold")
ax.set_ylabel("Score")
ax.set_title("Threshold Sensitivity: Precision, Recall, F1, and Positive Prediction Rate")
ax.legend(loc="upper right", fontsize=9)
ax.set_xlim(0.08, 0.92)
plt.tight_layout()
plt.savefig("figures/threshold_sensitivity.png", dpi=150)
plt.close()

print("\nSaved figures/threshold_sensitivity.csv, threshold_sensitivity.png")