import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, f1_score

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")
calibrated = joblib.load("models/xgboost_calibrated.pkl")

proba = calibrated.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, proba)

# F1 at each threshold (precision/recall arrays are 1 longer than thresholds)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-9)
best_idx = np.argmax(f1_scores)
best_threshold = thresholds[best_idx]

print(f"Best threshold by F1: {best_threshold:.3f}")
print(f"  Precision: {precision[best_idx]:.3f}")
print(f"  Recall:    {recall[best_idx]:.3f}")
print(f"  F1:        {f1_scores[best_idx]:.3f}")

# Also show a "high recall" operating point, e.g. threshold that gives ~75% recall
target_recall = 0.75
idx_recall = np.argmin(np.abs(recall[:-1] - target_recall))
print(f"\nThreshold for ~{target_recall:.0%} recall: {thresholds[idx_recall]:.3f}")
print(f"  Precision at that threshold: {precision[idx_recall]:.3f}")

plt.figure(figsize=(7, 5))
plt.plot(thresholds, precision[:-1], label="Precision")
plt.plot(thresholds, recall[:-1], label="Recall")
plt.plot(thresholds, f1_scores, label="F1", linestyle="--")
plt.axvline(best_threshold, color="gray", linestyle=":", label=f"Best F1 threshold ({best_threshold:.2f})")
plt.xlabel("Decision threshold")
plt.ylabel("Score")
plt.title("Precision / Recall / F1 vs. Threshold (calibrated model)")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/threshold_tuning.png", dpi=150)
print("\nSaved outputs/threshold_tuning.png")