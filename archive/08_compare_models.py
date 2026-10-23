import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": (joblib.load("models/logreg.pkl"), X_test_scaled),
    "Random Forest":       (joblib.load("models/random_forest.pkl"), X_test),
    "XGBoost":             (joblib.load("models/xgboost.pkl"), X_test),
}

colors = {"Logistic Regression": "#5B7B80", "Random Forest": "#00A896", "XGBoost": "#028090"}

fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

# --- ROC curves ---
for name, (model, X) in models.items():
    proba = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})", color=colors[name], linewidth=2)

axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
axes[0].set_xlabel("False Positive Rate")
axes[0].set_ylabel("True Positive Rate")
axes[0].set_title("ROC Curves")
axes[0].legend(loc="lower right", fontsize=9)

# --- Calibration curves ---
for name, (model, X) in models.items():
    proba = model.predict_proba(X)[:, 1]
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    axes[1].plot(mean_pred, frac_pos, marker="o", label=name, color=colors[name], linewidth=2)

axes[1].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
axes[1].set_xlabel("Mean predicted probability")
axes[1].set_ylabel("Fraction of positives (observed)")
axes[1].set_title("Calibration Curves")
axes[1].legend(loc="upper left", fontsize=9)

plt.tight_layout()
plt.savefig("outputs/model_comparison.png", dpi=150)
print("Saved plot to outputs/model_comparison.png")
plt.show()