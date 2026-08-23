import joblib
import matplotlib.pyplot as plt
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import roc_auc_score, brier_score_loss, classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")
xgb_model = joblib.load("models/xgboost.pkl")

# Hold out a calibration set from the training data (never touch X_test until the end)
X_fit, X_calib, y_fit, y_calib = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
)

# Re-fit a fresh XGBoost on the smaller X_fit so the calibration set is truly unseen
from xgboost import XGBClassifier
neg, pos = (y_fit == 0).sum(), (y_fit == 1).sum()
base_model = XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=neg / pos, eval_metric="logloss",
    n_jobs=-1, random_state=42
)
base_model.fit(X_fit, y_fit)

results = {}
from sklearn.frozen import FrozenEstimator

for method in ["sigmoid", "isotonic"]:
    calibrated = CalibratedClassifierCV(FrozenEstimator(base_model), method=method)
    calibrated.fit(X_calib, y_calib)

    proba = calibrated.predict_proba(X_test)[:, 1]
    pred = calibrated.predict(X_test)

    auc = roc_auc_score(y_test, proba)
    brier = brier_score_loss(y_test, proba)
    results[method] = (calibrated, proba, auc, brier)

    print(f"\n=== XGBoost + {method} calibration ===")
    print(classification_report(y_test, pred, target_names=["No Diabetes", "Diabetes"]))
    print(f"AUC-ROC:     {auc:.4f}")
    print(f"Brier score: {brier:.4f}")

# Uncalibrated baseline for comparison
raw_proba = xgb_model.predict_proba(X_test)[:, 1]
raw_brier = brier_score_loss(y_test, raw_proba)
raw_auc = roc_auc_score(y_test, raw_proba)
print(f"\n=== XGBoost (uncalibrated, original) ===")
print(f"AUC-ROC:     {raw_auc:.4f}")
print(f"Brier score: {raw_brier:.4f}")

# Plot all three calibration curves together
plt.figure(figsize=(7, 6))
frac_pos, mean_pred = calibration_curve(y_test, raw_proba, n_bins=10, strategy="quantile")
plt.plot(mean_pred, frac_pos, marker="o", label=f"Uncalibrated (Brier={raw_brier:.3f})", color="#5B7B80")

for method, (model, proba, auc, brier) in results.items():
    frac_pos, mean_pred = calibration_curve(y_test, proba, n_bins=10, strategy="quantile")
    plt.plot(mean_pred, frac_pos, marker="o", label=f"{method} (Brier={brier:.3f})")

plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
plt.xlabel("Mean predicted probability")
plt.ylabel("Fraction of positives (observed)")
plt.title("Calibration: Before vs. After")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/calibration_comparison.png", dpi=150)
print("\nSaved outputs/calibration_comparison.png")

# Save the better-calibrated model
best_method = min(results, key=lambda m: results[m][3])
joblib.dump(results[best_method][0], "models/xgboost_calibrated.pkl")
print(f"Saved best-calibrated model ({best_method}) to models/xgboost_calibrated.pkl")