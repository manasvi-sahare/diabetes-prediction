import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")
xgb_model = joblib.load("models/xgboost.pkl")  # the uncalibrated model — SHAP needs the raw tree structure

print("Computing SHAP values (TreeExplainer, sample of 3000 test cases)...")
explainer = shap.TreeExplainer(xgb_model)

# Sample for speed — SHAP on the full 50,736-row test set would be slow
sample_idx = np.random.RandomState(42).choice(X_test.index, size=3000, replace=False)
X_sample = X_test.loc[sample_idx]

shap_values = explainer(X_sample)

# Global importance: mean absolute SHAP value per feature
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
shap_importance = pd.DataFrame({
    "feature": X_test.columns,
    "mean_abs_shap": mean_abs_shap
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print("\nTop 10 features by SHAP importance:")
print(shap_importance.head(10).to_string(index=False))
shap_importance.to_csv("figures/shap_importance.csv", index=False)

# Figure 1: SHAP summary (beeswarm) — shows direction + magnitude per feature
plt.figure(figsize=(8, 6))
shap.summary_plot(shap_values, X_sample, show=False, max_display=12)
plt.title("SHAP Summary — Feature Impact on Diabetes Risk")
plt.tight_layout()
plt.savefig("figures/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved figures/shap_summary.png")

# Figure 2: SHAP importance bar chart
fig, ax = plt.subplots(figsize=(8, 6))
top = shap_importance.head(10).iloc[::-1]
ax.barh(top["feature"], top["mean_abs_shap"], color="#028090")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top 10 Features by SHAP Importance")
plt.tight_layout()
plt.savefig("figures/shap_importance_bar.png", dpi=150)
plt.close()
print("Saved figures/shap_importance_bar.png")