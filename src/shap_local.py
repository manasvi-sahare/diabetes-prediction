import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")
xgb_model = joblib.load("models/xgboost.pkl")

explainer = shap.TreeExplainer(xgb_model)

# Pick one real patient from the test set who was actually diabetic
diabetic_idx = y_test[y_test == 1].index[0]
patient = X_test.loc[[diabetic_idx]]

shap_val = explainer(patient)
proba = xgb_model.predict_proba(patient)[0, 1]

print(f"Patient index: {diabetic_idx}")
print(f"Actual label: {'Diabetic' if y_test.loc[diabetic_idx] == 1 else 'Non-diabetic'}")
print(f"Predicted probability: {proba:.3f}")
print("\nFeature values for this patient:")
print(patient.T)

plt.figure(figsize=(9, 5))
shap.plots.waterfall(shap_val[0], max_display=10, show=False)
plt.tight_layout()
plt.savefig("figures/shap_waterfall_example.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved figures/shap_waterfall_example.png")
