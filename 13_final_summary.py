import joblib
import pandas as pd
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score,
    average_precision_score, brier_score_loss
)

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")

results = []
model_files = {
    "Logistic Regression": ("models/logreg.pkl", True, 0.5),
    "Random Forest": ("models/random_forest.pkl", False, 0.5),
    "XGBoost": ("models/xgboost.pkl", False, 0.5),
    "XGBoost (calibrated, thresh=0.5)": ("models/xgboost_calibrated.pkl", False, 0.5),
    "XGBoost (calibrated, tuned thresh=0.245)": ("models/xgboost_calibrated.pkl", False, 0.245),
    "LightGBM": ("models/lgbm.pkl", False, 0.5),
}

for name, (path, needs_scaling, threshold) in model_files.items():
    print(f"Loading {name}...")
    model = joblib.load(path)
    X_eval = scaler.transform(X_test) if needs_scaling else X_test
    proba = model.predict_proba(X_eval)[:, 1]
    pred = (proba >= threshold).astype(int)

    results.append({
        "Model": name,
        "Threshold": threshold,
        "Precision": round(precision_score(y_test, pred), 4),
        "Recall": round(recall_score(y_test, pred), 4),
        "F1": round(f1_score(y_test, pred), 4),
        "ROC_AUC": round(roc_auc_score(y_test, proba), 4),
        "PR_AUC": round(average_precision_score(y_test, proba), 4),
        "Brier_Score": round(brier_score_loss(y_test, proba), 4),
    })

summary = pd.DataFrame(results)
print("\n" + summary.to_string(index=False))
summary.to_csv("figures/final_results_summary.csv", index=False)
print("\nSaved figures/final_results_summary.csv")