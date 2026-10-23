import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")

age_buckets = {
    "18-39": X_test["Age"].between(1, 4),
    "40-59": X_test["Age"].between(5, 8),
    "60+": X_test["Age"].between(9, 13),
}

neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
spw = neg / pos

models = {
    "Logistic Regression": (joblib.load("models/logreg.pkl"), True),
    "Random Forest": (joblib.load("models/random_forest.pkl"), False),
    "XGBoost": (joblib.load("models/xgboost.pkl"), False),
    "LightGBM": (joblib.load("models/lgbm.pkl"), False),
}

results = []
for name, (model, needs_scaling) in models.items():
    X_eval = scaler.transform(X_test) if needs_scaling else X_test
    proba = model.predict_proba(X_eval)[:, 1]

    subgroup_aucs = {}
    for label, mask in age_buckets.items():
        y_g, p_g = y_test.values[mask.values], proba[mask.values]
        if len(np.unique(y_g)) < 2:
            continue
        subgroup_aucs[label] = roc_auc_score(y_g, p_g)

    gap = max(subgroup_aucs.values()) - min(subgroup_aucs.values())
    row = {"Model": name, "Subgroup_Gap_AgeAUC": round(gap, 4)}
    row.update({f"AUC_{k}": round(v, 4) for k, v in subgroup_aucs.items()})
    results.append(row)
    print(f"{name}: gap={gap:.4f}  " + "  ".join(f"{k}={v:.4f}" for k, v in subgroup_aucs.items()))

summary = pd.DataFrame(results)
summary.to_csv("figures/subgroup_gap_all_models.csv", index=False)
print("\nSaved figures/subgroup_gap_all_models.csv")