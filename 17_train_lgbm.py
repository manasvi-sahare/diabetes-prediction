import joblib
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, brier_score_loss,
    confusion_matrix
)

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")

# ratio of negative to positive samples in the training set (same as XGBoost script)
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.3f}")

model = LGBMClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    n_jobs=-1,
    random_state=42,
    verbose=-1  # suppress LightGBM's default training logs
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== LightGBM ===")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
print(f"AUC-ROC:     {roc_auc_score(y_test, y_proba):.4f}")
print(f"Brier score: {brier_score_loss(y_test, y_proba):.4f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "models/lgbm.pkl")
print("\nSaved model to models/lgbm.pkl")

import os
size_mb = os.path.getsize("models/lgbm.pkl") / (1024 * 1024)
print(f"Model file size: {size_mb:.1f} MB")