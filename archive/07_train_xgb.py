import joblib
from xgboost import XGBClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, brier_score_loss,
    confusion_matrix
)

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")

# ratio of negative to positive samples in the training set
neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
scale_pos_weight = neg / pos
print(f"scale_pos_weight = {scale_pos_weight:.3f}")

model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    eval_metric="logloss",
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n=== XGBoost ===")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
print(f"AUC-ROC:     {roc_auc_score(y_test, y_proba):.4f}")
print(f"Brier score: {brier_score_loss(y_test, y_proba):.4f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "models/xgboost.pkl")
print("\nSaved model to models/xgboost.pkl")