import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, brier_score_loss,
    confusion_matrix
)

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("=== Logistic Regression (baseline) ===")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
print(f"AUC-ROC:     {roc_auc_score(y_test, y_proba):.4f}")
print(f"Brier score: {brier_score_loss(y_test, y_proba):.4f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "models/logreg.pkl")
print("\nSaved model to models/logreg.pkl")