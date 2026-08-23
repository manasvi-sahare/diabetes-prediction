import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, roc_auc_score, brier_score_loss,
    confusion_matrix
)

X_train, X_test, y_train, y_test, scaler = joblib.load("models/split.pkl")

model = RandomForestClassifier(
    n_estimators=150,        # fewer trees
    max_depth=12,             # shallower
    min_samples_leaf=20,      # larger minimum leaf size = fewer, coarser splits
    max_features="sqrt",      # standard RF setting, reduces per-split cost too
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("=== Random Forest (n_estimators=150, max_depth=12, min_samples_leaf=20) ===")
print(classification_report(y_test, y_pred, target_names=["No Diabetes", "Diabetes"]))
print(f"AUC-ROC:     {roc_auc_score(y_test, y_proba):.4f}")
print(f"Brier score: {brier_score_loss(y_test, y_proba):.4f}")
print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))

joblib.dump(model, "models/random_forest.pkl", compress=3)  # add compression too
print("\nSaved model to models/random_forest.pkl")

import os
size_mb = os.path.getsize("models/random_forest.pkl") / (1024 * 1024)
print(f"\nModel file size: {size_mb:.1f} MB")