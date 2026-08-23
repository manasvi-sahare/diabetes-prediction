import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/diabetes.csv")

X = df.drop(columns=["Diabetes_binary"])
y = df["Diabetes_binary"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

print(f"Train set: {X_train.shape}, positive rate: {y_train.mean():.2%}")
print(f"Test set:  {X_test.shape}, positive rate: {y_test.mean():.2%}")

# Scale features (needed for Logistic Regression later)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nScaled training features — mean~0, std~1 check:")
print(f"  Mean of first 3 features: {X_train_scaled[:, :3].mean(axis=0).round(4)}")
print(f"  Std  of first 3 features: {X_train_scaled[:, :3].std(axis=0).round(4)}")

# Save everything to disk so later scripts can reuse it without re-splitting
joblib.dump((X_train, X_test, y_train, y_test, scaler), "models/split.pkl")
print("\nSaved split + scaler to models/split.pkl")