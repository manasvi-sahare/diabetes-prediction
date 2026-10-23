import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pipeline import X_tr, X_te, y_tr, y_te, models

df = pd.read_csv("data/diabetes.csv")
feats = [c for c in df.columns if c != "Diabetes_binary"]
n = len(df)

print("Label values:", sorted(df["Diabetes_binary"].unique()))
d_full = df.duplicated().sum()
d_feat = df.duplicated(subset=feats).sum()
print(f"Exact duplicate rows (features + label): {d_full:,} ({d_full/n:.1%})")
print(f"Rows whose feature profile repeats an earlier row: {d_feat:,} ({d_feat/n:.1%})")
g = df.groupby(feats)["Diabetes_binary"].nunique()
print(f"Feature profiles with conflicting labels: {(g > 1).sum():,}")

h = lambda X: pd.util.hash_pandas_object(X[feats], index=False)
in_train = h(X_te).isin(set(h(X_tr))).values
print(f"Test rows with an identical profile in train: {in_train.sum():,} ({in_train.mean():.1%})")

m = models(False)["XGB"]; m.fit(X_tr, y_tr)
p = m.predict_proba(X_te)[:, 1]; yt = np.asarray(y_te)
print(f"XGB AUC, all test rows:        {roc_auc_score(yt, p):.4f}")
print(f"XGB AUC, rows unseen in train: {roc_auc_score(yt[~in_train], p[~in_train]):.4f}")
print(f"XGB AUC, rows seen in train:   {roc_auc_score(yt[in_train], p[in_train]):.4f}")