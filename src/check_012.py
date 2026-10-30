import pandas as pd

b = pd.read_csv("data/diabetes.csv")
t = pd.read_csv("data/diabetes_012.csv")
print("rows:", len(b), len(t))
print("label counts (3-class):\n", t.iloc[:, 0].value_counts().sort_index())

feats = [c for c in b.columns if c != "Diabetes_binary"]
same_cols = all(c in t.columns for c in feats)
print("same feature columns:", same_cols)
if same_cols and len(b) == len(t):
    print("features identical row by row:", (b[feats].values == t[feats].values).all())
    lab = t.columns[0]
    print("binary label == (3-class > 0):", ((t[lab] > 0).astype(float).values == b["Diabetes_binary"].values).all())