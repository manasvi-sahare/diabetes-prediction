import pandas as pd

df = pd.read_csv("data/diabetes.csv")

print("Feature              | # unique | min   | max   | likely type")
print("-" * 75)
for col in df.columns:
    nunique = df[col].nunique()
    vmin, vmax = df[col].min(), df[col].max()
    if nunique == 2:
        ftype = "binary flag"
    elif nunique <= 6:
        ftype = "ordinal (small scale)"
    elif nunique <= 14:
        ftype = "ordinal (larger scale)"
    else:
        ftype = "continuous"
    print(f"{col:<22}{nunique:<10}{vmin:<7}{vmax:<7}{ftype}")