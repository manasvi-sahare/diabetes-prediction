import pandas as pd

df = pd.read_csv("data/diabetes.csv")

print("=" * 60)
print("SHAPE AND STRUCTURE")
print("=" * 60)
print(f"Rows: {df.shape[0]:,}")
print(f"Columns: {df.shape[1]}")

print("\n" + "=" * 60)
print("MISSING VALUES")
print("=" * 60)
missing = df.isnull().sum()
print(f"Total missing values: {missing.sum()}")

print("\n" + "=" * 60)
print("DATA TYPES")
print("=" * 60)
print(df.dtypes.value_counts())

print("\n" + "=" * 60)
print("TARGET VARIABLE: Diabetes_binary")
print("=" * 60)
counts = df['Diabetes_binary'].value_counts()
print(counts)
print(f"\nClass imbalance ratio: {counts[0]/counts[1]:.2f} : 1")
print(f"Positive (diabetic) rate: {df['Diabetes_binary'].mean():.2%}")