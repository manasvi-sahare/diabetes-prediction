import glob, os, shutil
import kagglehub

path = kagglehub.dataset_download("alexteboul/diabetes-health-indicators-dataset")
print("Downloaded to:", path)
for f in sorted(os.listdir(path)):
    print("  ", f)

match = glob.glob(os.path.join(path, "*012*.csv"))   # the 3-class file only, not the 50/50 split
assert len(match) == 1, f"Expected one 012 file, found: {match}"
shutil.copy(match[0], "data/diabetes_012.csv")
print("Saved data/diabetes_012.csv")