import os
import matplotlib.pyplot as plt
import numpy as np

os.makedirs("figures", exist_ok=True)

models = ["Logistic Regression", "Random Forest", "XGBoost", "LightGBM"]
a1839 = [0.8368, 0.8435, 0.8377, 0.8454]
a4059 = [0.8325, 0.8324, 0.8318, 0.8330]
a60   = [0.7632, 0.7707, 0.7709, 0.7729]

x = np.arange(len(models))
w = 0.25
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(x - w, a1839, w, label="18-39")
ax.bar(x, a4059, w, label="40-59")
ax.bar(x + w, a60, w, label="60+")
ax.set_xticks(x)
ax.set_xticklabels(models, rotation=15)
ax.set_ylim(0.70, 0.87)
ax.set_ylabel("ROC-AUC")
ax.set_title("Age Stratified ROC-AUC by Model")
ax.legend()
plt.tight_layout()
plt.savefig("figures/fig8_age_gap_all_models.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved")