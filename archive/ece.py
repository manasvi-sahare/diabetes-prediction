import os
import matplotlib.pyplot as plt

os.makedirs("figures", exist_ok=True)

groups = ["Age 40-59", "Age 60+", "Income: Higher", "Income: Lower"]
ece = [0.0054, 0.0055, 0.0033, 0.0160]
colors = ["#4C72B0", "#4C72B0", "#55A868", "#C44E52"]

fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(groups, ece, color=colors)
for b, v in zip(bars, ece):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.0003, f"{v:.4f}",
            ha="center", fontsize=9)
ax.set_ylabel("Expected Calibration Error")
ax.set_title("ECE by Subgroup")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("figures/fig7_ece_gap.png", dpi=300, bbox_inches="tight")
plt.close()
print("saved")