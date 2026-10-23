import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from pipeline import (X_tr, X_va, X_te, y_tr, y_va, y_te, models,
                      fit_calibrator, cv_brier, best_threshold)

yva, yte = np.asarray(y_va), np.asarray(y_te)

# ---------- fit plain and weighted XGB; calibrate each on validation ----------
S = {}
for lab, w in (("plain", False), ("weighted", True)):
    m = models(w)["XGB"]; m.fit(X_tr, y_tr)
    s_va, s_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
    kind = min(("platt", "isotonic"), key=lambda c: cv_brier(c, s_va, yva))
    cal = fit_calibrator(kind, s_va, yva)
    S[lab] = dict(raw_va=s_va, raw_te=s_te, cal_va=cal(s_va), cal_te=cal(s_te), kind=kind)
    print(lab, "calibrator:", kind)

def curve(y, p, bins=10):
    parts = np.array_split(np.argsort(p), bins)     # equal-mass bins, same as the ECE
    return np.array([p[i].mean() for i in parts]), np.array([y[i].mean() for i in parts])

# ---------- Fig A: calibration, plain vs weighted, before and after ----------
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
for lab, key, name, c in (("plain", "raw_te", "Unweighted, raw", "#028090"),
                          ("weighted", "raw_te", "Class-weighted, raw", "#E86A5C"),
                          ("weighted", "cal_te", "Class-weighted, recalibrated", "#6A4C93")):
    x, yy = curve(yte, S[lab][key])
    ax.plot(x, yy, "o-", color=c, label=name)
ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
ax.set_title("XGBoost calibration (test set)"); ax.legend()
plt.tight_layout(); plt.savefig("figures/v2_fig_calibration.png", dpi=150); plt.close()

# ---------- Fig B: threshold sweep, unweighted + calibrated XGB ----------
p_va, p_te = S["plain"]["cal_va"], S["plain"]["cal_te"]
thr_star = best_threshold(yva, p_va)                # chosen on validation only
grid = np.arange(0.05, 0.60, 0.01)
sweep = pd.DataFrame([dict(thr=t, prec=precision_score(yte, p_te >= t, zero_division=0),
                           rec=recall_score(yte, p_te >= t), F1=f1_score(yte, p_te >= t),
                           pos_rate=(p_te >= t).mean()) for t in grid]).round(4)
sweep.to_csv("figures/v2_threshold_sweep.csv", index=False)
keep = sorted(set([0.10, 0.20, 0.30, 0.50, round(float(thr_star), 3)]))
rows = [dict(thr=t, prec=precision_score(yte, p_te >= t, zero_division=0), rec=recall_score(yte, p_te >= t),
             F1=f1_score(yte, p_te >= t), pos_rate=(p_te >= t).mean()) for t in keep]
print("\nValidation-chosen threshold:", round(float(thr_star), 3))
print(pd.DataFrame(rows).round(4).to_string(index=False))

fig, ax = plt.subplots(figsize=(7.5, 5))
for col, c in (("prec", "#028090"), ("rec", "#E86A5C"), ("F1", "#2A9D8F"), ("pos_rate", "#888888")):
    ax.plot(sweep["thr"], sweep[col], color=c, label=col)
ax.axvline(thr_star, ls=":", color="k", label=f"validation-chosen = {thr_star:.3f}")
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Score"); ax.legend()
ax.set_title("Threshold sensitivity (calibrated unweighted XGBoost, test set)")
plt.tight_layout(); plt.savefig("figures/v2_fig_threshold.png", dpi=150); plt.close()

# ---------- Fig C: subgroup AUC and ECE with bootstrap CIs (from subgroups.py output) ----------
tab = pd.read_csv("figures/v2_subgroup_metrics.csv")
y = np.arange(len(tab))
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
for ax, col in zip(axes, ("AUC", "ECE")):
    lo = np.clip(tab[col] - tab[col + "_lo"], 0, None)    # bootstrap ECE can sit below its own CI
    hi = np.clip(tab[col + "_hi"] - tab[col], 0, None)
    ax.errorbar(tab[col], y, xerr=[lo, hi], fmt="o", capsize=4, color="#028090")
    ax.set_title(f"{col} by subgroup (95% bootstrap CI)"); ax.set_xlabel(col)
axes[0].set_yticks(y); axes[0].set_yticklabels(tab["group"]); axes[0].invert_yaxis()
plt.tight_layout(); plt.savefig("figures/v2_fig_subgroups.png", dpi=150); plt.close()

# ---------- Fig D: recall and precision at the common threshold, by subgroup ----------
fig, ax = plt.subplots(figsize=(9, 4.5))
w = 0.38
ax.bar(y - w / 2, tab["recall"], w, label="Recall", color="#E86A5C")
ax.bar(y + w / 2, tab["precision"], w, label="Precision", color="#028090")
ax.set_xticks(y); ax.set_xticklabels(tab["group"], rotation=25, ha="right"); ax.legend()
ax.set_title("Recall and precision at one validation-chosen threshold")
plt.tight_layout(); plt.savefig("figures/v2_fig_operating_point.png", dpi=150); plt.close()

# ---------- Fig E: age-stratified AUC, all models (from subgroups.py output) ----------
ag = pd.read_csv("figures/v2_age_gap_all_models.csv")
cols = ["Age: 18-39", "Age: 40-59", "Age: 60+"]
ax = ag.set_index("model")[cols].plot.bar(figsize=(7.5, 5), ylim=(0.70, 0.86), rot=0)
ax.set_ylabel("ROC-AUC"); ax.set_title("Age-stratified ROC-AUC, all models")
plt.tight_layout(); plt.savefig("figures/v2_fig_age_gap.png", dpi=150); plt.close()

print("\nSaved figures/v2_fig_*.png and v2_threshold_sweep.csv")