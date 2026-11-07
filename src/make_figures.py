import os
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score
from pipeline import X_tr, X_va, X_te, y_tr, y_va, y_te, models, fit_calibrator, cv_brier

os.makedirs("figures", exist_ok=True)
yva, yte = np.asarray(y_va), np.asarray(y_te)
C1, C2, C3, C4 = "#028090", "#E86A5C", "#6A4C93", "#2A9D8F"
FIGSIZE = (7, 4.2)


def err(center, lo, hi):
    """Asymmetric error bars, clipped so they are never negative."""
    c = np.asarray(center, float)
    return [np.clip(c - np.asarray(lo, float), 0, None),
            np.clip(np.asarray(hi, float) - c, 0, None)]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"figures/{name}", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 0. Fit models + calibrators FIRST (this was defined too late before -> NameError)
# ---------------------------------------------------------------------------
S = {}
for lab, w in (("plain", False), ("weighted", True)):
    m = models(w)["XGB"]
    m.fit(X_tr, y_tr)
    s_va, s_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
    kind = min(("platt", "isotonic"), key=lambda c: cv_brier(c, s_va, yva))
    S[lab] = dict(raw=s_te, cal=fit_calibrator(kind, s_va, yva)(s_te), kind=kind)


def curve(y_, p, bins=10):
    parts = np.array_split(np.argsort(p), bins)
    return np.array([p[i].mean() for i in parts]), np.array([y_[i].mean() for i in parts])


# ---------------------------------------------------------------------------
# 1. AUC with bootstrap CIs (y from 0, values printed)
# ---------------------------------------------------------------------------
ci = pd.read_csv("figures/v2_bootstrap_ci_summary.csv")
fig, ax = plt.subplots(figsize=FIGSIZE)
xs = np.arange(len(ci))
ax.errorbar(xs, ci["AUC"], yerr=err(ci["AUC"], ci["AUC_lo"], ci["AUC_hi"]),
            fmt="o", capsize=5, color=C1)
for x_, a, lo, hi in zip(xs, ci["AUC"], ci["AUC_lo"], ci["AUC_hi"]):
    # put label just above the CI, but never above the top of the axes
    ax.text(x_ + 0.05, min(hi + 0.02, 0.90), f"{a:.3f}\n[{lo:.3f}, {hi:.3f}]",
            ha="left", va="bottom", fontsize=7)
ax.axhline(0.5, color="k", lw=0.8, ls="--")
ax.text(len(ci) - 0.55, 0.52, "chance (0.5)", ha="right", fontsize=7)
ax.set_xticks(xs); ax.set_xticklabels(ci["model"])
ax.set_xlim(-0.5, len(ci) - 0.5); ax.set_ylim(0, 1)
ax.set_ylabel("ROC-AUC"); ax.set_title("Test ROC-AUC, 95% bootstrap CI")
save(fig, "v2_bootstrap_ci_plot.png")

# ---------------------------------------------------------------------------
# 2. AUC by single age category (both axes from 0)
# ---------------------------------------------------------------------------
ac = pd.read_csv("figures/v2_age_by_category.csv")
ac["age_category"] = pd.to_numeric(ac["age_category"], errors="coerce")  # must be numeric for xlim
ac = ac.sort_values("age_category")
fig, ax = plt.subplots(figsize=FIGSIZE)
ax.errorbar(ac["age_category"], ac["AUC"], yerr=err(ac["AUC"], ac["lo"], ac["hi"]),
            fmt="o-", capsize=3, color=C1)
ax.axhline(0.5, color="k", lw=0.8, ls="--")
ax.set_xlim(0, 14); ax.set_ylim(0, 1)
ax.set_xticks(range(0, 15, 1))
ax.set_xlabel("BRFSS age category (1 = 18-24, 13 = 80 and over)")
ax.set_ylabel("ROC-AUC")
save(fig, "v2_fig_age_categories.png")

# ---------------------------------------------------------------------------
# 3. Threshold sweep from 0 (everyone flagged), both axes from 0
# ---------------------------------------------------------------------------
p_cal = S["plain"]["cal"]  # calibrated, unweighted XGBoost, test set
grid = np.arange(0, 0.6001, 0.005)
rows = []
for t in grid:
    f = p_cal >= t
    rows.append(dict(thr=t,
                     prec=precision_score(yte, f, zero_division=0),
                     rec=recall_score(yte, f, zero_division=0),
                     F1=f1_score(yte, f, zero_division=0),
                     pos_rate=f.mean()))
sw = pd.DataFrame(rows)
sw.to_csv("figures/v2_threshold_sweep.csv", index=False)

fig, ax = plt.subplots(figsize=FIGSIZE)
for col, c, lab in (("prec", C1, "Precision"), ("rec", C2, "Recall"),
                    ("F1", C4, "F1"), ("pos_rate", "#888888", "Share flagged")):
    ax.plot(sw["thr"], sw[col], color=c, label=lab)
ax.axvline(0.235, ls=":", color="k", label="validation-chosen = 0.235")
ax.set_xlim(0, 0.6); ax.set_ylim(0, 1)
ax.set_xlabel("Decision threshold"); ax.set_ylabel("Metric value")
ax.legend(fontsize=8)
save(fig, "v2_fig_threshold.png")

# ---------------------------------------------------------------------------
# 4. Subgroups: both panels start at 0, ECE limit set from the data
# ---------------------------------------------------------------------------
tab = pd.read_csv("figures/v2_subgroup_metrics.csv")
y = np.arange(len(tab))
fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), sharey=True)
for ax, col in zip(axes, ("AUC", "ECE")):
    ax.errorbar(tab[col], y, xerr=err(tab[col], tab[col + "_lo"], tab[col + "_hi"]),
                fmt="o", capsize=4, color=C1)
    ax.set_title(f"{col} by subgroup (95% CI)"); ax.set_xlabel(col)
axes[0].axvline(0.5, color="k", lw=0.8, ls="--")
axes[0].set_xlim(0, 1)
axes[1].set_xlim(0, max(tab["ECE_hi"].max(), tab["ECE"].max()) * 1.15)  # never clips the CI
axes[1].xaxis.set_major_locator(plt.MaxNLocator(4))   # few ticks so labels do not overlap
axes[0].set_yticks(y); axes[0].set_yticklabels(tab["group"]); axes[0].invert_yaxis()
save(fig, "v2_fig_subgroups.png")

# ---------------------------------------------------------------------------
# 5. Calibration: full 0-1 view + zoomed view (rare outcome bunches near 0)
# ---------------------------------------------------------------------------
series = (("plain", "raw", "Unweighted, raw", C1),
          ("weighted", "raw", "Class-weighted, raw", C2),
          ("weighted", "cal", "Class-weighted, recalibrated", C3))
curves = [(name, c, *curve(yte, S[lab][key])) for lab, key, name, c in series]
zoom = min(1.0, max(max(x.max(), yy.max()) for _, _, x, yy in curves) * 1.1)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, lim, title in ((axes[0], 1.0, "Full range"), (axes[1], zoom, f"Zoom (0 to {zoom:.2f})")):
    ax.plot([0, lim], [0, lim], "k--", label="Perfect calibration")
    for name, c, x_, y_ in curves:
        ax.plot(x_, y_, "o-", color=c, label=name)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim); ax.set_aspect("equal")
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Observed frequency")
    ax.set_title(title)
axes[0].legend(fontsize=8, loc="upper left")
save(fig, "v2_fig_calibration.png")

print("figures saved")

# 6. Subgroup calibration curves (calibrated unweighted XGBoost, test set), axes 0-1
p_cal = S["plain"]["cal"]
GR = {"Age": {"18-39": X_te["Age"].between(1, 4).values, "40-59": X_te["Age"].between(5, 8).values,
               "60+": X_te["Age"].between(9, 13).values},
      "Income": {"Lower (1-4)": X_te["Income"].between(1, 4).values,
                 "Higher (5-8)": X_te["Income"].between(5, 8).values},
      "Sex": {"Female": (X_te["Sex"] == 0).values, "Male": (X_te["Sex"] == 1).values}}
fig, axes = plt.subplots(1, 3, figsize=(11, 3.9))
for ax, (gname, grp) in zip(axes, GR.items()):
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    for c, (lab, m) in zip((C2, C1, C3), grp.items()):
        x_, y_ = curve(yte[m], p_cal[m], bins=10)
        ax.plot(x_, y_, "o-", color=c, ms=3, label=lab)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal")
    ax.set_title(gname); ax.set_xlabel("Mean predicted probability"); ax.legend(fontsize=7)
axes[0].set_ylabel("Observed frequency")
plt.tight_layout(); plt.savefig("figures/v2_fig_subgroup_calibration.png", dpi=200); plt.close()