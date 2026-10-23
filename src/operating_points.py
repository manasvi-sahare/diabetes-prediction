import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from pipeline import (X_tr, X_va, X_te, y_tr, y_va, y_te, models, SEED,
                      fit_calibrator, cv_brier, best_threshold)

WEIGHTED = False
N_BOOT = 300
rng = np.random.default_rng(SEED)
yva, yte = np.asarray(y_va), np.asarray(y_te)

# ---------- calibrated unweighted XGBoost; everything chosen on validation ----------
m = models(WEIGHTED)["XGB"]; m.fit(X_tr, y_tr)
s_va, s_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
kind = min(("platt", "isotonic"), key=lambda c: cv_brier(c, s_va, yva))
cal = fit_calibrator(kind, s_va, yva)
p_va, p_te = cal(s_va), cal(s_te)
thr_star = float(best_threshold(yva, p_va))
rec_star = float((p_va[yva == 1] >= thr_star).mean())      # validation recall of the global rule
print(f"calibrator={kind}  global threshold={thr_star:.3f}  validation recall={rec_star:.3f}")

def parts(X):
    return {
        "Age": {"18-39": X["Age"].between(1, 4).values, "40-59": X["Age"].between(5, 8).values,
                "60+": X["Age"].between(9, 13).values},
        "Income": {"Lower (1-4)": X["Income"].between(1, 4).values,
                   "Higher (5-8)": X["Income"].between(5, 8).values},
        "Sex": {"Female": (X["Sex"] == 0).values, "Male": (X["Sex"] == 1).values},
    }
PV, PT = parts(X_va), parts(X_te)

# ---------- helpers ----------
def rates(y, flag):
    tp = (flag & (y == 1)).sum(); fp = (flag & (y == 0)).sum(); fn = (~flag & (y == 1)).sum()
    return dict(recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1), flag_rate=flag.mean(), tp=tp, fp=fp)

def net_benefit(y, flag, pt):
    n = len(y); r = rates(y, flag)
    return r["tp"] / n - r["fp"] / n * pt / (1 - pt)

def thresholds(strategy, grouping, target=None):
    out = {}
    for g, mv in PV[grouping].items():
        if strategy == "global":
            out[g] = thr_star
        elif strategy == "group_F1":
            out[g] = float(best_threshold(yva[mv], p_va[mv]))
        else:  # equal_recall
            out[g] = float(np.quantile(p_va[mv & (yva == 1)], 1 - target))
    return out

def apply(grouping, thr):
    flag = np.zeros(len(yte), bool)
    for g, mt in PT[grouping].items():
        flag[mt] = p_te[mt] >= thr[g]
    return flag

def recall_gap(grouping, flag, idx=None):
    rec = []
    for g, mt in PT[grouping].items():
        mm = mt if idx is None else mt[idx]
        yy = yte if idx is None else yte[idx]
        ff = flag if idx is None else flag[idx]
        rec.append(rates(yy[mm], ff[mm])["recall"])
    return max(rec) - min(rec)

# ---------- 1. Three threshold strategies ----------
STRATS = ("global", "group_F1", "equal_recall")
detail, summary, flags = [], [], {}
for grouping in PT:
    for st in STRATS:
        thr = thresholds(st, grouping, rec_star)
        flag = apply(grouping, thr); flags[(grouping, st)] = flag
        for g, mt in PT[grouping].items():
            r = rates(yte[mt], flag[mt])
            detail.append(dict(grouping=grouping, strategy=st, group=g, threshold=thr[g],
                               recall=r["recall"], precision=r["precision"], flag_rate=r["flag_rate"]))
        o = rates(yte, flag)
        summary.append(dict(grouping=grouping, strategy=st, F1=f1_score(yte, flag),
                            recall=o["recall"], precision=o["precision"], flag_rate=o["flag_rate"],
                            recall_gap=recall_gap(grouping, flag), net_benefit=net_benefit(yte, flag, thr_star)))

# bootstrap intervals for F1 and the recall gap (thresholds fixed, test rows resampled)
boots = {k: {"F1": [], "gap": []} for k in flags}
for _ in range(N_BOOT):
    idx = rng.integers(0, len(yte), len(yte))
    if yte[idx].min() == yte[idx].max(): continue
    for (grouping, st), f in flags.items():
        boots[(grouping, st)]["F1"].append(f1_score(yte[idx], f[idx]))
        boots[(grouping, st)]["gap"].append(recall_gap(grouping, f, idx))
summ = pd.DataFrame(summary)
for s, col in (("F1", "F1"), ("gap", "recall_gap")):
    lo = [np.percentile(boots[(r.grouping, r.strategy)][s], 2.5) for r in summ.itertuples()]
    hi = [np.percentile(boots[(r.grouping, r.strategy)][s], 97.5) for r in summ.itertuples()]
    summ[col + "_lo"], summ[col + "_hi"] = lo, hi
summ = summ.round(4); det = pd.DataFrame(detail).round(4)
summ.to_csv("figures/v2_opoint_summary.csv", index=False)
det.to_csv("figures/v2_opoint_detail.csv", index=False)

# ---------- 2. Price of equalising recall ----------
cost = []
for grouping in PT:
    for r in (0.4, 0.5, 0.6, 0.7, 0.8):
        f_eq = apply(grouping, thresholds("equal_recall", grouping, r))
        f_one = p_te >= float(np.quantile(p_va[yva == 1], 1 - r))
        e, o = rates(yte, f_eq), rates(yte, f_one)
        cost.append(dict(grouping=grouping, target_recall=r,
                         eq_flag_rate=e["flag_rate"], one_flag_rate=o["flag_rate"],
                         extra_flagged_pts=100 * (e["flag_rate"] - o["flag_rate"]),
                         eq_precision=e["precision"], one_precision=o["precision"],
                         eq_recall_gap=recall_gap(grouping, f_eq), one_recall_gap=recall_gap(grouping, f_one)))
cost = pd.DataFrame(cost).round(4)
cost.to_csv("figures/v2_opoint_cost_of_equal_recall.csv", index=False)

# ---------- 3. Decision curves by subgroup ----------
pts = np.arange(0.05, 0.41, 0.01)
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
colors = ["#E86A5C", "#028090", "#6A4C93"]
for ax, grouping in zip(axes, PT):
    for c, (g, mt) in zip(colors, PT[grouping].items()):
        y, p = yte[mt], p_te[mt]; prev = y.mean()
        ax.plot(pts, [net_benefit(y, p >= t, t) for t in pts], color=c, label=f"{g} model")
        ax.plot(pts, [prev - (1 - prev) * t / (1 - t) for t in pts], color=c, ls="--", lw=1, label=f"{g} treat all")
    ax.axhline(0, color="k", lw=0.8); ax.set_ylim(-0.05, 0.25)
    ax.set_title(f"Decision curves by {grouping}"); ax.set_xlabel("Threshold probability")
axes[0].set_ylabel("Net benefit"); axes[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig("figures/v2_fig_decision_curves.png", dpi=150); plt.close()

print("\n=== STRATEGY SUMMARY (test) ===\n", summ.to_string(index=False))
print("\n=== PER-GROUP DETAIL (test) ===\n", det.to_string(index=False))
print("\n=== COST OF EQUALISING RECALL ===\n", cost.to_string(index=False))