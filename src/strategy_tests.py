import numpy as np, pandas as pd
from pipeline import (X_tr, X_va, X_te, y_tr, y_va, y_te, models, SEED,
                      fit_calibrator, cv_brier, best_threshold)

N_BOOT = 1000
rng = np.random.default_rng(SEED)
yva, yte = np.asarray(y_va), np.asarray(y_te)

# ---- same model, calibrator and thresholds logic as operating_points.py ----
m = models(False)["XGB"]; m.fit(X_tr, y_tr)
s_va, s_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
kind = min(("platt", "isotonic"), key=lambda c: cv_brier(c, s_va, yva))
cal = fit_calibrator(kind, s_va, yva)
p_va, p_te = cal(s_va), cal(s_te)
thr_star = float(best_threshold(yva, p_va))
rec_star = float((p_va[yva == 1] >= thr_star).mean())
print(f"calibrator={kind}  global threshold={thr_star:.3f}  target recall={rec_star:.3f}")

def parts(X):
    return {"Age": {"18-39": X["Age"].between(1, 4).values, "40-59": X["Age"].between(5, 8).values,
                    "60+": X["Age"].between(9, 13).values},
            "Income": {"Lower (1-4)": X["Income"].between(1, 4).values,
                       "Higher (5-8)": X["Income"].between(5, 8).values},
            "Sex": {"Female": (X["Sex"] == 0).values, "Male": (X["Sex"] == 1).values}}
PV, PT = parts(X_va), parts(X_te)
SIGNED = {"Age": ("60+", "18-39"), "Income": ("Lower (1-4)", "Higher (5-8)"), "Sex": ("Male", "Female")}
STRATS = ("global", "group_F1", "equal_recall")

def thresholds(st, grouping):
    out = {}
    for g, mv in PV[grouping].items():
        if st == "global": out[g] = thr_star
        elif st == "group_F1": out[g] = float(best_threshold(yva[mv], p_va[mv]))
        else: out[g] = float(np.quantile(p_va[mv & (yva == 1)], 1 - rec_star))
    return out

flags = {}
for gr in PT:
    for st in STRATS:
        thr = thresholds(st, gr)
        f = np.zeros(len(yte), bool)
        for g, mt in PT[gr].items():
            f[mt] = p_te[mt] >= thr[g]
        flags[(gr, st)] = f

def stats(f, y, mA, mB):
    tp = (f & (y == 1)).sum(); fp = (f & (y == 0)).sum(); fn = (~f & (y == 1)).sum()
    n = len(y)
    rA = (f & (y == 1) & mA).sum() / max(((y == 1) & mA).sum(), 1)
    rB = (f & (y == 1) & mB).sum() / max(((y == 1) & mB).sum(), 1)
    return np.array([2 * tp / (2 * tp + fp + fn), f.mean() * 100,
                     tp / n - fp / n * thr_star / (1 - thr_star), rA - rB])
NAMES = ["F1", "flagged_pct", "net_benefit", "signed_recall_gap"]

def masks(gr):
    a, b = SIGNED[gr]; return PT[gr][a], PT[gr][b]

point = {k: stats(f, yte, *masks(k[0])) for k, f in flags.items()}
boot = {k: [] for k in flags}
for _ in range(N_BOOT):
    idx = rng.integers(0, len(yte), len(yte))
    y = yte[idx]
    if y.min() == y.max(): continue
    for k, f in flags.items():
        mA, mB = masks(k[0])
        boot[k].append(stats(f[idx], y, mA[idx], mB[idx]))
boot = {k: np.array(v) for k, v in boot.items()}
ci = lambda a: np.percentile(a, [2.5, 97.5], axis=0)

# ---- strategy summaries with signed gap ----
rows = []
for k in flags:
    lo, hi = ci(boot[k])
    for j, nme in enumerate(NAMES):
        rows.append(dict(grouping=k[0], strategy=k[1], stat=nme, estimate=point[k][j], lo=lo[j], hi=hi[j]))
summ = pd.DataFrame(rows).round(4)

# ---- paired differences between strategies (same resamples) ----
PAIRS = [("equal_recall", "global"), ("group_F1", "global"), ("equal_recall", "group_F1")]
rows = []
for gr in PT:
    for a, b in PAIRS:
        d = boot[(gr, a)] - boot[(gr, b)]
        est = point[(gr, a)] - point[(gr, b)]
        lo, hi = ci(d)
        for j, nme in enumerate(NAMES):
            rows.append(dict(grouping=gr, comparison=f"{a} - {b}", stat=nme, diff=est[j], lo=lo[j], hi=hi[j],
                             excludes_zero=bool(lo[j] > 0 or hi[j] < 0)))
diff = pd.DataFrame(rows).round(4)

summ.to_csv("figures/v2_strategy_summary_signed.csv", index=False)
diff.to_csv("figures/v2_strategy_paired_diffs.csv", index=False)
gap = summ[summ["stat"] == "signed_recall_gap"].drop(columns="stat")
print("\n=== SIGNED RECALL GAP BY STRATEGY ===\n", gap.to_string(index=False))
print("\n=== PAIRED DIFFERENCES BETWEEN STRATEGIES ===\n", diff.to_string(index=False))