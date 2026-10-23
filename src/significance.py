import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, brier_score_loss
from pipeline import (X_tr, X_va, X_te, y_tr, y_va, y_te, models, SEED,
                      fit_calibrator, cv_brier)

WEIGHTED = False      # True = same setup as your published table; plain-vs-weighted lives in pipeline.py
N_BOOT   = 1000
MARGIN   = 0.005     # pre-declared equivalence margin on AUC
NAMES    = ["LR", "RF", "XGB", "LGBM"]
PAIRS    = list(combinations(NAMES, 2)) + [("XGB", "XGB_cal")]
ALPHA_ADJ = 0.05 / len(PAIRS)   # Bonferroni

# ---- 1. Fit on train, predict test (calibrator chosen on validation only) ----
fitted = {}
for k, m in models(WEIGHTED).items():
    m.fit(X_tr, y_tr)
    fitted[k] = m.predict_proba(X_te)[:, 1]
    if k == "XGB":
        s_va = m.predict_proba(X_va)[:, 1]
        kind = min(("platt", "isotonic"), key=lambda c: cv_brier(c, s_va, y_va))
        fitted["XGB_cal"] = fit_calibrator(kind, s_va, y_va)(fitted[k])
        print("XGB calibrator chosen on validation:", kind)
ALL = list(fitted)

# ---- 2. Paired bootstrap on the test set ----
rng = np.random.default_rng(SEED)
yt = np.asarray(y_te); n = len(yt)
auc_b   = {k: np.full(N_BOOT, np.nan) for k in ALL}
brier_b = {k: np.full(N_BOOT, np.nan) for k in ALL}
for b in range(N_BOOT):
    idx = rng.integers(0, n, n)                  # same resample for all models = paired
    if yt[idx].min() == yt[idx].max(): continue
    for k in ALL:
        auc_b[k][b]   = roc_auc_score(yt[idx], fitted[k][idx])
        brier_b[k][b] = brier_score_loss(yt[idx], fitted[k][idx])

def ci(a, al=0.05):
    return np.nanpercentile(a, [100 * al / 2, 100 * (1 - al / 2)])

summ = pd.DataFrame([dict(
    model=k, AUC=roc_auc_score(yt, fitted[k]), AUC_lo=ci(auc_b[k])[0], AUC_hi=ci(auc_b[k])[1],
    Brier=brier_score_loss(yt, fitted[k]), Brier_lo=ci(brier_b[k])[0], Brier_hi=ci(brier_b[k])[1]
) for k in ALL]).round(4)
summ.to_csv("figures/v2_bootstrap_ci_summary.csv", index=False)

rows = []
for a, b in PAIRS:
    d = auc_b[a] - auc_b[b]
    lo, hi = ci(d); lo_a, hi_a = ci(d, ALPHA_ADJ)
    rows.append(dict(comparison=f"{a} - {b}", dAUC=np.nanmean(d), CI95_lo=lo, CI95_hi=hi,
                     CIadj_lo=lo_a, CIadj_hi=hi_a,
                     significant_adj=(lo_a > 0) or (hi_a < 0),
                     practically_equivalent=(lo_a > -MARGIN) and (hi_a < MARGIN)))
pairs = pd.DataFrame(rows).round(4)
pairs.to_csv("figures/v2_pairwise_significance.csv", index=False)

# ---- 3. CI plot ----
fig, ax = plt.subplots(figsize=(8, 5))
pts = summ["AUC"].values
ax.errorbar(summ["model"], pts, yerr=[pts - summ["AUC_lo"].values, summ["AUC_hi"].values - pts],
            fmt="o", capsize=6, color="#028090", markersize=8, linewidth=2)
ax.set_ylabel("ROC-AUC"); ax.set_title("Test ROC-AUC with 95% bootstrap CIs")
plt.tight_layout(); plt.savefig("figures/v2_bootstrap_ci_plot.png", dpi=150); plt.close()

# ---- 4. 5-fold CV on train+validation (independent of test), now incl. LightGBM ----
X_dev = pd.concat([X_tr, X_va]); y_dev = np.concatenate([y_tr, y_va])
cv_rows = []
for k in NAMES:
    sc = []
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_dev, y_dev):
        m = models(WEIGHTED)[k]
        m.fit(X_dev.iloc[tr], y_dev[tr])
        sc.append(roc_auc_score(y_dev[te], m.predict_proba(X_dev.iloc[te])[:, 1]))
    cv_rows.append(dict(model=k, cv_mean=np.mean(sc), cv_std=np.std(sc, ddof=1)))
    print(k, "cv done")
cv = pd.DataFrame(cv_rows).round(4)
cv.to_csv("figures/v2_cv_all_models_summary.csv", index=False)

print("\nBootstrap:\n", summ.to_string(index=False))
print("\nPairwise (Bonferroni over", len(PAIRS), "tests):\n", pairs.to_string(index=False))
print("\n5-fold CV:\n", cv.to_string(index=False))