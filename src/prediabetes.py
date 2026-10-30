import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from pipeline import (df as DF, X_tr, X_va, X_te, y_tr, y_va, y_te, models, SEED,
                      fit_calibrator, cv_brier, best_threshold)

N_BOOT = 500
rng = np.random.default_rng(SEED)

# ---- align the 3-class label with the deduplicated data ----
b = pd.read_csv("data/diabetes.csv")
t = pd.read_csv("data/diabetes_012.csv")
t3 = t.loc[~b.duplicated().values].reset_index(drop=True)
feats = [c for c in DF.columns if c != "Diabetes_binary"]
assert len(t3) == len(DF) and (t3[feats].values == DF[feats].values).all(), "alignment failed"
yte = np.asarray(y_te)
c_te = t3["Diabetes_012"].values[X_te.index.values].astype(int)
assert ((c_te == 2).astype(int) == yte).all()

# ---- same model, calibrator and threshold as the paper ----
m = models(False)["XGB"]; m.fit(X_tr, y_tr)
s_va, s_te = m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1]
yva = np.asarray(y_va)
kind = min(("platt", "isotonic"), key=lambda c: cv_brier(c, s_va, yva))
cal = fit_calibrator(kind, s_va, yva)
p_va, p_te = cal(s_va), cal(s_te)
thr = float(best_threshold(yva, p_va)); flag = p_te >= thr
print(f"calibrator={kind}  threshold={thr:.3f}")

ALL = np.arange(len(yte))
def auc_of(idx, pos, neg_, g=None):
    c, p = c_te[idx], p_te[idx]
    mk = np.isin(c, pos + neg_)
    if g is not None: mk &= g[idx]
    y = np.isin(c[mk], pos).astype(int)
    return np.nan if (len(y) == 0 or y.min() == y.max()) else roc_auc_score(y, p[mk])
boot_idx = [rng.integers(0, len(yte), len(yte)) for _ in range(N_BOOT)]
ci = lambda a: np.nanpercentile(a, [2.5, 97.5])

# 1. by true category
cls = {0: "No diabetes", 1: "Prediabetes", 2: "Diabetes"}
t1 = pd.DataFrame([dict(category=cls[k], n=int((c_te == k).sum()),
                        mean_predicted_risk=p_te[c_te == k].mean(),
                        flagged=flag[c_te == k].mean()) for k in (0, 1, 2)]).round(4)

# 2. separation between categories
CMP = {"Diabetes vs no diabetes": ([2], [0]), "Prediabetes vs no diabetes": ([1], [0]),
       "Diabetes vs prediabetes": ([2], [1]), "Diabetes vs all others (paper AUC)": ([2], [0, 1])}
rows = []
for name, (pos, ng) in CMP.items():
    lo, hi = ci([auc_of(i, pos, ng) for i in boot_idx])
    rows.append(dict(comparison=name, AUC=auc_of(ALL, pos, ng), lo=lo, hi=hi))
t2 = pd.DataFrame(rows).round(4)

# 3. false-positive composition
neg = yte == 0; fp = flag & neg; tp = flag & (yte == 1)
t3 = pd.DataFrame([dict(
    prediabetes_share_of_negatives=(c_te[neg] == 1).mean(),
    prediabetes_share_of_false_positives=(c_te[fp] == 1).mean(),
    enrichment=(c_te[fp] == 1).mean() / (c_te[neg] == 1).mean(),
    precision_paper=tp.sum() / flag.sum(),
    precision_if_prediabetes_counted=(flag & (c_te > 0)).sum() / flag.sum())]).round(4)

# 4. subgroups: does excluding prediabetes change the AUC gaps?
G = {"Sex: Female": (X_te["Sex"] == 0).values, "Sex: Male": (X_te["Sex"] == 1).values,
     "Age: 18-39": X_te["Age"].between(1, 4).values, "Age: 40-59": X_te["Age"].between(5, 8).values,
     "Age: 60+": X_te["Age"].between(9, 13).values,
     "Income: Lower (1-4)": X_te["Income"].between(1, 4).values,
     "Income: Higher (5-8)": X_te["Income"].between(5, 8).values}
t4 = pd.DataFrame([dict(group=g, n=int(gm.sum()), prediabetes_pct=(c_te[gm] == 1).mean(),
                        prediabetes_flagged=flag[gm & (c_te == 1)].mean(),
                        AUC_paper=auc_of(ALL, [2], [0, 1], gm),
                        AUC_excl_prediabetes=auc_of(ALL, [2], [0], gm)) for g, gm in G.items()]).round(4)

PAIRS = [("Age: 18-39 minus 60+", "Age: 18-39", "Age: 60+"),
         ("Income: Higher minus Lower", "Income: Higher (5-8)", "Income: Lower (1-4)"),
         ("Sex: Female minus Male", "Sex: Female", "Sex: Male")]
DEF = {"paper": ([2], [0, 1]), "excl_prediabetes": ([2], [0])}
rows = []
for label, g1, g2 in PAIRS:
    for dn, (pos, ng) in DEF.items():
        est = auc_of(ALL, pos, ng, G[g1]) - auc_of(ALL, pos, ng, G[g2])
        lo, hi = ci([auc_of(i, pos, ng, G[g1]) - auc_of(i, pos, ng, G[g2]) for i in boot_idx])
        rows.append(dict(contrast=label, auc_definition=dn, gap=est, lo=lo, hi=hi))
t5 = pd.DataFrame(rows).round(4)

for n, d in (("v2_pre_by_category", t1), ("v2_pre_separation", t2), ("v2_pre_false_positives", t3),
             ("v2_pre_subgroups", t4), ("v2_pre_gap_contrasts", t5)):
    d.to_csv(f"figures/{n}.csv", index=False)
print("\n=== BY TRUE CATEGORY ===\n", t1.to_string(index=False))
print("\n=== SEPARATION ===\n", t2.to_string(index=False))
print("\n=== FALSE POSITIVES ===\n", t3.to_string(index=False))
print("\n=== SUBGROUPS ===\n", t4.to_string(index=False))
print("\n=== AUC GAPS, WITH AND WITHOUT PREDIABETES ===\n", t5.to_string(index=False))