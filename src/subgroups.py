import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score
from pipeline import (X_tr, X_va, X_te, y_tr, y_va, y_te, models, SEED,
                      fit_calibrator, cv_brier, ece_eqmass, cal_slope_intercept, best_threshold)

WEIGHTED = False       # matches the paper's setup
N_BOOT   = 500        # raise to 1000 for the final run
rng = np.random.default_rng(SEED)
yva, yte = np.asarray(y_va), np.asarray(y_te)

# ---------- 1. Fit models; calibrator and threshold chosen on VALIDATION only ----------
fit = {}
for k, m in models(WEIGHTED).items():
    m.fit(X_tr, y_tr)
    fit[k] = (m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1])
    print(k, "fitted")

s_va, s_te = fit["XGB"]
kind = min(("platt", "isotonic"), key=lambda c: cv_brier(c, s_va, yva))
cal = fit_calibrator(kind, s_va, yva)
p_va, p_te = cal(s_va), cal(s_te)
thr = best_threshold(yva, p_va)
print(f"XGB calibrator: {kind} | validation-chosen threshold: {thr:.3f}")

# ---------- 2. Group definitions (BRFSS coding) ----------
def groups(X):
    return {
        "Sex: Female": (X["Sex"] == 0).values, "Sex: Male": (X["Sex"] == 1).values,
        "Age: 18-39": X["Age"].between(1, 4).values, "Age: 40-59": X["Age"].between(5, 8).values,
        "Age: 60+": X["Age"].between(9, 13).values,
        "Income: Lower (1-4)": X["Income"].between(1, 4).values,
        "Income: Higher (5-8)": X["Income"].between(5, 8).values,
    }
G_va, G_te = groups(X_va), groups(X_te)
AGE = ["Age: 18-39", "Age: 40-59", "Age: 60+"]

# ---------- 3. Metrics and bootstrap ----------
STAT = {
    "AUC": roc_auc_score,
    "ECE": ece_eqmass,
    "OE":  lambda y, p: y.mean() / p.mean(),
    "BSS": lambda y, p: 1 - brier_score_loss(y, p) / (y.mean() * (1 - y.mean())),
}
ci = lambda a: np.percentile(a, [2.5, 97.5])

def boot(y, preds, stats):
    """Resample once per iteration, evaluate every prediction vector on the same resample."""
    out = {nm: {s: [] for s in stats} for nm in preds}
    for _ in range(N_BOOT):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max(): continue
        for nm, p in preds.items():
            for s in stats: out[nm][s].append(STAT[s](y[i], p[i]))
    return {nm: {s: np.array(v) for s, v in d.items()} for nm, d in out.items()}

def paired(a, b):
    k = min(len(a), len(b)); return a[:k], b[:k]

# ---------- 4. Full subgroup table ----------
rows, B = [], {}
for g, m in G_te.items():
    y, p = yte[m], p_te[m]
    B[g] = boot(y, {"cal": p}, ["AUC", "ECE", "OE", "BSS"])["cal"]
    sl, ic = cal_slope_intercept(y, p); pred = p >= thr
    r = dict(group=g, N=int(m.sum()), prev=y.mean(), mean_pred=p.mean(),
             Brier=brier_score_loss(y, p), slope=sl, intercept=ic,
             recall=recall_score(y, pred), precision=precision_score(y, pred, zero_division=0))
    for s in ["AUC", "ECE", "OE", "BSS"]:
        r[s] = STAT[s](y, p); r[s + "_lo"], r[s + "_hi"] = ci(B[g][s])
    rows.append(r)
tab = pd.DataFrame(rows).round(4)
tab.to_csv("figures/v2_subgroup_metrics.csv", index=False)
PT = {r["group"]: r for r in rows}

# ---------- 5. Tested contrasts ----------
LI, HI = "Income: Lower (1-4)", "Income: Higher (5-8)"
CONTRASTS = [
    ("ECE ratio: Lower / Higher income", LI, HI, "ECE", "ratio"),
    ("ECE diff:  Lower - Higher income", LI, HI, "ECE", "diff"),
    ("O/E diff:  Lower - Higher income", LI, HI, "OE", "diff"),
    ("AUC diff:  Lower - Higher income", LI, HI, "AUC", "diff"),
    ("AUC diff:  18-39 - 60+",           "Age: 18-39", "Age: 60+", "AUC", "diff"),
    ("ECE diff:  60+ - 40-59",           "Age: 60+", "Age: 40-59", "ECE", "diff"),
    ("BSS diff:  60+ - 18-39",           "Age: 60+", "Age: 18-39", "BSS", "diff"),
    ("AUC diff:  Female - Male",         "Sex: Female", "Sex: Male", "AUC", "diff"),
    ("ECE diff:  Male - Female",         "Sex: Male", "Sex: Female", "ECE", "diff"),
]
crow = []
for lab, g1, g2, s, how in CONTRASTS:
    a, b = paired(B[g1][s], B[g2][s])
    d = a - b if how == "diff" else a / b
    est = PT[g1][s] - PT[g2][s] if how == "diff" else PT[g1][s] / PT[g2][s]
    lo, hi = ci(d); null = 0 if how == "diff" else 1
    crow.append(dict(contrast=lab, estimate=est, lo=lo, hi=hi, excludes_null=bool(lo > null or hi < null)))
contr = pd.DataFrame(crow).round(4)
contr.to_csv("figures/v2_subgroup_contrasts.csv", index=False)

# ---------- 6. Age gap for ALL models (fixed contrast, with CI) ----------
arow = []
for k, (_, st) in fit.items():
    per = {g: boot(yte[G_te[g]], {k: st[G_te[g]]}, ["AUC"])[k]["AUC"] for g in AGE}
    a, b = paired(per["Age: 18-39"], per["Age: 60+"]); d = a - b
    pt = {g: roc_auc_score(yte[G_te[g]], st[G_te[g]]) for g in AGE}
    arow.append(dict(model=k, **{g: pt[g] for g in AGE}, gap_18_39_minus_60=pt["Age: 18-39"] - pt["Age: 60+"],
                     gap_lo=ci(d)[0], gap_hi=ci(d)[1]))
agegap = pd.DataFrame(arow).round(4)
agegap.to_csv("figures/v2_age_gap_all_models.csv", index=False)

# Diagnostic: is the 60+ AUC drop partly range restriction on Age itself?
print("\nAge-only AUC within each age bucket vs full-model AUC:")
for g in AGE:
    m = G_te[g]
    print(f"  {g}: Age feature alone = {roc_auc_score(yte[m], X_te['Age'].values[m]):.4f} | model = {PT[g]['AUC']:.4f}")

# ---------- 7. Step 4: group-wise recalibration (fit on validation rows, judged on test) ----------
glob_platt = fit_calibrator("platt", s_va, yva)
rrow = []
for g in G_te:
    mv, mt = G_va[g], G_te[g]
    gp = fit_calibrator("platt", s_va[mv], yva[mv])
    y = yte[mt]
    preds = {"global_chosen": p_te[mt], "global_platt": glob_platt(s_te[mt]), "group_platt": gp(s_te[mt])}
    b = boot(y, preds, ["ECE"])
    d = b["group_platt"]["ECE"] - b["global_chosen"]["ECE"]
    rrow.append(dict(group=g,
                     ECE_global_chosen=ece_eqmass(y, preds["global_chosen"]),
                     ECE_global_platt=ece_eqmass(y, preds["global_platt"]),
                     ECE_group_platt=ece_eqmass(y, preds["group_platt"]),
                     change_vs_global=ece_eqmass(y, preds["group_platt"]) - ece_eqmass(y, preds["global_chosen"]),
                     change_lo=ci(d)[0], change_hi=ci(d)[1]))
recal = pd.DataFrame(rrow).round(4)
recal.to_csv("figures/v2_group_recalibration.csv", index=False)

for title, df in [("SUBGROUP METRICS", tab), ("CONTRASTS", contr), ("AGE GAP, ALL MODELS", agegap), ("GROUP-WISE RECALIBRATION", recal)]:
    print(f"\n=== {title} ===\n", df.to_string(index=False))