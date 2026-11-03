import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score
from pipeline import df, models, fit_calibrator, cv_brier, best_threshold, ece_eqmass

N_REPEATS = 10                          # raise to 20 if time allows
SEEDS = list(range(101, 101 + N_REPEATS))
feats = [c for c in df.columns if c != "Diabetes_binary"]
X_all, y_all = df[feats], df["Diabetes_binary"].values

def groups(X):
    a, inc, sx = X["Age"].values, X["Income"].values, X["Sex"].values
    return {"age": {"18-39": (a >= 1) & (a <= 4), "40-59": (a >= 5) & (a <= 8), "60+": (a >= 9) & (a <= 13)},
            "inc": {"low": (inc >= 1) & (inc <= 4), "high": (inc >= 5) & (inc <= 8)},
            "sex": {"F": sx == 0, "M": sx == 1}}
PAIR = {"age": ("60+", "18-39"), "inc": ("low", "high"), "sex": ("M", "F")}   # signed recall gap = first minus second
AUCP = {"age": ("18-39", "60+"), "inc": ("high", "low"), "sex": ("F", "M")}   # AUC gap = first minus second

def recall(y, f):
    return f[y == 1].mean()

def signed_gap(gr, y, f, GT):
    a, b = PAIR[gr]
    return recall(y[GT[gr][a]], f[GT[gr][a]]) - recall(y[GT[gr][b]], f[GT[gr][b]])

def one_split(seed):
    X_tmp, X_te, y_tmp, y_te = train_test_split(X_all, y_all, test_size=0.20, stratify=y_all, random_state=seed)
    X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=seed)
    out, preds = {"seed": seed}, {}
    for name, m in models(False).items():
        m.fit(X_tr, y_tr)
        preds[name] = (m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1])
        out[f"AUC_{name}"] = roc_auc_score(y_te, preds[name][1])

    s_va, s_te = preds["XGB"]
    kind = min(("platt", "isotonic"), key=lambda c: cv_brier(c, s_va, y_va))
    cal = fit_calibrator(kind, s_va, y_va)
    p_va, p_te = cal(s_va), cal(s_te)
    thr = float(best_threshold(y_va, p_va))
    rec_star = float((p_va[y_va == 1] >= thr).mean())
    flag = p_te >= thr
    out.update(thr=thr, F1_global=f1_score(y_te, flag))

    GV, GT = groups(X_va), groups(X_te)
    out["max_subgroup_ECE"] = max(ece_eqmass(y_te[m], p_te[m]) for gr in GT.values() for m in gr.values())
    for gr in GT:
        a, b = AUCP[gr]
        out[f"AUCgap_{gr}"] = roc_auc_score(y_te[GT[gr][a]], p_te[GT[gr][a]]) - roc_auc_score(y_te[GT[gr][b]], p_te[GT[gr][b]])
        out[f"recall_gap_global_{gr}"] = signed_gap(gr, y_te, flag, GT)
        thr_g = {g: float(np.quantile(p_va[mv & (y_va == 1)], 1 - rec_star)) for g, mv in GV[gr].items()}
        f_eq = np.zeros(len(y_te), bool)
        for g, mt in GT[gr].items():
            f_eq[mt] = p_te[mt] >= thr_g[g]
        out[f"recall_gap_equal_{gr}"] = signed_gap(gr, y_te, f_eq, GT)
        out[f"dF1_equal_{gr}"] = f1_score(y_te, f_eq) - out["F1_global"]
        out[f"dFlagged_pts_equal_{gr}"] = 100 * (f_eq.mean() - flag.mean())
    return out

rows = []
for s in SEEDS:
    rows.append(one_split(s)); print("split", s, "done", flush=True)
res = pd.DataFrame(rows)
res.to_csv("figures/v2_repeated_splits_per_seed.csv", index=False)

cols = [c for c in res.columns if c != "seed"]
summ = pd.DataFrame({"mean": res[cols].mean(), "sd": res[cols].std(), "min": res[cols].min(), "max": res[cols].max()}).round(4)
summ.to_csv("figures/v2_repeated_splits_summary.csv")
print("\n=== SUMMARY OVER", N_REPEATS, "SPLITS ===\n", summ.to_string())

n = len(res)
print("\n=== HOW OFTEN EACH CLAIM HOLDS (out of", n, "splits) ===")
for m_ in ("RF", "XGB", "LGBM"):
    print(f"{m_} AUC > LR AUC:", int((res[f'AUC_{m_}'] > res['AUC_LR']).sum()))
print("age recall gap > 0.3 (single threshold):", int((res["recall_gap_global_age"] > 0.3).sum()))
print("income recall gap > 0.1 (single threshold):", int((res["recall_gap_global_inc"] > 0.1).sum()))
print("age equal-recall costs more F1 than income:", int((res["dF1_equal_age"] < res["dF1_equal_inc"]).sum()))
print("max subgroup ECE <= 0.01:", int((res["max_subgroup_ECE"] <= 0.01).sum()))