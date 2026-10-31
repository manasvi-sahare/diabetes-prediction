import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from pipeline import X_tr, X_te, y_tr, y_te, models, SEED

N_BOOT = 500
rng = np.random.default_rng(SEED)
ytr, yte = np.asarray(y_tr), np.asarray(y_te)

def groups(X):
    a = X["Age"].values
    return {"18-39": (a >= 1) & (a <= 4), "40-59": (a >= 5) & (a <= 8), "60+": (a >= 9) & (a <= 13)}
GTR, GTE = groups(X_tr), groups(X_te)

def fit_pred(name, Xa, ya, Xb, cols=None):
    m = models(False)[name]
    if cols is not None:
        Xa, Xb = Xa[cols], Xb[cols]
    m.fit(Xa, ya)
    return m.predict_proba(Xb)[:, 1]

def auc_ci(y, p):
    n, bs = len(y), []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        if y[i].min() != y[i].max():
            bs.append(roc_auc_score(y[i], p[i]))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return roc_auc_score(y, p), lo, hi

# global unweighted XGBoost (same as the paper)
p_xgb = fit_pred("XGB", X_tr, ytr, X_te)

# ---- 1. AUC by single age category ----
rows = []
for a in range(1, 14):
    m = (X_te["Age"].values == a)
    y, p = yte[m], p_xgb[m]
    auc, lo, hi = auc_ci(y, p)
    rows.append(dict(age_category=a, n=int(m.sum()), prevalence=y.mean(), AUC=auc, lo=lo, hi=hi))
t1 = pd.DataFrame(rows).round(4)

fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.errorbar(t1["age_category"], t1["AUC"], yerr=[t1["AUC"] - t1["lo"], t1["hi"] - t1["AUC"]],
            fmt="o-", capsize=3, color="#028090")
ax.set_xlabel("BRFSS age category (1 = 18-24, 13 = 80+)"); ax.set_ylabel("ROC-AUC")
ax.set_title("Global model AUC by single age category (95% bootstrap CI)")
plt.tight_layout(); plt.savefig("figures/v2_fig_age_categories.png", dpi=150); plt.close()

# ---- 2. age-specific models versus the global model ----
rows = []
for name in ("XGB", "LR"):
    p_glob = fit_pred(name, X_tr, ytr, X_te)
    for g in GTE:
        mt, mr = GTE[g], GTR[g]
        p_spec = fit_pred(name, X_tr[mr], ytr[mr], X_te[mt])
        y, pg = yte[mt], p_glob[mt]
        d = []
        for _ in range(N_BOOT):
            i = rng.integers(0, len(y), len(y))
            if y[i].min() != y[i].max():
                d.append(roc_auc_score(y[i], p_spec[i]) - roc_auc_score(y[i], pg[i]))
        a_s, a_g = roc_auc_score(y, p_spec), roc_auc_score(y, pg)
        rows.append(dict(model=name, group=g, n_train=int(mr.sum()), n_test=int(mt.sum()),
                         AUC_global=a_g, AUC_specific=a_s, diff=a_s - a_g,
                         lo=np.percentile(d, 2.5), hi=np.percentile(d, 97.5)))
t2 = pd.DataFrame(rows).round(4)

# ---- 3. learning curves within each age group (single seed, so treat as indicative) ----
rows = []
for g in GTE:
    perm = rng.permutation(np.where(GTR[g])[0])
    for f in (0.1, 0.25, 0.5, 1.0):
        sub = perm[:int(len(perm) * f)]
        p = fit_pred("XGB", X_tr.iloc[sub], ytr[sub], X_te[GTE[g]])
        rows.append(dict(group=g, fraction=f, n_train=len(sub), AUC=roc_auc_score(yte[GTE[g]], p)))
t3 = pd.DataFrame(rows).round(4)

# ---- 4. feature saturation: mean and strength of separation within each age group ----
FEATS = ["GenHlth", "HighBP", "HighChol", "BMI", "DiffWalk", "HeartDiseaseorAttack", "PhysHlth", "Age"]
rows = []
for g, mt in GTE.items():
    y = yte[mt]
    for f in FEATS:
        x = X_te[f].values[mt]
        a = roc_auc_score(y, x)
        rows.append(dict(group=g, feature=f, mean=x.mean(), separation_AUC=max(a, 1 - a)))
t4 = pd.DataFrame(rows)
t4_mean = t4.pivot(index="feature", columns="group", values="mean").loc[FEATS, ["18-39", "40-59", "60+"]].round(3)
t4_auc = t4.pivot(index="feature", columns="group", values="separation_AUC").loc[FEATS, ["18-39", "40-59", "60+"]].round(3)

# ---- 5. ablation: retrain without the Age feature ----
cols = [c for c in X_tr.columns if c != "Age"]
p_noage = fit_pred("XGB", X_tr, ytr, X_te, cols=cols)
rows = []
for g, mt in GTE.items():
    y = yte[mt]
    rows.append(dict(group=g, AUC_with_age=roc_auc_score(y, p_xgb[mt]), AUC_without_age=roc_auc_score(y, p_noage[mt])))
t5 = pd.DataFrame(rows).round(4)
t5["change"] = (t5["AUC_without_age"] - t5["AUC_with_age"]).round(4)

for n, d in (("v2_age_by_category", t1), ("v2_age_specific_models", t2), ("v2_age_learning_curve", t3),
             ("v2_age_feature_means", t4_mean.reset_index()), ("v2_age_feature_separation", t4_auc.reset_index()),
             ("v2_age_ablation", t5)):
    d.to_csv(f"figures/{n}.csv", index=False)

print("\n=== 1. AUC BY AGE CATEGORY ===\n", t1.to_string(index=False))
print("\n=== 2. AGE-SPECIFIC VS GLOBAL MODEL ===\n", t2.to_string(index=False))
print("\n=== 3. LEARNING CURVES ===\n", t3.to_string(index=False))
print("\n=== 4a. FEATURE MEANS BY AGE GROUP ===\n", t4_mean.to_string())
print("\n=== 4b. WITHIN-GROUP SEPARATION AUC ===\n", t4_auc.to_string())
print("\n=== 5. ABLATION: WITHOUT AGE FEATURE ===\n", t5.to_string(index=False))