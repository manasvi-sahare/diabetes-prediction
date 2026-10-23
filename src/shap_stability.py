import os, joblib
import numpy as np, pandas as pd, shap
import matplotlib.pyplot as plt
from itertools import combinations
from scipy.stats import spearmanr
from sklearn.model_selection import StratifiedKFold, train_test_split
from pipeline import X_tr, X_va, y_tr, models, SEED

# ---------------- settings ----------------
WEIGHTED  = True               # keep consistent with significance.py / subgroups.py
N_EXPLAIN = 500                # held-out validation rows explained for every model and fold
NAMES     = ["LR", "XGB", "LGBM", "RF"]   # RF last: slowest. Remove "RF" to run only the fast models
RF_TREES  = 50                 # stability check only; disclose in Methods
FEATS     = X_tr.columns.tolist()
y_tr      = np.asarray(y_tr)

os.makedirs("cache", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# Fixed explained rows: held out from all training, identical for every model and fold
X_exp = X_va.sample(N_EXPLAIN, random_state=SEED)

# ---------------- SHAP helpers ----------------
def mean_abs_shap(name, model, X_fit):
    if name == "LR":
        sc = model.named_steps["standardscaler"]
        lr = model.named_steps["logisticregression"]
        bg = sc.transform(X_fit.sample(100, random_state=SEED))
        sv = shap.LinearExplainer(lr, bg).shap_values(sc.transform(X_exp))
    else:
        sv = shap.TreeExplainer(model).shap_values(X_exp)
    if isinstance(sv, list):            # some versions return [class0, class1]
        sv = sv[-1]
    sv = np.asarray(sv)
    if sv.ndim == 3:                    # (n, features, classes) -> positive class
        sv = sv[:, :, -1]
    return pd.Series(np.abs(sv).mean(axis=0), index=FEATS)

def fit_importance(name, X_fit, y_fit, tag):
    path = f"cache/shap_{name}_{tag}_w{int(WEIGHTED)}_n{N_EXPLAIN}.pkl"
    if os.path.exists(path):
        return joblib.load(path)
    m = models(WEIGHTED)[name]
    if name == "RF":
        m.set_params(n_estimators=RF_TREES)
    m.fit(X_fit, y_fit)
    s = mean_abs_shap(name, m, X_fit)
    joblib.dump(s, path)
    print(f"   {name} {tag} done", flush=True)
    return s

# ---------------- stability metrics ----------------
top = lambda s, k: set(s.nlargest(k).index)
jac = lambda a, b: len(a & b) / len(a | b)
rho = lambda a, b: spearmanr(a.values, b.values)[0]

def stability(imps):
    pr = list(combinations(range(len(imps)), 2))
    r = [rho(imps[i], imps[j]) for i, j in pr]
    return dict(rho_mean=np.mean(r), rho_min=np.min(r),
                top5_jaccard=np.mean([jac(top(imps[i], 5), top(imps[j], 5)) for i, j in pr]),
                top10_jaccard=np.mean([jac(top(imps[i], 10), top(imps[j], 10)) for i, j in pr]))

# ---------------- splits ----------------
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X_tr, y_tr))
a_idx, b_idx = train_test_split(np.arange(len(X_tr)), test_size=0.5,
                                stratify=y_tr, random_state=SEED)   # disjoint halves

# ---------------- run ----------------
cv_rows, half_rows, full = [], [], {}
for name in NAMES:
    print(f"\n{name}: starting", flush=True)

    cv_imps = [fit_importance(name, X_tr.iloc[tr], y_tr[tr], f"cv{k}")
               for k, (tr, _) in enumerate(folds)]
    cv_rows.append(dict(model=name, **stability(cv_imps)))

    h = [fit_importance(name, X_tr.iloc[i], y_tr[i], f"half{n}")
         for n, i in enumerate((a_idx, b_idx))]
    half_rows.append(dict(model=name, rho=rho(*h),
                          top5_jaccard=jac(top(h[0], 5), top(h[1], 5)),
                          top10_jaccard=jac(top(h[0], 10), top(h[1], 10))))

    full[name] = fit_importance(name, X_tr, y_tr, "full")
    print(f"{name}: finished", flush=True)

cv_tab   = pd.DataFrame(cv_rows).round(4)
half_tab = pd.DataFrame(half_rows).round(4)
cv_tab.to_csv("figures/v2_shap_stability_cv.csv", index=False)
half_tab.to_csv("figures/v2_shap_stability_splithalf.csv", index=False)

# ---------------- cross-model agreement (ranks and normalized shares only) ----------------
rho_mat = pd.DataFrame(index=NAMES, columns=NAMES, dtype=float)
for a in NAMES:
    for b in NAMES:
        rho_mat.loc[a, b] = rho(full[a], full[b])
rho_mat = rho_mat.round(4)
rho_mat.to_csv("figures/v2_shap_cross_model_rho.csv")

pair_tab = pd.DataFrame([dict(pair=f"{a} vs {b}", rho=rho(full[a], full[b]),
                              top5_jaccard=jac(top(full[a], 5), top(full[b], 5)),
                              top10_jaccard=jac(top(full[a], 10), top(full[b], 10)))
                         for a, b in combinations(NAMES, 2)]).round(4)
pair_tab.to_csv("figures/v2_shap_cross_model_pairs.csv", index=False)

share = pd.DataFrame({k: v / v.sum() for k, v in full.items()})
ranks = pd.DataFrame({k: v.rank(ascending=False).astype(int) for k, v in full.items()})
out = share.round(4).add_suffix("_share").join(ranks.add_suffix("_rank"))
out = out.loc[share.mean(axis=1).sort_values(ascending=False).index]
out.to_csv("figures/v2_shap_importance_all_models.csv")

# ---------------- plot ----------------
t10 = share.loc[share.mean(axis=1).nlargest(10).index].iloc[::-1]
ax = t10.plot.barh(figsize=(8, 6), width=0.8)
ax.set_xlabel("Share of total mean |SHAP| (units differ by model, so shares are compared)")
ax.set_title("SHAP importance, top 10 features, all models")
plt.tight_layout()
plt.savefig("figures/v2_shap_all_models.png", dpi=150)
plt.close()

# ---------------- print ----------------
print("\n=== CV-fold stability (explained rows fixed) ===\n", cv_tab.to_string(index=False))
print("\n=== Split-half stability (disjoint training data) ===\n", half_tab.to_string(index=False))
print("\n=== Cross-model Spearman rho ===\n", rho_mat.to_string())
print("\n=== Cross-model pairs ===\n", pair_tab.to_string(index=False))
print("\n=== Importance shares and ranks (top 10) ===\n", out.head(10).to_string())