import numpy as np, pandas as pd
from itertools import combinations
from sklearn.metrics import roc_auc_score
from pipeline import X_tr, X_te, y_tr, y_te, models, SEED

N_BOOT, MARGIN = 1000, 0.005
ytr, yte = np.asarray(y_tr), np.asarray(y_te)

BEST = {
    "LR": {},   # regularisation strength had no effect on validation AUC
    "RF": {"n_estimators": 300, "min_samples_leaf": 20, "max_features": "sqrt", "max_depth": 20},
    "XGB": {"subsample": 1.0, "reg_lambda": 5, "n_estimators": 600, "min_child_weight": 10,
            "max_depth": 4, "learning_rate": 0.03, "colsample_bytree": 0.85},
    "LGBM": {"subsample": 0.7, "reg_lambda": 10, "num_leaves": 31, "n_estimators": 400,
             "min_child_samples": 50, "max_depth": 4, "learning_rate": 0.05, "colsample_bytree": 1.0},
}
pred = {}
for name, params in BEST.items():
    m = models(False)[name]
    if params:
        m.set_params(**params)
        if name == "LGBM":
            m.set_params(subsample_freq=1)
    m.fit(X_tr, ytr); pred[name] = m.predict_proba(X_te)[:, 1]
    print(name, "fitted; test AUC =", round(roc_auc_score(yte, pred[name]), 4), flush=True)

NAMES = list(BEST); PAIRS = list(combinations(NAMES, 2)); ALPHA = 0.05 / len(PAIRS)
rng = np.random.default_rng(SEED)
boot = {k: [] for k in NAMES}
for _ in range(N_BOOT):
    i = rng.integers(0, len(yte), len(yte))
    if yte[i].min() == yte[i].max(): continue
    for k in NAMES: boot[k].append(roc_auc_score(yte[i], pred[k][i]))
boot = {k: np.array(v) for k, v in boot.items()}
ci = lambda a, al: np.percentile(a, [100 * al / 2, 100 * (1 - al / 2)])

rows = []
for a, b in PAIRS:
    d = boot[a] - boot[b]; lo, hi = ci(d, ALPHA)
    rows.append(dict(comparison=f"{a} - {b}", dAUC=roc_auc_score(yte, pred[a]) - roc_auc_score(yte, pred[b]),
                     CIadj_lo=lo, CIadj_hi=hi, significant=bool(lo > 0 or hi < 0),
                     within_margin=bool(lo > -MARGIN and hi < MARGIN)))
tab = pd.DataFrame(rows).round(4)
tab.to_csv("figures/v2_tuned_pairwise.csv", index=False)
print(f"\n=== TUNED MODELS, PAIRED BOOTSTRAP (Bonferroni over {len(PAIRS)}) ===\n", tab.to_string(index=False))