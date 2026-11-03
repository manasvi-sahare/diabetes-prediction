import numpy as np, pandas as pd
from sklearn.model_selection import ParameterSampler
from sklearn.metrics import roc_auc_score, log_loss, f1_score
from pipeline import (X_tr, X_va, X_te, y_tr, y_va, y_te, models, SEED,
                      fit_calibrator, cv_brier, best_threshold, ece_eqmass)

N_ITER = 8                       # random configurations per model; lower if it is too slow
ytr, yva, yte = np.asarray(y_tr), np.asarray(y_va), np.asarray(y_te)

GRIDS = {
    "LR":   {"logisticregression__C": [0.01, 0.1, 1, 10]},
    "RF":   {"n_estimators": [150, 300], "max_depth": [8, 12, 16, 20],
             "min_samples_leaf": [5, 10, 20, 40], "max_features": ["sqrt", 0.3, 0.5]},
    "XGB":  {"n_estimators": [200, 400, 600], "max_depth": [3, 4, 6, 8],
             "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
             "colsample_bytree": [0.7, 0.85, 1.0], "min_child_weight": [1, 5, 10],
             "reg_lambda": [1, 5, 10]},
    "LGBM": {"n_estimators": [200, 400, 600], "max_depth": [4, 6, 8], "num_leaves": [15, 31, 63],
             "learning_rate": [0.03, 0.05, 0.1], "subsample": [0.7, 0.85, 1.0],
             "colsample_bytree": [0.7, 0.85, 1.0], "min_child_samples": [20, 50, 100],
             "reg_lambda": [1, 5, 10]},
}

def build(name, params):
    m = models(False)[name]
    if params:
        m.set_params(**params)
        if name == "LGBM" and params.get("subsample", 1.0) < 1.0:
            m.set_params(subsample_freq=1)          # LightGBM ignores subsample without this
    return m

# ---------- 1. search on train, score on validation (defaults are candidate zero) ----------
best, rows = {}, []
for name, grid in GRIDS.items():
    cands = [{}] + list(ParameterSampler(grid, N_ITER, random_state=SEED))
    scored = []
    for i, params in enumerate(cands):
        m = build(name, params); m.fit(X_tr, ytr)
        p = m.predict_proba(X_va)[:, 1]
        scored.append((roc_auc_score(yva, p), log_loss(yva, p), params))
        print(f"{name} {i}/{len(cands)-1}  val AUC={scored[-1][0]:.4f}  logloss={scored[-1][1]:.4f}", flush=True)
    best[name] = max(scored, key=lambda r: r[0])
    rows.append(dict(model=name, default_val_auc=scored[0][0], tuned_val_auc=best[name][0],
                     tuned_is_default=(best[name][2] == {}), params=str(best[name][2])))
search = pd.DataFrame(rows).round(4)

# ---------- 2. refit best config on train, evaluate on test once ----------
pred = {}
for name in GRIDS:
    for tag, params in (("default", {}), ("tuned", best[name][2])):
        m = build(name, params); m.fit(X_tr, ytr)
        pred[(name, tag)] = (m.predict_proba(X_va)[:, 1], m.predict_proba(X_te)[:, 1])
test = pd.DataFrame([dict(model=n, default_test_auc=roc_auc_score(yte, pred[(n, "default")][1]),
                          tuned_test_auc=roc_auc_score(yte, pred[(n, "tuned")][1])) for n in GRIDS]).round(4)
test["change"] = (test["tuned_test_auc"] - test["default_test_auc"]).round(4)

# ---------- 3. does tuning change the subgroup and operating-point conclusions? (XGBoost) ----------
def groups(X):
    a, inc, sx = X["Age"].values, X["Income"].values, X["Sex"].values
    return {"age": {"18-39": (a >= 1) & (a <= 4), "40-59": (a >= 5) & (a <= 8), "60+": (a >= 9) & (a <= 13)},
            "inc": {"low": (inc >= 1) & (inc <= 4), "high": (inc >= 5) & (inc <= 8)}}
GV, GT = groups(X_va), groups(X_te)
PAIR = {"age": ("60+", "18-39"), "inc": ("low", "high")}
rec = lambda y, f: f[y == 1].mean()

def op_block(s_va, s_te):
    kind = min(("platt", "isotonic"), key=lambda c: cv_brier(c, s_va, yva))
    cal = fit_calibrator(kind, s_va, yva); p_va, p_te = cal(s_va), cal(s_te)
    thr = float(best_threshold(yva, p_va)); rstar = float((p_va[yva == 1] >= thr).mean())
    flag = p_te >= thr; f1g = f1_score(yte, flag)
    out = dict(AUC=roc_auc_score(yte, s_te), F1_global=f1g,
               max_subgroup_ECE=max(ece_eqmass(yte[m], p_te[m]) for gr in GT.values() for m in gr.values()))
    for gr in ("age", "inc"):
        a, b = PAIR[gr]
        out[f"recall_gap_{gr}"] = rec(yte[GT[gr][a]], flag[GT[gr][a]]) - rec(yte[GT[gr][b]], flag[GT[gr][b]])
        f_eq = np.zeros(len(yte), bool)
        for g, mt in GT[gr].items():
            t = float(np.quantile(p_va[GV[gr][g] & (yva == 1)], 1 - rstar))
            f_eq[mt] = p_te[mt] >= t
        out[f"dF1_equal_{gr}"] = f1_score(yte, f_eq) - f1g
        out[f"recall_gap_equal_{gr}"] = rec(yte[GT[gr][a]], f_eq[GT[gr][a]]) - rec(yte[GT[gr][b]], f_eq[GT[gr][b]])
        out[f"flagged_pts_equal_{gr}"] = 100 * (f_eq.mean() - flag.mean())
    return out

op = pd.DataFrame({tag: op_block(*pred[("XGB", tag)]) for tag in ("default", "tuned")}).T.round(4)

search.to_csv("figures/v2_tuning_search.csv", index=False)
test.to_csv("figures/v2_tuning_test_auc.csv", index=False)
op.to_csv("figures/v2_tuning_xgb_operating_point.csv")
print("\n=== SEARCH (validation) ===\n", search.to_string(index=False))
print("\n=== TEST AUC, DEFAULT VS TUNED ===\n", test.to_string(index=False))
print("\n=== XGBOOST OPERATING POINT, DEFAULT VS TUNED ===\n", op.to_string())