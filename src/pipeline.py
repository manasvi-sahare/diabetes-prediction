import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, brier_score_loss, f1_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

SEED = 42
df = pd.read_csv("data/diabetes.csv")
n_raw = len(df)
df = df.drop_duplicates().reset_index(drop=True)
print(f"Dropped {n_raw - len(df):,} exact duplicate rows; {len(df):,} remain")
X, y = df.drop(columns="Diabetes_binary"), df["Diabetes_binary"].values

# 60/20/20 stratified split
X_tmp, X_te, y_tmp, y_te = train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEED)
X_tr, X_va, y_tr, y_va = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)
spw = (y_tr == 0).sum() / (y_tr == 1).sum()

def models(weighted):
    return {
        "LR": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced" if weighted else None)),
        "RF": RandomForestClassifier(n_estimators=150, max_depth=12, min_samples_leaf=20, max_features="sqrt",
                                     class_weight="balanced" if weighted else None, n_jobs=-1, random_state=SEED),
        "XGB": XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, eval_metric="logloss",
                             scale_pos_weight=spw if weighted else 1, n_jobs=-1, random_state=SEED),
        "LGBM": LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.1,
                               scale_pos_weight=spw if weighted else 1, n_jobs=-1, random_state=SEED, verbose=-1),
    }

# ---------- helpers ----------
def logit(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

def ece_eqmass(y, p, bins=10):
    idx = np.argsort(p); n = len(y)
    return sum(len(b) / n * abs(y[b].mean() - p[b].mean()) for b in np.array_split(idx, bins))

def cal_slope_intercept(y, p):
    lr = LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return lr.coef_[0][0], lr.intercept_[0]

def fit_calibrator(kind, s, y):
    if kind == "platt":
        m = LogisticRegression(C=1e6, max_iter=1000).fit(logit(s).reshape(-1, 1), y)
        return lambda z: m.predict_proba(logit(z).reshape(-1, 1))[:, 1]
    m = IsotonicRegression(out_of_bounds="clip").fit(s, y)
    return m.predict

def cv_brier(kind, s, y):  # honest comparison of calibrators on validation
    out = []
    for a, b in StratifiedKFold(5, shuffle=True, random_state=SEED).split(s, y):
        out.append(brier_score_loss(y[b], fit_calibrator(kind, s[a], y[a])(s[b])))
    return np.mean(out)

def best_threshold(y, p):
    grid = np.arange(0.05, 0.60, 0.005)
    return grid[np.argmax([f1_score(y, p >= t) for t in grid])]

# ---------- run ----------
if __name__ == "__main__":
    rows = []
    for weighted in (False, True):
        for name, mdl in models(weighted).items():
            mdl.fit(X_tr, y_tr)
            s_va, s_te = mdl.predict_proba(X_va)[:, 1], mdl.predict_proba(X_te)[:, 1]

            # choose calibrator on validation only
            kind = min(("platt", "isotonic"), key=lambda k: cv_brier(k, s_va, y_va))
            cal = fit_calibrator(kind, s_va, y_va)
            p_va, p_te = cal(s_va), cal(s_te)
            thr = best_threshold(y_va, p_va)          # threshold chosen on validation only

            for stage, p in (("raw", s_te), ("calibrated", p_te)):
                slope, icpt = cal_slope_intercept(y_te, p)
                rows.append(dict(model=name, weighted=weighted, stage=stage, calibrator=kind if stage == "calibrated" else "-",
                                AUC=roc_auc_score(y_te, p), Brier=brier_score_loss(y_te, p),
                                ECE=ece_eqmass(y_te, p), cal_slope=slope, cal_intercept=icpt,
                                thr=thr if stage == "calibrated" else 0.5,
                                F1=f1_score(y_te, p >= (thr if stage == "calibrated" else 0.5))))
            print(name, "weighted" if weighted else "plain", "done")

    res = pd.DataFrame(rows).round(4)
    res.to_csv("figures/clean_pipeline_results.csv", index=False)
    print(res.to_string(index=False))