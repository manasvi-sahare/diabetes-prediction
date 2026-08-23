# Explainable & Calibrated ML for Diabetes Risk Prediction

A reproducible machine learning pipeline for diabetes risk classification on the CDC Diabetes Health
Indicators dataset (BRFSS 2015), comparing Logistic Regression, Random Forest, and XGBoost, with a
focus on **probability calibration** and **SHAP explainability** — two aspects most diabetes-prediction
studies skip.

## Key Real Findings

| Model | ROC-AUC | F1 | Brier Score | Notes |
|---|---|---|---|---|
| Logistic Regression | 0.8196 | 0.4413 | 0.1776 | Interpretable baseline |
| Random Forest | 0.8238 | 0.4415 | 0.1738 | Regularized: `max_depth=12`, `min_samples_leaf=20` |
| XGBoost | 0.8236 | 0.4422 | 0.1714 | `scale_pos_weight=6.177` for imbalance |
| XGBoost (calibrated, isotonic) | 0.8232 | 0.2162 @ thresh=0.5 | **0.0981** | Best of sigmoid vs. isotonic (near-tied) |
| XGBoost (calibrated, tuned threshold=0.245) | 0.8232 | **0.4652** | 0.0981 | Best F1 of any configuration |

*Confirmed reproducible: identical to 4 decimal places across two independent full pipeline runs.*

**Headline findings:**
- All three models converge tightly on ROC-AUC (~0.82) — **algorithm choice matters far less than
  threshold tuning and calibration** on this dataset.
- Raw XGBoost (trained with imbalance correction) is **badly over-confident**: at a predicted
  probability of ~0.62, the true observed rate is only ~0.21. Platt scaling cuts the Brier Score by
  43% and brings predicted/observed probabilities into close agreement.
- Real SHAP analysis identifies **General Health, High Blood Pressure, Age, BMI, and High Cholesterol**
  as the top 5 predictors, consistent with prior literature (Rafie et al. 2025; Kutlu et al. 2024).

### SHAP Feature Importance

![SHAP summary plot showing feature impact on diabetes risk](figures/shap_summary.png)

### Calibration: Raw vs. Sigmoid vs. Isotonic

![Calibration comparison showing reliability curves before and after recalibration](figures/calibration_comparison.png)

## Dataset

**CDC Diabetes Health Indicators** (derived from the 2015 BRFSS, cleaned version)
- 253,680 records, 21 features, binary target (`Diabetes_binary`)
- Class imbalance: 6.18:1 (13.93% positive rate)
- Zero missing values
- Source: [UCI ML Repository, DOI: 10.24432/C53919](https://doi.org/10.24432/C53919)

```powershell
# Download (one-time manual step — not a persisted script)
curl -L -o data/diabetes.csv "https://raw.githubusercontent.com/Helmy2/Diabetes-Health-Indicators/main/diabetes_binary_health_indicators_BRFSS2015.csv"
```

## Project Structure

```
diabetes-prediction/
├── data/
│   └── diabetes.csv                    # raw dataset (not committed — see .gitignore)
├── models/                             # trained model artifacts (not committed — see .gitignore)
│   ├── split.pkl                       # train/test split + scaler
│   ├── logreg.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── xgboost_calibrated.pkl
├── figures/                            # generated plots and result tables
├── requirements.txt
├── requirements-exact.txt              # pinned versions from `pip freeze`
├── 02_inspect_data.py                  # data shape, missingness, target distribution
├── 03_feature_types.py                 # classify features (binary / ordinal / continuous)
├── 04_split_and_scale.py               # stratified train/test split + StandardScaler
├── 05_train_logreg.py                  # Logistic Regression baseline
├── 06_train_rf.py                      # Random Forest (regularized)
├── 07_train_xgb.py                     # XGBoost with imbalance correction
├── 08_compare_models.py                # side-by-side metric comparison
├── 09_calibrate.py                     # Brier Score + Platt scaling
├── 10_threshold_tuning.py              # precision/recall/F1 vs. decision threshold
├── 11_shap_explainability.py           # SHAP analysis (global importance + summary plot)
├── 12_shap_local_example.py            # per-patient SHAP waterfall explanation
└── 13_final_summary.py                 # consolidated results table across all models
```


## Setup

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproducing the Results

Run the scripts in numeric order:

```powershell
python 02_inspect_data.py
python 03_feature_types.py
python 04_split_and_scale.py
python 05_train_logreg.py
python 06_train_rf.py
python 07_train_xgb.py
python 08_compare_models.py
python 09_calibrate.py
python 10_threshold_tuning.py
python 11_shap_explainability.py
python 12_shap_local_example.py
python 13_final_summary.py
```

Each script saves its trained model/artifacts to `models/` and any figures/tables to `figures/`, so
later scripts can reuse earlier outputs without recomputing from scratch.

## Methodology Summary

1. **Preprocessing** — stratified 80/20 train/test split (preserves 13.93% positive rate in both sets);
   `StandardScaler` fit on training data only (used for Logistic Regression).
2. **Models** — Logistic Regression (`class_weight="balanced"`), Random Forest
   (`max_depth=12, min_samples_leaf=20, class_weight="balanced"`), XGBoost
   (`scale_pos_weight` set to the true class ratio).
3. **Evaluation** — Precision, Recall, F1, ROC-AUC, PR-AUC, confusion matrix; stratified 5-fold CV
   used to confirm stability (XGBoost: 0.8282 ± 0.0022 across folds).
4. **Calibration** — Brier Score compared across **both sigmoid (Platt) and isotonic calibration**
   (`CalibratedClassifierCV`), with the better-performing method (isotonic, in this run — 0.0981 vs.
   0.0981 sigmoid, effectively tied) saved as the final calibrated model; reliability diagrams via
   `sklearn.calibration.calibration_curve`.
5. **Threshold tuning** — precision/recall/F1 swept across all thresholds on the calibrated model's
   output to find the F1-optimal cutoff (0.245) and a high-recall operating point (0.160, ~75% recall).
6. **Explainability** — SHAP `TreeExplainer` on XGBoost (3,000-case test sample), both global (summary
   plot, mean |SHAP|) and local (single-patient waterfall) explanations.

## Known Issues / Open Items

- [x] ~~Consolidate `09_shap_explain.py` and `12_shap_explainability.py`~~ — resolved: removed the
  superseded draft (`09_shap_explain.py`, which pointed at a non-existent `outputs/` folder and used
  an older SHAP API); renumbered remaining scripts to close the gap.
- [x] ~~Full pipeline re-run after renumbering~~ — confirmed working end-to-end, all 12 scripts (`02`
  through `13`) run cleanly in sequence with no import/path errors.
- [ ] Consolidate output folders: `08`, `09`, `10` currently save to `outputs/` while `11`, `12`, `13`
  save to `figures/` — merge into a single `figures/` folder for consistency.
- [ ] Add a dedicated `01_download_data.py` for full one-command reproducibility
- [ ] LightGBM was discussed but not yet trained/evaluated
- [ ] Consider a subgroup calibration check (e.g. by Sex or Age) for a fairness angle

## License / Data Attribution

Dataset: CDC Diabetes Health Indicators, UCI Machine Learning Repository, DOI: 10.24432/C53919.
Code in this repository is original work for a research proposal on explainable, calibrated ML for
diabetes risk prediction.