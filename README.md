# Calibrated and Explainable ML for Diabetes Risk Prediction

Leakage-controlled comparison of logistic regression, random forest, XGBoost and
LightGBM on the CDC Diabetes Health Indicators data (BRFSS, as distributed by UCI),
with calibration, bootstrap significance testing, subgroup analysis, operating-point
analysis and SHAP stability.

**Target:** `Diabetes_binary` (1 = self-reported diagnosed diabetes; 0 = otherwise).
In the file used here, respondents with prediabetes (4,631) are coded 0, verified by
cross-tabulation against the three-class version of the data. The UCI documentation
describes the positive class as including prediabetes, which does not match this file.

## Protocol
- 24,206 exact duplicate rows removed before splitting (229,474 records remain)
- Stratified 60/20/20 train / validation / test split (seed 42)
- Calibrator (Platt or isotonic) and decision threshold chosen on validation data only
- Test set used for reporting only
- Subgroup analysis by sex, age (18-39, 40-59, 60+) and income (levels 1-4, 5-8). This is a
  performance and calibration comparison, not a formal fairness audit: no fairness
  criterion such as demographic parity or equalised odds is tested.

## Main findings (test set, unweighted models)
- Ensembles beat logistic regression by about 0.005-0.006 AUC (Bonferroni-adjusted paired
  bootstrap): reliable but small. Ensembles are practically equivalent to each other
  (differences of at most 0.0014 AUC).
- Unweighted models are already well calibrated (ECE 0.004-0.015). Class weighting
  inflates predicted risk (calibration intercept about -1.6 to -1.7, ECE about 0.23) and
  gives no gain in AUC, Brier score or tuned F1 after recalibration.
- Calibration holds across subgroups (ECE <= 0.008), but discrimination does not:
  AUC is 0.062 lower for ages 60+ than 18-39, 0.026 lower for men than women and
  0.040 lower for lower-income respondents.
- One population-wide threshold gives very different recall by group (age gap 0.47,
  income gap 0.19). Group-specific F1 thresholds narrow the gaps without a detectable
  loss of F1 but do not close them. Equal-recall thresholds close them, at a cost of
  about 0.022 F1 for age and 0.008 for income (paired bootstrap).
- Respondents with prediabetes (coded negative) receive intermediate risk scores, but
  excluding them leaves the subgroup AUC gaps essentially unchanged.
- SHAP rankings are stable under retraining (mean Spearman rho >= 0.967), and the same
  five features lead every model: GenHlth, HighBP, Age, BMI, HighChol.

## Run (from the project root, in this order)
```
python src/pipeline.py            # models, calibration, thresholds
python src/significance.py        # bootstrap CIs, pairwise tests, 5-fold CV
python src/subgroups.py           # subgroup metrics, contrasts, group-wise recalibration
python src/shap_stability.py      # SHAP stability (slow; cached in cache/)
python src/operating_points.py    # threshold strategies, decision curves
python src/strategy_tests.py      # paired tests between threshold strategies
python src/figures.py             # figures
python src/data_checks.py         # duplicate and leakage diagnostics
python src/prediabetes.py         # prediabetes label breakdown (needs data/diabetes_012.csv)
```

Download the data first:
```
curl -L -o data/diabetes.csv "https://raw.githubusercontent.com/Helmy2/Diabetes-Health-Indicators/main/diabetes_binary_health_indicators_BRFSS2015.csv"
```

The prediabetes analysis also needs the three-class file. Download it with
`python src/get_012.py` (requires `pip install kagglehub`), which saves
`data/diabetes_012.csv`, then check row alignment with `python src/check_012.py`.

## Layout
- `src/` current analysis scripts
- `figures/` current results (`v2_*`)
- `figures/old/` superseded outputs from earlier versions of the pipeline, kept for reference
  (including `full_data_weighted/`, the class-weighted run on the deduplicated data)
- `archive/` earlier exploratory scripts, kept for reference only
- `diabetes_prediction_pipeline.ipynb` Colab notebook from an earlier version of the pipeline (not yet rebuilt to match `src/`; do not use it to reproduce the current results)

## Status and limitations
Work in progress. Results are from one survey, with a self-reported diabetes label
that codes prediabetes as negative, ignoring survey weights, and have not been
externally validated. Cross-environment reproducibility has not been re-checked since
the pipeline was rewritten.

## License / Data Attribution
Dataset: CDC Diabetes Health Indicators, UCI Machine Learning Repository,
DOI: 10.24432/C53919.