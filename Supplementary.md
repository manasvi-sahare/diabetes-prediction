# Supplementary material

Supports the paper "Calibration Is Not Enough: Operating-Point Disparities in Machine Learning Models for Diabetes Screening". All numbers come from the scripts in `src/` and the CSVs in `figures/` (`v2_*`).

## 1. Data handling
- Source file: binary-label file circulated on Kaggle/GitHub (253,680 rows); features identical row by row to the Kaggle three-class file (`src/check_012.py`).
- 24,206 exact duplicate rows removed before splitting (229,474 remain; 15.3% positive).
- Primary split: stratified 60/20/20, seed 42. Repeated splits: seeds 101-110 (`src/repeated_splits.py`).

## 2. Calibration and threshold procedure
1. Fit the model on the training set.
2. Compute scores on the validation set. For Platt (logistic regression on the logit of the score) and isotonic calibration, estimate the Brier score by 5-fold cross-validation within the validation set (seed 42).
3. Keep the method with the lower cross-validated Brier score and refit it on all validation scores.
4. Apply the calibrator to validation and test scores.
5. Global threshold: the value on the grid 0.05, 0.055, ..., 0.595 (step 0.005) that maximises F1 on calibrated validation predictions.
6. Group-specific F1 thresholds: same grid and criterion, within each group's validation rows.
7. Equal-recall thresholds: for each group, the (1 - r) quantile (linear interpolation) of calibrated validation scores of that group's positive cases, where r is the validation recall of the global rule (0.623). The quantile is not restricted to the grid. Test recall can differ from r because each threshold is estimated from a finite number of validation positives.
8. The test set is used only to report results.

For the tuned XGBoost model, steps 2-7 were repeated with the tuned model (calibrator and global threshold reselected on validation data; equal-recall thresholds recomputed).

## 3. Bootstrap procedures
- Model AUC comparisons (`src/significance.py`, `src/tuned_tests.py`): paired percentile bootstrap on the test set, 10,000 resamples, same resample for every model; intervals Bonferroni-adjusted (7 comparisons for default models, including a trivial calibrated-versus-raw XGBoost check; 6 for tuned models). Practical-equivalence margin: +/-0.005 AUC.
- Subgroup metrics (`src/subgroups.py`): 500 paired percentile resamples, 95% intervals.
- Threshold strategies (`src/strategy_tests.py`): 1,000 paired percentile resamples with thresholds fixed, 95% intervals.
- Age-gap and prediabetes analyses: 500 resamples.

## 4. Hyperparameter search (`src/tuning.py`)
Procedure: the default configuration plus 8 random configurations per tree model (sampled uniformly from the lists below, seed 42), fitted on the training set and compared by validation AUC; the best configuration was refitted and evaluated once on the test set. Logistic regression: C in {0.01, 0.1, 1, 10} (all four evaluated; validation AUC identical at four decimals).

| Model | Search space |
|---|---|
| Random forest | n_estimators {150, 300}; max_depth {8, 12, 16, 20}; min_samples_leaf {5, 10, 20, 40}; max_features {sqrt, 0.3, 0.5} |
| XGBoost | n_estimators {200, 400, 600}; max_depth {3, 4, 6, 8}; learning_rate {0.03, 0.05, 0.1}; subsample {0.7, 0.85, 1.0}; colsample_bytree {0.7, 0.85, 1.0}; min_child_weight {1, 5, 10}; reg_lambda {1, 5, 10} |
| LightGBM | n_estimators {200, 400, 600}; max_depth {4, 6, 8}; num_leaves {15, 31, 63}; learning_rate {0.03, 0.05, 0.1}; subsample {0.7, 0.85, 1.0} (with subsample_freq = 1); colsample_bytree {0.7, 0.85, 1.0}; min_child_samples {20, 50, 100}; reg_lambda {1, 5, 10} |

| Model | Selected configuration | Validation AUC, default to selected | Test AUC, default to selected |
|---|---|---|---|
| Logistic regression | C = 0.01 (all values equal at four decimals) | 0.8088 to 0.8088 | 0.8103 to 0.8103 |
| Random forest | 300 trees, max_depth 20, min_samples_leaf 20, max_features sqrt | 0.8134 to 0.8139 | 0.8163 to 0.8165 |
| XGBoost | 600 trees, max_depth 4, learning_rate 0.03, subsample 1.0, colsample_bytree 0.85, min_child_weight 10, reg_lambda 5 | 0.8135 to 0.8172 | 0.8151 to 0.8198 |
| LightGBM | 400 trees, max_depth 4, num_leaves 31, learning_rate 0.05, subsample 0.7, colsample_bytree 1.0, min_child_samples 50, reg_lambda 10 | 0.8149 to 0.8174 | 0.8165 to 0.8195 |

The search is small, and the calibrator is fitted on the same validation set that is used to select hyperparameters.

## 5. SHAP stability settings
Five CV folds and two disjoint training halves; the same 500 held-out validation rows are explained for every model and fold; random forest uses 50 trees for this analysis only; cross-model comparison uses rankings and normalised shares because SHAP units differ by model (`src/shap_stability.py`).

## 6. Software versions
Fill in by running:

```
python -c "import sys,sklearn,xgboost,lightgbm,shap,numpy,pandas,scipy; print(sys.version.split()[0], 'sklearn',sklearn.__version__, 'xgboost',xgboost.__version__, 'lightgbm',lightgbm.__version__, 'shap',shap.__version__, 'numpy',numpy.__version__, 'pandas',pandas.__version__, 'scipy',scipy.__version__)"
```

Versions: (paste output here)

## 7. Additional tables not shown in the paper
Age-specific models, single-feature separation by age group, learning curves, prediabetes breakdown and tuning comparisons are in `figures/v2_age_*.csv`, `figures/v2_pre_*.csv`, `figures/v2_tuning_*.csv`.