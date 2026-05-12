# Hyperparameter Tuning Notes

This project uses bounded, reproducible tuning rather than a large automated
search, because UCDDB has only 25 subjects and the validation split must remain
subject-level.

## Validation Strategy

- Validation uses 5-fold subject-level cross-validation grouped by `record_id`.
- `StratifiedGroupKFold` is used when available; otherwise the scripts fall back
  to `GroupKFold`.
- Model selection is based on out-of-fold validation results only.
- The main ranking metric is ROC-AUC. F1, sensitivity, specificity, average
  precision, and confusion matrices are kept for practical interpretation.

## Search Spaces Used In The Repository

The main ML pipeline evaluates feature-space and model variants:

- regimes: all epochs and sleep-only epochs;
- feature variants: base features and enhanced record-normalized/contextual
  features;
- modality groups: Flow, SpO2, ECG, respiratory effort, respiratory+SpO2
  fusion, late fusion;
- final temporal ensemble: SpO2 XGBoost plus Flow+SpO2 HistGradientBoosting.

Advanced XGBoost experiments use a fixed grid over:

| Parameter | Values |
| --- | --- |
| `n_estimators` | 300, 500 |
| `learning_rate` | 0.02, 0.03, 0.05 |
| `max_depth` | 2, 3, 4 |
| `min_child_weight` | 1, 3, 5 |
| `reg_lambda` | 1, 5, 10 |
| `subsample` | 0.8, 0.9 |
| `colsample_bytree` | 0.8, 0.9 |

The final temporal post-processing search uses:

- raw probabilities;
- rolling mean smoothing windows: 15, 31, and 61 epochs;
- causal smoothing and centered offline retrospective smoothing;
- threshold grid from 0.05 to 0.95 with step 0.05.

## Stopping Criteria

- Grid searches stop after all predefined configurations are evaluated.
- Threshold and smoothing searches stop after all thresholds/windows are
  evaluated on saved out-of-fold probabilities.
- Deep learning experiments use a maximum of 40 epochs and early stopping
  patience of 6 epochs by validation ROC-AUC.
- Optional Optuna experiments should use the same grouped CV protocol, a fixed
  trial budget, and the same metric priorities to avoid patient-level leakage.

## Main Tuning Artifacts

- `reports/tables/improved_best_results.csv`
- `reports/tables/advanced_best_results.csv`
- `reports/tables/temporal_ensemble_results_summary.csv`
- `reports/tables/temporal_ensemble_best_thresholds.csv`
- `reports/temporal_ensemble_report.md`
