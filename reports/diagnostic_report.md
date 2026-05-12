# Diagnostic Report

This report analyzes already trained baseline models without retraining.

## Best Experiments

- Best by ROC-AUC: `late_fusion` / `XGBoost` with mean ROC-AUC 0.6210.
- Best by F1: `full_minus_ecg` / `XGBoost` with mean F1 0.3419.

## Fold Variability

There is moderate variability between folds (max ROC-AUC std=0.084, max F1 std=0.104).

## Sleep Stages

Observed sleep_stage values: 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, nan.
A sleep_only diagnostic baseline is worth considering because positive rates vary by sleep_stage with spread=0.301. It should be used only as analysis, not as a primary deployment model.

## Thresholding

A fixed threshold of 0.5 may be suboptimal because the positive class is imbalanced and the project may prefer higher sensitivity for screening. The threshold sweep tables show alternatives that maximize F1, Youden index, or sensitivity-constrained specificity.

## Top Experiments

| experiment | model | roc_auc_mean | f1_mean | sensitivity_mean | specificity_mean | average_precision_mean |
| --- | --- | --- | --- | --- | --- | --- |
| late_fusion | XGBoost | 0.6210 | 0.3222 | 0.3991 | 0.7539 | 0.3033 |
| full_minus_ecg | XGBoost | 0.6201 | 0.3419 | 0.5014 | 0.6618 | 0.3046 |
| respiratory_spo2_fusion | XGBoost | 0.6201 | 0.3419 | 0.5014 | 0.6618 | 0.3046 |
| respiratory_fusion | XGBoost | 0.6119 | 0.3266 | 0.4836 | 0.6663 | 0.2802 |
| full_core_fusion | LogisticRegression | 0.6059 | 0.3277 | 0.5015 | 0.6270 | 0.2986 |
| flow_only | XGBoost | 0.6007 | 0.3340 | 0.4767 | 0.6632 | 0.2737 |
| full_minus_effort | XGBoost | 0.5977 | 0.3282 | 0.4273 | 0.7104 | 0.2855 |
| full_core_fusion | XGBoost | 0.5966 | 0.3119 | 0.4071 | 0.7199 | 0.2776 |
| full_minus_flow | XGBoost | 0.5864 | 0.3124 | 0.3964 | 0.7252 | 0.2778 |
| full_minus_spo2 | XGBoost | 0.5837 | 0.2669 | 0.3485 | 0.7289 | 0.2439 |

## Output Tables

- `reports\tables\fold_label_distribution.csv`
- `reports\tables\stage_label_distribution.csv`
- `reports\tables\threshold_sweep.csv`
- `reports\tables\best_thresholds_by_experiment.csv`
- `reports\tables\top_experiments_for_report.csv`
