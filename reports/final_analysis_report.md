# Final Analysis Report

This report summarizes improved-model results without retraining.

## Best Models

- Best by AUC-ROC: `sleep_only / enhanced / spo2_only` with mean ROC-AUC 0.673.
- Best by F1: `sleep_only / enhanced / spo2_only` with mean F1 0.415.

## Baseline vs Enhanced

For all-epoch late fusion, enhanced features changed ROC-AUC by +0.048 and F1 by +0.032 compared with base features.

## Modality Findings

Among enhanced all-epoch experiments, the strongest ROC-AUC was `late_fusion` with 0.669.

## Fusion Findings

The strongest fusion result was `all_epochs / enhanced / late_fusion` with ROC-AUC 0.669.

## Thresholding

A threshold of 0.5 is not necessarily optimal under class imbalance. The threshold sweep tables include alternatives that maximize F1, Youden index, or specificity under sensitivity >= 0.70.

## Limitations

The analysis uses 25 UCDDB records with subject-level cross-validation. Fold variability is moderate: the maximum ROC-AUC fold std across experiments is 0.095. Results should therefore be reported as comparative evidence rather than final clinical performance.

## Final Table

| regime | feature_variant | experiment | n_features | roc_auc_mean | roc_auc_std | f1_mean | sensitivity_mean | specificity_mean | average_precision_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all_epochs | base | flow_only | 10 | 0.601 | 0.043 | 0.334 | 0.477 | 0.663 | 0.274 |
| all_epochs | base | respiratory_spo2_fusion | 41 | 0.620 | 0.050 | 0.342 | 0.501 | 0.662 | 0.305 |
| all_epochs | base | late_fusion | 53 | 0.621 | 0.051 | 0.322 | 0.399 | 0.754 | 0.303 |
| all_epochs | enhanced | flow_only | 30 | 0.633 | 0.056 | 0.342 | 0.490 | 0.673 | 0.302 |
| all_epochs | enhanced | spo2_only | 38 | 0.664 | 0.086 | 0.380 | 0.506 | 0.721 | 0.348 |
| all_epochs | enhanced | respiratory_spo2_fusion | 127 | 0.666 | 0.074 | 0.398 | 0.549 | 0.701 | 0.351 |
| all_epochs | enhanced | late_fusion | 151 | 0.669 | 0.070 | 0.354 | 0.435 | 0.769 | 0.361 |
| sleep_only | enhanced | spo2_only | 38 | 0.673 | 0.057 | 0.415 | 0.526 | 0.718 | 0.372 |
| sleep_only | enhanced | respiratory_spo2_fusion | 127 | 0.645 | 0.078 | 0.384 | 0.486 | 0.716 | 0.352 |
| sleep_only | enhanced | late_fusion | 151 | 0.655 | 0.078 | 0.385 | 0.431 | 0.779 | 0.363 |
