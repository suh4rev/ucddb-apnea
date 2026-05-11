# 1D-CNN DL Improvement Analysis

## Goal

Improve the DL part of the project without retraining or leakage by using saved subject-level out-of-fold predictions from:

- `reports/tables/cnn_cv_predictions.csv`
- `reports/tables/cnn_improved_cv_predictions.csv`

The analysis tests fixed, label-free probability post-processing:

- causal rolling mean inside each `record_id`;
- equal-weight probability averaging between simple CNN and ResNet1D components.

No labels are used for smoothing or probability averaging. Labels are used only for reporting metrics and threshold selection.

## Leakage Controls

- Predictions are out-of-fold under subject-level validation.
- Smoothing is applied only within the same `record_id`.
- The improvement variants here use causal smoothing only.
- The underlying CNN context models are still offline retrospective where their input uses future neighboring epochs.
- The longest causal smoothing window is 121 epochs, about 60.5 minutes. It should be interpreted as probability stabilization over a long PSG interval, not as a local event detector.
- This is an exploratory DL post-processing comparison on the same out-of-fold predictions, so the selected best variant can be mildly optimistic and should be reported as an internal validation result.

## Results At Threshold 0.5

| experiment | components | postprocessing | smoothing_window_epochs | roc_auc_mean | f1_mean | sensitivity_mean | specificity_mean | average_precision_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| cnn30_resnet_causal_121_mean | cnn_30s + resnet_150s | equal_mean_causal | 121 | 0.6213 | 0.3440 | 0.4581 | 0.7269 | 0.3563 |
| cnn90_resnet_causal_121_mean | cnn_90s_context + resnet_150s | equal_mean_causal | 121 | 0.6152 | 0.3791 | 0.4533 | 0.7392 | 0.3456 |
| resnet_causal_121 | resnet_150s | rolling_mean_causal | 121 | 0.6141 | 0.4018 | 0.4898 | 0.7236 | 0.3456 |
| cnn90_resnet_causal_31_mean | cnn_90s_context + resnet_150s | equal_mean_causal | 31 | 0.6061 | 0.3767 | 0.4765 | 0.7061 | 0.3154 |
| resnet_causal_61 | resnet_150s | rolling_mean_causal | 61 | 0.6022 | 0.3743 | 0.4856 | 0.6865 | 0.3094 |

## Threshold Sweep

| experiment | selection_rule | threshold | f1 | sensitivity | specificity | precision |
| --- | --- | --- | --- | --- | --- | --- |
| resnet_causal_61 | max_f1 | 0.4500 | 0.3837 | 0.5979 | 0.5789 | 0.2824 |
| resnet_causal_61 | max_youden | 0.4500 | 0.3837 | 0.5979 | 0.5789 | 0.2824 |
| resnet_causal_61 | sensitivity_ge_0_70_max_specificity | 0.3500 | 0.3704 | 0.7577 | 0.3533 | 0.2451 |
| resnet_causal_121 | max_f1 | 0.5000 | 0.3929 | 0.4875 | 0.7244 | 0.3290 |
| resnet_causal_121 | max_youden | 0.5000 | 0.3929 | 0.4875 | 0.7244 | 0.3290 |
| resnet_causal_121 | sensitivity_ge_0_70_max_specificity | 0.3500 | 0.3779 | 0.7852 | 0.3430 | 0.2488 |
| cnn90_resnet_causal_31_mean | max_f1 | 0.4500 | 0.3934 | 0.5942 | 0.6046 | 0.2941 |
| cnn90_resnet_causal_31_mean | max_youden | 0.4500 | 0.3934 | 0.5942 | 0.6046 | 0.2941 |
| cnn90_resnet_causal_31_mean | sensitivity_ge_0_70_max_specificity | 0.3500 | 0.3677 | 0.7746 | 0.3241 | 0.2411 |
| cnn90_resnet_causal_121_mean | max_f1 | 0.4500 | 0.3972 | 0.5687 | 0.6410 | 0.3051 |
| cnn90_resnet_causal_121_mean | max_youden | 0.4500 | 0.3972 | 0.5687 | 0.6410 | 0.3051 |
| cnn90_resnet_causal_121_mean | sensitivity_ge_0_70_max_specificity | 0.3500 | 0.3674 | 0.7843 | 0.3111 | 0.2399 |
| cnn30_resnet_causal_121_mean | max_f1 | 0.4500 | 0.3913 | 0.6051 | 0.5876 | 0.2891 |
| cnn30_resnet_causal_121_mean | max_youden | 0.4500 | 0.3913 | 0.6051 | 0.5876 | 0.2891 |
| cnn30_resnet_causal_121_mean | sensitivity_ge_0_70_max_specificity | 0.4000 | 0.3845 | 0.7379 | 0.4179 | 0.2600 |

## Comparison

Previous simple CNN: ROC-AUC=0.5953, F1=0.3453.

Previous ResNet1D: ROC-AUC=0.5998, F1=0.3756.

Temporal ensemble: ROC-AUC=0.7066, F1=0.4349.

Best DL-improvement ROC-AUC: 0.6213 (`cnn30_resnet_causal_121_mean`).

Best DL-improvement F1 at threshold 0.5: 0.4018 (`resnet_causal_121`).

Best tuned DL-improvement F1: 0.3972 (`cnn90_resnet_causal_121_mean`, threshold=0.45).

## Interpretation

The DL improvement layer improves the DL-only metrics compared with the simple CNN and the first ResNet1D run, mainly by stabilizing noisy epoch-level probabilities. It still does not beat the classical temporal ensemble. The long smoothing window means the best DL AUC should be described as a retrospective/post-processing result, not as real-time event detection. This supports the thesis conclusion that, on small UCDDB subject-level validation, simple deep models on raw signals need either stronger architectures, more data, or external pretraining to outperform carefully engineered temporal ML features.
