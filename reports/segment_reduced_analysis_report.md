# Reduced Segment-Level Experiment

## Summary

Best ROC-AUC: exclude_ucddb005_sleep_only / window_10s / flow_spo2 / XGBoost / xgb_shallow (ROC-AUC=0.6051, F1=0.3057, sensitivity=0.4717, specificity=0.6962).

Best F1 at threshold 0.5: all_records_sleep_only / window_60s / flow_spo2 / XGBoost / xgb_shallow (ROC-AUC=0.5939, F1=0.3995, sensitivity=0.4757, specificity=0.6714).

Best tuned F1: exclude_ucddb005_sleep_only / window_60s / respiratory_spo2 / XGBoost / xgb_shallow @ threshold=0.2000 (F1=0.4342, sensitivity=0.9838, specificity=0.0257).

Comparison with previous references:

- Previous best AUC: 0.6725; reduced best delta: -0.0674
- Previous best F1: 0.4152; reduced best delta: -0.0157
- Previous tuned F1: 0.4311; reduced tuned delta: 0.0031

Segment-level representation did not improve the best ROC-AUC over the previous reference in this reduced honest run.

The best tuned F1 is only marginally above the previous tuned F1, and it is achieved at very low specificity. This should not be interpreted as a clinically useful improvement.

AUC 0.8 was not reached. This is plausible under subject-level CV on UCDDB: the number of subjects is small, inter-subject physiology varies, tabular segment features still discard raw waveform morphology, and apnea/hypopnea event labels are noisy around onset and recovery. A sequence model or raw-signal deep learning approach is a more realistic route to a large jump.

## Top Results

| regime | window_set | feature_set | model | config_name | roc_auc_mean | f1_mean | sensitivity_mean | specificity_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| exclude_ucddb005_sleep_only | window_10s | flow_spo2 | XGBoost | xgb_shallow | 0.6051 | 0.3057 | 0.4717 | 0.6962 |
| exclude_ucddb005_sleep_only | window_10s | flow_spo2 | XGBoost | xgb_regularized | 0.6002 | 0.2984 | 0.4508 | 0.7061 |
| exclude_ucddb005_sleep_only | window_10s | respiratory_spo2 | XGBoost | xgb_shallow | 0.5967 | 0.2922 | 0.4816 | 0.6578 |
| all_records_sleep_only | window_60s | flow_spo2 | XGBoost | xgb_shallow | 0.5939 | 0.3995 | 0.4757 | 0.6714 |
| all_records_sleep_only | window_60s | respiratory_spo2 | XGBoost | xgb_shallow | 0.5936 | 0.3935 | 0.4604 | 0.6861 |
| exclude_ucddb005_sleep_only | window_10s | respiratory_spo2 | XGBoost | xgb_regularized | 0.5911 | 0.2851 | 0.4453 | 0.6882 |
| all_records_sleep_only | window_60s | respiratory_spo2 | XGBoost | xgb_regularized | 0.5900 | 0.3807 | 0.4294 | 0.7085 |
| exclude_ucddb005_sleep_only | window_10s | respiratory_spo2 | ExtraTrees | extra_trees | 0.5900 | 0.0140 | 0.0075 | 0.9946 |
| exclude_ucddb005_sleep_only | window_10s | flow_spo2 | ExtraTrees | extra_trees | 0.5886 | 0.0188 | 0.0102 | 0.9920 |
| all_records_sleep_only | window_60s | respiratory_spo2 | ExtraTrees | extra_trees | 0.5869 | 0.1117 | 0.0721 | 0.9547 |

## Window Comparison

| window_set | regime | feature_set | model | config_name | roc_auc_mean | f1_mean |
| --- | --- | --- | --- | --- | --- | --- |
| window_10s | exclude_ucddb005_sleep_only | flow_spo2 | XGBoost | xgb_shallow | 0.6051 | 0.3057 |
| window_60s | all_records_sleep_only | flow_spo2 | XGBoost | xgb_shallow | 0.5939 | 0.3995 |

## Feature Set Comparison

| feature_set | regime | window_set | model | config_name | roc_auc_mean | f1_mean |
| --- | --- | --- | --- | --- | --- | --- |
| flow_spo2 | exclude_ucddb005_sleep_only | window_10s | XGBoost | xgb_shallow | 0.6051 | 0.3057 |
| respiratory_spo2 | exclude_ucddb005_sleep_only | window_10s | XGBoost | xgb_shallow | 0.5967 | 0.2922 |

## ucddb005 Exclusion

| regime | window_set | feature_set | model | config_name | roc_auc_mean | f1_mean |
| --- | --- | --- | --- | --- | --- | --- |
| exclude_ucddb005_sleep_only | window_10s | flow_spo2 | XGBoost | xgb_shallow | 0.6051 | 0.3057 |
| all_records_sleep_only | window_60s | flow_spo2 | XGBoost | xgb_shallow | 0.5939 | 0.3995 |

## Leakage Controls

The reduced experiment uses only subject-level CV grouped by `record_id`. The feature matrix excludes `record_id`, segment IDs, time columns, sleep stage, labels, overlap diagnostics, and any `event_*` columns. Respiratory event annotations are used only through the already-built label in `segment_features.csv`.

SpO2 future-context features (`spo2_next_30s_*`, `spo2_next_60s_*`) are retained from the segment feature table. They do not use labels or event metadata, but they are valid only for offline retrospective PSG analysis, not real-time detection.
