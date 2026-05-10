# Temporal Ensemble Experiment

## Summary

Fold source: saved improved baseline folds. All folds are subject-level groups by `record_id`.

Best ROC-AUC: spo2_flow_spo2 / MeanEnsemble / rolling_mean_centered window=61 centered=True (ROC-AUC=0.7066, F1=0.4349, sensitivity=0.5570, specificity=0.7304).

Best F1 at threshold 0.5: spo2_flow_spo2 / MeanEnsemble / rolling_mean_causal window=61 centered=False (ROC-AUC=0.7014, F1=0.4360, sensitivity=0.5568, specificity=0.7326).

Best tuned F1: spo2_flow_spo2 / MeanEnsemble / rolling_mean_centered window=61 centered=True @ threshold=0.4500 (F1=0.4502, sensitivity=0.6556, specificity=0.6516).

Comparison with previous references:

- Previous best AUC: 0.6725; delta: 0.0341
- Previous best F1: 0.4152; delta: 0.0208
- Previous tuned F1: 0.4311; delta: 0.0191

The 0.70 target was reached, but 0.75 was not reached.

## Top Results

| feature_set | model | postprocessing | smoothing_window_epochs | smoothing_centered | roc_auc_mean | roc_auc_std | f1_mean | sensitivity_mean | specificity_mean | average_precision_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| spo2_flow_spo2 | MeanEnsemble | rolling_mean_centered | 61 | True | 0.7066 | 0.0570 | 0.4349 | 0.5570 | 0.7304 | 0.4355 |
| spo2_flow_spo2 | MeanEnsemble | rolling_mean_centered | 31 | True | 0.7027 | 0.0560 | 0.4287 | 0.5520 | 0.7257 | 0.4273 |
| spo2_flow_spo2 | MeanEnsemble | rolling_mean_centered | 15 | True | 0.7018 | 0.0551 | 0.4228 | 0.5421 | 0.7253 | 0.4283 |
| spo2_flow_spo2 | MeanEnsemble | rolling_mean_causal | 61 | False | 0.7014 | 0.0575 | 0.4360 | 0.5568 | 0.7326 | 0.4243 |
| spo2_flow_spo2 | MeanEnsemble | rolling_mean_causal | 31 | False | 0.7005 | 0.0540 | 0.4247 | 0.5468 | 0.7253 | 0.4172 |
| spo2_flow_spo2 | MeanEnsemble | rolling_mean_causal | 15 | False | 0.6999 | 0.0536 | 0.4206 | 0.5391 | 0.7251 | 0.4188 |
| spo2_flow_spo2 | MeanEnsemble | raw | 0 | False | 0.6841 | 0.0639 | 0.4255 | 0.5515 | 0.7124 | 0.3973 |
| spo2_enhanced | XGBoost | raw | 0 | False | 0.6725 | 0.0570 | 0.4152 | 0.5264 | 0.7176 | 0.3717 |
| flow_spo2_enhanced | HistGradientBoosting | raw | 0 | False | 0.6721 | 0.0752 | 0.4149 | 0.5566 | 0.6919 | 0.3940 |

## Leakage Controls

The feature matrix excludes `record_id`, `epoch_id`, time columns, sleep stage, labels, sleep filter flags, and respiratory event metadata. Respiratory annotation files are not used as features.

Temporal smoothing is applied only to validation-fold probabilities inside each held-out record and does not use labels. Centered rolling smoothing uses future probabilities, so it is valid only for offline retrospective PSG analysis. The causal rolling variant avoids future probabilities, but the underlying enhanced SpO2 features still include `next1_*` and centered rolling signal context from the existing offline pipeline.

## Interpretation

The gain comes mostly from temporal probability smoothing and a small ensemble of complementary SpO2 and Flow+SpO2 models. This is physiologically plausible because apnea/hypopnea and desaturation patterns are temporally clustered. It should not be presented as a real-time detector, and it should be described as an offline retrospective classifier. The result still remains far from AUC 0.8, which likely requires raw-signal sequence modeling and external validation.
