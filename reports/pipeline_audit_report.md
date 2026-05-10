# Pipeline Audit Report

## Summary

The current pipeline passes the hard audit checks. The existing model scripts use subject-level validation by record_id, and reconstructed X columns do not include labels, event metadata, epoch IDs, sleep stage, or record IDs.

Checks: PASS=26, WARNING=0, FAIL=0.

## Hard Failures

_No rows available._

## Warnings And Risks

_No rows available._

## Leakage And Split Review

Forbidden X columns audited: end_sec, epoch_id, event_duration_sec, event_start_sec, event_types, is_sleep_epoch, label, label_binary, n_events, record_id, sleep_stage, start_sec.

| script | n_reconstructed_x_columns |
| --- | --- |
| 04_train_models.py | 53 |
| 04_train_improved_models.py | 151 |
| 04_train_advanced_models.py | 260 |

No reconstructed X feature set used respiratory event metadata or labels. Existing contextual lead/next features use future signal context; this is acceptable only for offline retrospective PSG analysis and should not be described as real-time detection.

## Feature Missingness

The detailed missingness table is saved to `reports/tables/pipeline_audit_feature_missingness.csv`.

Top features with NaN ratio over 20 percent:

_No rows available._

## Result Review

Best improved baseline: regime=sleep_only / feature_variant=enhanced / experiment=spo2_only / model=XGBoost (roc_auc_mean=0.6725, f1_mean=0.4152).

Best advanced result: regime=sleep_only / experiment=advanced_respiratory_spo2 / model=XGBoost / config_name=config_4 (roc_auc_mean=0.6304, f1_mean=0.3780).

Advanced features did not improve the best improved baseline (delta ROC-AUC=-0.0421).

Best all_epochs result: regime=all_epochs / feature_variant=enhanced / experiment=late_fusion / model=XGBoost (roc_auc_mean=0.6687, f1_mean=0.3542).

Best sleep_only result: regime=sleep_only / feature_variant=enhanced / experiment=spo2_only / model=XGBoost (roc_auc_mean=0.6725, f1_mean=0.4152).

Best multimodal result: regime=all_epochs / feature_variant=enhanced / experiment=late_fusion / model=XGBoost (roc_auc_mean=0.6687, f1_mean=0.3542).

## Why Current ROC-AUC May Be Limited

The UCDDB subject count is small, and subject-level validation is intentionally harder than random epoch-level splits. Thirty-second aggregate tabular features lose waveform morphology and temporal event structure. Labels are event-overlap labels, so borderline windows around event onset and recovery are intrinsically noisy. SpO2 desaturation can lag airflow reduction, which makes single-window classification difficult without temporal modeling.

## Recommendation

Continue only if FAIL=0. The next honest experiment should move closer to event-detection logic: use raw Flow and SpO2-derived segment features, evaluate 60-second and 10-second windows, keep record_id grouped CV, exclude all event metadata from X, and clearly mark any future signal context as offline retrospective analysis.
