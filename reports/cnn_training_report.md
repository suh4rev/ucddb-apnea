# 1D-CNN Raw Signal Training Report

## Goal

Controlled deep learning baseline for binary sleep-only epoch classification on UCDDB: `normal` vs `apnea_hypopnea`.

## Data And Leakage Controls

- Input files: `data/processed/epochs.csv` and `data/raw/ucddb*.rec`.
- Signals: Flow, SpO2, ribcage, abdo.
- Excluded from the model input: ECG, sleep stage, record/epoch/time identifiers, labels, and event metadata.
- Sleep stages are used only to filter sleep-only epochs.
- Cross-validation: subject-level 5-fold CV by `record_id`.
- Fold source: saved improved baseline folds.

## Architecture

`Conv1d(4, 32, kernel_size=7, padding=3)` -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.1)

`Conv1d(32, 64, kernel_size=5, padding=2)` -> BatchNorm -> ReLU -> MaxPool -> Dropout(0.15)

`Conv1d(64, 128, kernel_size=3, padding=1)` -> BatchNorm -> ReLU -> AdaptiveAvgPool1d(1)

`Linear(128, 1)`

Training used BCEWithLogitsLoss with fold-specific `pos_weight`, Adam (`lr=1e-3`, `weight_decay=1e-4`), batch size 128, maximum 50 epochs, and early stopping with patience 7 by validation ROC-AUC.

Device used: `cpu`.

## Input Modes

- `cnn_30s`: current 30-second epoch, input shape `4 x 240`.
- `cnn_90s_context`: previous + current + next epoch, input shape `4 x 720`. This mode is offline retrospective because it uses the future neighboring epoch.

## Results At Threshold 0.5

| input_mode | input_shape | offline_retrospective | roc_auc_mean | f1_mean | sensitivity_mean | specificity_mean | average_precision_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| cnn_90s_context | 4x720 | True | 0.5953 | 0.3453 | 0.4400 | 0.6951 | 0.2934 |
| cnn_30s | 4x240 | False | 0.5778 | 0.3326 | 0.4915 | 0.6333 | 0.2794 |

## Threshold Sweep

| input_mode | selection_rule | threshold | f1 | sensitivity | specificity |
| --- | --- | --- | --- | --- | --- |
| cnn_30s | max_f1 | 0.4000 | 0.3611 | 0.7129 | 0.3802 |
| cnn_30s | max_youden | 0.5000 | 0.3473 | 0.4835 | 0.6393 |
| cnn_30s | sensitivity_ge_0_70_max_specificity | 0.4000 | 0.3611 | 0.7129 | 0.3802 |
| cnn_90s_context | max_f1 | 0.4000 | 0.3782 | 0.6754 | 0.4745 |
| cnn_90s_context | max_youden | 0.4500 | 0.3764 | 0.5670 | 0.5992 |
| cnn_90s_context | sensitivity_ge_0_70_max_specificity | 0.3500 | 0.3679 | 0.7396 | 0.3677 |

## Comparison With Temporal Ensemble

Previous best temporal ensemble: ROC-AUC=0.7066, F1=0.4349.

Best CNN ROC-AUC: 0.5953 (-0.1113 vs temporal ensemble).

Best CNN F1 at threshold 0.5: 0.3453 (-0.0896 vs temporal ensemble).

Best tuned CNN F1: 0.3782.

The CNN did not improve over the temporal ensemble. This is plausible: UCDDB has few subjects, validation is subject-level, the architecture is intentionally simple, and no large-scale pretraining is used.

## Limitations

- The dataset has only 25 subjects, so subject-level folds are high variance.
- The model is a deliberately small 1D-CNN baseline, not a large tuned deep learning system.
- No external dataset validation is performed here.
- `cnn_90s_context` is suitable only for offline retrospective PSG analysis.
