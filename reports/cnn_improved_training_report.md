# Improved 1D-CNN Training Report

## Goal

Controlled DL improvement experiment for UCDDB sleep-only binary classification: test whether longer offline context plus a residual CNN improves the simpler raw-signal 1D-CNN baseline.

## Data And Leakage Controls

- Input files: `data/processed/epochs.csv` and `data/raw/ucddb*.rec`.
- Signals: Flow, SpO2, ribcage, abdo.
- Excluded from model input: ECG, sleep stage, record/epoch/time identifiers, labels, and event metadata.
- Sleep stage is used only to select sleep-only epochs.
- Cross-validation: subject-level 5-fold CV by `record_id`.
- Fold source: saved improved baseline folds.

## Input Mode

- `cnn_150s_context`: prev2 + prev1 + current + next1 + next2 epochs.
- Input shape: `4 x 1200`.
- Label belongs to the central epoch.
- This is an offline retrospective mode because it uses future neighboring epochs. Centered smoothing is also offline retrospective because it uses future probabilities within a record.

## Architecture

Stem: `Conv1d(4, 32, kernel_size=7, padding=3)` -> BatchNorm -> ReLU.

Residual block 1: two `Conv1d(32, 32, kernel_size=7, padding=3)` layers, residual connection, ReLU, MaxPool, Dropout(0.1).

Residual block 2: two `Conv1d(..., 64, kernel_size=5, padding=2)` layers, residual projection 32->64, ReLU, MaxPool, Dropout(0.15).

Residual block 3: two `Conv1d(..., 128, kernel_size=3, padding=1)` layers, residual projection 64->128, ReLU.

Head: AdaptiveAvgPool1d(1) -> Linear(128, 1).

Training used BCEWithLogitsLoss with fold-specific `pos_weight`, Adam (`lr=1e-3`, `weight_decay=1e-4`), batch size 128, maximum 40 epochs, and early stopping patience 6 by validation ROC-AUC. WeightedRandomSampler was not used.

Device used: `cpu`.

## Results At Threshold 0.5

| postprocessing | smoothing_window_epochs | smoothing_centered | roc_auc_mean | f1_mean | sensitivity_mean | specificity_mean | average_precision_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| rolling_mean_centered | 31 | True | 0.5998 | 0.3724 | 0.5065 | 0.6613 | 0.2953 |
| rolling_mean_centered | 15 | True | 0.5972 | 0.3707 | 0.5119 | 0.6533 | 0.2934 |
| rolling_mean_causal | 31 | False | 0.5970 | 0.3756 | 0.5067 | 0.6669 | 0.2932 |
| rolling_mean_causal | 15 | False | 0.5938 | 0.3642 | 0.5016 | 0.6534 | 0.2877 |
| raw | 0 | False | 0.5926 | 0.3726 | 0.5202 | 0.6485 | 0.2864 |

## Threshold Sweep

| postprocessing | smoothing_window_epochs | selection_rule | threshold | f1 | sensitivity | specificity |
| --- | --- | --- | --- | --- | --- | --- |
| raw | 0 | max_f1 | 0.4500 | 0.3716 | 0.5879 | 0.5630 |
| raw | 0 | max_youden | 0.5000 | 0.3694 | 0.5145 | 0.6478 |
| raw | 0 | sensitivity_ge_0_70_max_specificity | 0.3500 | 0.3676 | 0.7244 | 0.3855 |
| rolling_mean_causal | 15 | max_f1 | 0.4500 | 0.3733 | 0.5902 | 0.5644 |
| rolling_mean_causal | 15 | max_youden | 0.4500 | 0.3733 | 0.5902 | 0.5644 |
| rolling_mean_causal | 15 | sensitivity_ge_0_70_max_specificity | 0.3500 | 0.3718 | 0.7473 | 0.3700 |
| rolling_mean_centered | 15 | max_f1 | 0.3500 | 0.3769 | 0.7571 | 0.3734 |
| rolling_mean_centered | 15 | max_youden | 0.5000 | 0.3679 | 0.5065 | 0.6544 |
| rolling_mean_centered | 15 | sensitivity_ge_0_70_max_specificity | 0.3500 | 0.3769 | 0.7571 | 0.3734 |
| rolling_mean_causal | 31 | max_f1 | 0.4500 | 0.3810 | 0.6054 | 0.5641 |
| rolling_mean_causal | 31 | max_youden | 0.4500 | 0.3810 | 0.6054 | 0.5641 |
| rolling_mean_causal | 31 | sensitivity_ge_0_70_max_specificity | 0.3500 | 0.3701 | 0.7516 | 0.3598 |
| rolling_mean_centered | 31 | max_f1 | 0.4000 | 0.3807 | 0.6934 | 0.4596 |
| rolling_mean_centered | 31 | max_youden | 0.4500 | 0.3795 | 0.6042 | 0.5619 |
| rolling_mean_centered | 31 | sensitivity_ge_0_70_max_specificity | 0.3500 | 0.3788 | 0.7674 | 0.3669 |

## Comparison

Previous simple CNN: ROC-AUC=0.5953, F1=0.3453.

Temporal ensemble: ROC-AUC=0.7066, F1=0.4349.

Best ResNet1D ROC-AUC: 0.5998.

Best ResNet1D F1 at threshold 0.5: 0.3756.

Best tuned ResNet1D F1: 0.3810.

ResNet1D improved over the simpler 1D-CNN baseline, but did not beat the temporal ensemble. This is plausible with only 25 subjects, strict subject-level validation, and no large-scale pretraining.

## Limitations

- UCDDB is small: only 25 subjects.
- Subject-level CV is intentionally strict and can produce high fold variance.
- The model is a controlled residual CNN baseline, not a large DL search.
- No external validation is performed here.
- `cnn_150s_context` and centered smoothing are offline retrospective and are not suitable for real-time use.
