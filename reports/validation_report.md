# Dataset Validation Report

This report validates prepared UCDDB tables before model training.

## Summary

- PASS: 36
- WARNING: 0
- FAIL: 0
- Epoch rows: 20793
- Feature rows: 20793
- Records: 25

## Leakage Guard

The following columns must not be used as model features `X`:

`record_id`, `epoch_id`, `start_sec`, `end_sec`, `sleep_stage`, `label`, `label_binary`

`sleep_stage` should be kept for analysis only and not used for training.

## Recommended Split

Use `StratifiedGroupKFold` when available, or `GroupKFold` with `record_id` as the grouping variable. Do not randomly split individual epochs without grouping by record.

## Output Tables

- `reports\tables\validation_checks.csv`
- `reports\tables\validation_feature_missingness.csv`
- `reports\tables\validation_feature_ranges.csv`
- `reports\tables\validation_label_distribution_by_record.csv`

## Failed Checks

No failed checks.

## Warnings

No warnings.
