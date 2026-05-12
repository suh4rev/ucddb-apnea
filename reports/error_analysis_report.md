# Error Analysis Report

## Scope

This report analyzes out-of-fold predictions for the final temporal ensemble:
`spo2_flow_spo2` / `MeanEnsemble` / `rolling_mean_centered` with a
61-epoch centered smoothing window. The decision threshold
comes from `max_f1` and equals 0.45.

## Confusion Summary

- Total analyzed epochs: 16067
- TP: 2286
- FP: 4383
- FN: 1201
- TN: 8197
- FP rate among actual negative epochs: 0.3484
- FN rate among actual positive epochs: 0.3444

## Main Observations

- FP epochs have mean SpO2 min 89.60 versus 91.61 for TN, which suggests that desaturation-like patterns contribute to false alarms.
- FN epochs have mean SpO2 min 91.63 versus 88.80 for TP, so missed events are often less separable by oxygen saturation alone.
- Record-level concentration is visible: several records account for a large
  share of FP or FN epochs, so individual physiology and sensor quality should
  be discussed alongside aggregate metrics.
- Sleep-stage grouping is included because apnea manifestations and signal
  artifacts can differ across sleep stages and wake epochs.

## Records With Most FP

| record_id | n_epochs | fp | fp_rate_among_actual_negative | dominant_sleep_stage | mean_y_proba |
| --- | --- | --- | --- | --- | --- |
| ucddb010 | 833 | 661 | 1.0000 | 3 | 0.6017 |
| ucddb006 | 716 | 528 | 1.0000 | 5 | 0.5347 |
| ucddb027 | 766 | 478 | 1.0000 | 3 | 0.6140 |
| ucddb002 | 627 | 475 | 1.0000 | 2 | 0.5150 |
| ucddb023 | 579 | 394 | 0.9899 | 3 | 0.4826 |

## Records With Most FN

| record_id | n_epochs | fn | fn_rate_among_actual_positive | dominant_sleep_stage | mean_y_proba |
| --- | --- | --- | --- | --- | --- |
| ucddb012 | 735 | 179 | 0.8326 | 3 | 0.3899 |
| ucddb024 | 749 | 140 | 0.8046 | 3 | 0.4068 |
| ucddb019 | 781 | 126 | 1.0000 | 3 | 0.2256 |
| ucddb026 | 726 | 110 | 1.0000 | 1 | 0.4059 |
| ucddb007 | 729 | 98 | 1.0000 | 3 | 0.3384 |

## Sleep Stage Breakdown

| sleep_stage | n_epochs | tp | fp | fn | tn | fp_rate_among_actual_negative | fn_rate_among_actual_positive |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 3016 | 393 | 975 | 177 | 1471 | 0.3986 | 0.3105 |
| 2 | 3403 | 847 | 905 | 179 | 1472 | 0.3807 | 0.1745 |
| 3 | 6985 | 876 | 1922 | 546 | 3641 | 0.3455 | 0.3840 |
| 4 | 673 | 56 | 178 | 72 | 367 | 0.3266 | 0.5625 |
| 5 | 1990 | 114 | 403 | 227 | 1246 | 0.2444 | 0.6657 |

## Output Tables

- `reports/tables/error_analysis_by_record.csv`
- `reports/tables/error_analysis_by_sleep_stage.csv`
- `reports/tables/error_analysis_feature_summary.csv`
- `reports/tables/error_analysis_top_fp.csv`
- `reports/tables/error_analysis_top_fn.csv`
