# Interpretability Report

## Scope

Final component models were trained once on the sleep-only subset of `features_model_ready.csv` only for interpretability. No new validation results were produced. Rows used: 16067; SpO2 features: 38; Flow+SpO2 features: 68.

## Top SpO2 XGBoost Features

| rank | feature | importance |
| --- | --- | --- |
| 1 | spo2_std | 95.638046 |
| 2 | spo2_drop_from_median | 57.582985 |
| 3 | spo2_range | 51.546021 |
| 4 | rn_spo2_below_90_ratio | 45.472855 |
| 5 | rn_spo2_drop_from_median | 44.276482 |
| 6 | spo2_max | 38.352657 |
| 7 | spo2_min | 36.214657 |
| 8 | rn_spo2_range | 35.764557 |
| 9 | roll3_min_spo2_min | 33.560902 |
| 10 | roll3_max_spo2_below_90_ratio | 32.903439 |

## Top Flow+SpO2 HistGradientBoosting Features

| rank | feature | importance |
| --- | --- | --- |
| 1 | roll3_max_flow_low_amplitude_ratio | 0.016084 |
| 2 | spo2_std | 0.016026 |
| 3 | rn_spo2_drop_from_median | 0.014831 |
| 4 | rn_spo2_range | 0.012057 |
| 5 | rn_spo2_max | 0.010997 |
| 6 | spo2_drop_from_median | 0.010327 |
| 7 | rn_spo2_below_90_ratio | 0.007391 |
| 8 | roll3_mean_flow_energy | 0.006772 |
| 9 | rn_spo2_min | 0.005326 |
| 10 | rn_flow_min | 0.004503 |

## Physiological Interpretation

SpO2 minimum, desaturation/drop features, and ratios below 90/92 percent are directly related to oxygen desaturation during apnea and hypopnea episodes. Flow low-amplitude features reflect reduced or absent airflow, which is a core respiratory manifestation of apnea/hypopnea. Temporal and contextual features, including previous, next, and rolling-window summaries, reflect the delayed and clustered nature of desaturation relative to respiratory flow changes.

## Limitations

Feature importance does not prove causality. Correlated features can share or mask importance, and permutation importance depends on the sampled data. The models were trained on the small UCDDB dataset and are not a clinical validation. Some contextual features use future neighboring signal information and therefore belong to offline retrospective PSG analysis rather than real-time detection.

## SHAP

SHAP skipped: shap is not installed (ModuleNotFoundError).

## Files

- `reports/tables/final_feature_importance_spo2_xgboost.csv`
- `reports/tables/final_feature_importance_flow_spo2_hgb.csv`
- `reports/figures/final_feature_importance_spo2_xgboost.png`
- `reports/figures/final_feature_importance_flow_spo2_hgb.png`
