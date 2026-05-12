# Pipeline Map

This document separates the final thesis pipeline from supporting experiments.
All commands are intended to be run from the repository root.

## Main Pipeline

| Step | Script | Input | Main output | Role |
| --- | --- | --- | --- | --- |
| 1 | `scripts/pipeline/00_download_ucddb.py` | PhysioNet UCDDB | `data/raw/` | Download selected UCDDB files. |
| 2 | `scripts/pipeline/01_data_audit.py` | `data/raw/` | `reports/tables/ucddb_audit.csv` | Check raw records, channels, and annotations. |
| 3 | `scripts/pipeline/02_build_epochs.py` | raw records and annotations | `data/processed/epochs.csv` | Build 30-second epoch table with binary labels. |
| 4 | `scripts/pipeline/03_extract_features.py` | `epochs.csv`, raw records | `data/processed/features_all.csv` | Extract base Flow, SpO2, effort, and ECG features. |
| 5 | `scripts/pipeline/03_build_model_ready_features.py` | `features_all.csv` | `data/processed/features_model_ready.csv` | Add sleep flag, record-normalized features, and local temporal context. |
| 6 | `scripts/pipeline/03_validate_dataset.py` | processed feature tables | `reports/validation_report.md` | Validate labels, missingness, ranges, and split readiness. |
| 7 | `scripts/pipeline/04_train_improved_models.py` | `features_model_ready.csv` | `reports/tables/improved_*.csv` | Train improved subject-level CV models. |
| 8 | `scripts/analysis/05_analyze_results.py` | improved predictions/results | `reports/final_analysis_report.md` | Analyze improved-model results and thresholds. |
| 9 | `scripts/pipeline/04_train_temporal_ensemble.py` | model-ready features and improved predictions | `reports/temporal_ensemble_report.md` | Train/evaluate final temporal ensemble. |
| 10 | `scripts/mlops/benchmark_inference.py` | model-ready features | `reports/performance_report.md` | Export the final model bundle and measure inference latency/memory. |
| 11 | `scripts/analysis/06_error_analysis.py` | temporal OOF predictions and model-ready features | `reports/error_analysis_report.md` | Break down FP/FN by record, sleep stage, and diagnostic signal features. |
| 12 | `scripts/pipeline/06_audit_pipeline.py` | scripts and generated tables | `reports/pipeline_audit_report.md` | Check leakage controls and result consistency. |
| 13 | `scripts/pipeline/06_interpret_final_models.py` | `features_model_ready.csv` | `reports/interpretability_report.md` | Train final component models for feature importance. |
| 14 | `scripts/pipeline/07_make_final_artifacts.py` | generated reports/tables | `reports/final_artifact_index.md` | Collect final thesis tables and figures. |

## Supporting Analysis

| Script | Role |
| --- | --- |
| `scripts/analysis/04_diagnose_results.py` | Baseline diagnostics, fold balance, sleep-stage distribution, and threshold sweep. |
| `scripts/analysis/05_analyze_1dcnn_improvements.py` | Summarizes CNN/ResNet1D predictions and DL post-processing experiments. |
| `scripts/analysis/06_error_analysis.py` | Final temporal ensemble FP/FN analysis for thesis discussion. |

## Additional Experiments

| Script | Status | Role |
| --- | --- | --- |
| `scripts/experiments/04_train_models.py` | Keep | Classical baseline models on base features. |
| `scripts/experiments/03_build_advanced_features.py` | Keep | Builds wider contextual and interaction features. |
| `scripts/experiments/04_train_advanced_models.py` | Keep | Tests six XGBoost configurations on advanced features. |
| `scripts/experiments/03_build_segment_features.py` | Keep | Builds segment-level features from raw signals. |
| `scripts/experiments/04_train_segment_models_reduced.py` | Keep | Reduced segment-level experiment used in final discussion. |
| `scripts/experiments/04_train_1dcnn_raw_signals.py` | Keep | Controlled raw-signal CNN baseline. |
| `scripts/experiments/04_train_1dcnn_improved.py` | Keep | ResNet1D improvement experiment with longer context. |

## Legacy

| Script | Reason |
| --- | --- |
| `scripts/legacy/04_train_segment_models.py` | Superseded by the reduced segment-level experiment; kept for traceability. |
| `scripts/legacy/05_analyze_segment_results.py` | Analysis for the superseded full segment-level grid. |

## Reproducibility Notes

- Raw UCDDB files are not committed and should be downloaded with
  `scripts/pipeline/00_download_ucddb.py`.
- The project can be run in Docker. The default image is built from
  `requirements.txt`; optional DL and MLOps variants use `requirements-dl.txt`
  and `requirements-mlops.txt`.
- The main pipeline is described in `dvc.yaml`; the current artifact hashes are
  locked in `dvc.lock`. Outputs use `cache: false` because generated reports
  and processed tables are currently tracked in Git.
- Optional MLflow and W&B tracking is available for `04_train_improved_models.py`
  and `04_train_temporal_ensemble.py`. MLflow is enabled with
  `UCDDB_ENABLE_MLFLOW=1`; W&B is enabled with `UCDDB_ENABLE_WANDB=1`.
- The final evaluation uses subject-level cross-validation grouped by
  `record_id`.
- Leakage-prone columns such as labels, event metadata, `record_id`,
  `epoch_id`, and `sleep_stage` are excluded from model features.
- Centered temporal smoothing is valid only for offline retrospective analysis
  because it uses future probabilities inside the same record.
