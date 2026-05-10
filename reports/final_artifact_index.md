# Final Artifact Index

## Summary

Dataset: 25 records, 20793 30-second epochs, positive rate 0.2002. Sleep-only subset: 16067 epochs, positive rate 0.2170.

Final best result: temporal ensemble centered smoothing with ROC-AUC=0.7066, F1=0.4349, sensitivity=0.5570, specificity=0.7304.

Best classical multimodal result: improved respiratory_spo2_fusion with ROC-AUC=0.6657.

Best temporal multimodal/offline result: centered temporal ensemble with ROC-AUC=0.7066. The causal temporal ensemble reached ROC-AUC=0.7014.

Advanced and segment-level checks:

- advanced contextual features: best ROC-AUC=0.6304; Did not improve the improved SpO2 baseline or temporal ensemble.
- segment-level reduced experiment: best ROC-AUC=0.6051; Did not improve the improved SpO2 baseline or temporal ensemble.

## Figures

- `reports/figures/final_roc_curves_with_temporal.png`: Temporal ensemble OOF probability table was not saved, so `final_roc_curves_with_temporal.png` is a ROC-AUC bar chart rather than ROC curves.
- `reports/figures/final_confusion_matrix_temporal_best.png`: confusion matrix for temporal ensemble threshold 0.45.
- `reports/figures/final_model_comparison_auc.png`: ROC-AUC bar chart for key models.
- `reports/figures/final_threshold_tradeoff.png`: threshold 0.50 vs 0.45 tradeoff for temporal ensemble.

## Tables

- `reports/tables/final_dataset_summary.csv`: final dataset summary.
- `reports/tables/final_master_results_table.csv` and `reports/tables/final_master_results_table.md`: compact final results table for the thesis.
- `reports/tables/final_practical_process_metrics.csv`: process metrics for temporal ensemble threshold 0.45.
- `reports/tables/final_negative_experiments_summary.csv`: experiments that did not improve the final baseline.

## Source Reports

- `reports/pipeline_audit_report.md`: pipeline audit and leakage checks.
- `reports/temporal_ensemble_report.md`: temporal ensemble analysis.
- `reports/segment_reduced_analysis_report.md`: reduced segment-level analysis.

## Thesis Usage Note

Use the centered temporal ensemble only as an offline retrospective PSG analysis result, because centered smoothing uses future probabilities inside a record. For a stricter non-future post-processing variant, cite the causal temporal ensemble.
