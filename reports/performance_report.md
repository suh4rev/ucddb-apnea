# Inference Performance Report

## Scope

The benchmark measures inference for the final temporal ML ensemble on already
prepared model-ready features. It does not include raw PSG download, EDF/REC
reading, epoch construction, or feature extraction.

## Environment

- OS: `Windows-11-10.0.22631-SP0`
- Python: `3.12.3`
- CPU cores visible to Python: `16`
- GPU required for final ML model: no
- Model artifact: `reports/models/final_temporal_ensemble_components.joblib`
- Model artifact size: 0.658 MB

## Main Results

- Full sleep-only subset: 16067 epochs across 25 records.
- Mean full-subset inference time: 0.205563 s.
- Mean latency per epoch: 0.012794 ms.
- Mean latency per record in full-batch mode: 8.222519 ms.
- Median single-record benchmark latency: 67.764430 ms per record.
- Peak Python allocation during repeated full-batch inference: 19.153 MB.

## Optimization Notes

The final model is CPU-friendly because it uses tree-based tabular models and
rolling probability smoothing. GPU is not required for deployment. The component
models and feature lists are exported with `joblib`; XGBoost can also be exported
to its native JSON format. Quantization is not a primary optimization target for
this tree-based pipeline, but model export to ONNX/runtime-specific formats can
be considered for production inference.

## Output Table

Detailed measurements are saved to
`reports/tables/inference_benchmark.csv`.
