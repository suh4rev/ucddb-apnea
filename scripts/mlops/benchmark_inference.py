from __future__ import annotations

import os
import platform
import sys
import time
import tracemalloc
import importlib.util
from pathlib import Path
from typing import Callable

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import REPORTS_TABLES_DIR  # noqa: E402


def load_temporal_module():
    module_path = PROJECT_ROOT / "scripts" / "pipeline" / "04_train_temporal_ensemble.py"
    spec = importlib.util.spec_from_file_location("temporal_ensemble", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load temporal ensemble module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


temporal = load_temporal_module()


REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = REPORTS_DIR / "models"
BENCHMARK_TABLE_PATH = REPORTS_TABLES_DIR / "inference_benchmark.csv"
PERFORMANCE_REPORT_PATH = REPORTS_DIR / "performance_report.md"
MODEL_ARTIFACT_PATH = MODELS_DIR / "final_temporal_ensemble_components.joblib"

FINAL_THRESHOLD = 0.45
FINAL_SMOOTHING_WINDOW = 61
FINAL_SMOOTHING_CENTERED = True
REPETITIONS = 30


def try_import_psutil():
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return None

    return psutil


def current_rss_mb() -> float | None:
    psutil = try_import_psutil()
    if psutil is None:
        return None

    return float(psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024))


def fit_final_components(
    sleep_features: pd.DataFrame,
    feature_sets: dict[str, list[str]],
):
    y = sleep_features["label_binary"].astype(int)
    spo2_columns = feature_sets["spo2_enhanced"]
    flow_spo2_columns = feature_sets["flow_spo2_enhanced"]

    spo2_model = temporal.make_spo2_xgboost(y)
    flow_spo2_model = temporal.make_flow_spo2_hgb(y)

    spo2_model.fit(sleep_features[spo2_columns], y)
    flow_spo2_model.fit(sleep_features[flow_spo2_columns], y)

    return {
        "spo2_model": spo2_model,
        "flow_spo2_model": flow_spo2_model,
        "spo2_columns": spo2_columns,
        "flow_spo2_columns": flow_spo2_columns,
        "threshold": FINAL_THRESHOLD,
        "smoothing_window_epochs": FINAL_SMOOTHING_WINDOW,
        "smoothing_centered": FINAL_SMOOTHING_CENTERED,
    }


def predict_temporal_ensemble(
    model_bundle: dict[str, object],
    features: pd.DataFrame,
) -> np.ndarray:
    spo2_columns = model_bundle["spo2_columns"]
    flow_spo2_columns = model_bundle["flow_spo2_columns"]
    spo2_model = model_bundle["spo2_model"]
    flow_spo2_model = model_bundle["flow_spo2_model"]

    spo2_proba = spo2_model.predict_proba(features[spo2_columns])[:, 1]
    flow_spo2_proba = flow_spo2_model.predict_proba(features[flow_spo2_columns])[:, 1]
    ensemble = (spo2_proba + flow_spo2_proba) / 2.0

    prediction_frame = features[["record_id", "epoch_id"]].copy()
    prediction_frame["ensemble_mean"] = ensemble
    smoothed = temporal.smooth_scores(
        prediction_frame,
        "ensemble_mean",
        FINAL_SMOOTHING_WINDOW,
        FINAL_SMOOTHING_CENTERED,
    )
    return smoothed.to_numpy(dtype=float)


def benchmark_callable(
    name: str,
    fn: Callable[[], np.ndarray],
    n_epochs: int,
    n_records: int,
    repetitions: int,
) -> dict[str, object]:
    # Warm-up run.
    fn()

    rss_before = current_rss_mb()
    tracemalloc.start()
    durations = []
    last_result = None
    for _ in range(repetitions):
        start = time.perf_counter()
        last_result = fn()
        durations.append(time.perf_counter() - start)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = current_rss_mb()

    duration_mean = float(np.mean(durations))
    duration_std = float(np.std(durations, ddof=1)) if len(durations) > 1 else 0.0
    positive_rate = (
        float(np.mean(last_result >= FINAL_THRESHOLD)) if last_result is not None else np.nan
    )

    return {
        "task": name,
        "n_epochs": int(n_epochs),
        "n_records": int(n_records),
        "repetitions": int(repetitions),
        "total_time_sec_mean": duration_mean,
        "total_time_sec_std": duration_std,
        "time_per_epoch_ms": duration_mean * 1000.0 / max(n_epochs, 1),
        "time_per_record_ms": duration_mean * 1000.0 / max(n_records, 1),
        "predicted_positive_rate_at_0_45": positive_rate,
        "peak_tracemalloc_mb": float(peak_bytes / (1024 * 1024)),
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_delta_mb": None
        if rss_before is None or rss_after is None
        else float(rss_after - rss_before),
    }


def build_report(table: pd.DataFrame, model_artifact_size_mb: float) -> str:
    full = table[table["task"] == "full_sleep_dataset"].iloc[0]
    per_record = table[table["task"] == "median_single_record"].iloc[0]

    return f"""# Inference Performance Report

## Scope

The benchmark measures inference for the final temporal ML ensemble on already
prepared model-ready features. It does not include raw PSG download, EDF/REC
reading, epoch construction, or feature extraction.

## Environment

- OS: `{platform.platform()}`
- Python: `{platform.python_version()}`
- CPU cores visible to Python: `{os.cpu_count()}`
- GPU required for final ML model: no
- Model artifact: `{MODEL_ARTIFACT_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- Model artifact size: {model_artifact_size_mb:.3f} MB

## Main Results

- Full sleep-only subset: {int(full['n_epochs'])} epochs across {int(full['n_records'])} records.
- Mean full-subset inference time: {full['total_time_sec_mean']:.6f} s.
- Mean latency per epoch: {full['time_per_epoch_ms']:.6f} ms.
- Mean latency per record in full-batch mode: {full['time_per_record_ms']:.6f} ms.
- Median single-record benchmark latency: {per_record['time_per_record_ms']:.6f} ms per record.
- Peak Python allocation during repeated full-batch inference: {full['peak_tracemalloc_mb']:.3f} MB.

## Optimization Notes

The final model is CPU-friendly because it uses tree-based tabular models and
rolling probability smoothing. GPU is not required for deployment. The component
models and feature lists are exported with `joblib`; XGBoost can also be exported
to its native JSON format. Quantization is not a primary optimization target for
this tree-based pipeline, but model export to ONNX/runtime-specific formats can
be considered for production inference.

## Output Table

Detailed measurements are saved to
`{BENCHMARK_TABLE_PATH.relative_to(PROJECT_ROOT).as_posix()}`.
"""


def main() -> None:
    temporal.import_xgboost()
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    features = temporal.load_features()
    sleep_features = features[features["is_sleep_epoch"] == 1].copy().reset_index(drop=True)
    sleep_features = sleep_features.sort_values(["record_id", "epoch_id"]).reset_index(
        drop=True
    )
    feature_sets = temporal.build_feature_sets(sleep_features)
    model_bundle = fit_final_components(sleep_features, feature_sets)

    joblib.dump(model_bundle, MODEL_ARTIFACT_PATH)
    model_artifact_size_mb = MODEL_ARTIFACT_PATH.stat().st_size / (1024 * 1024)

    rows = [
        benchmark_callable(
            name="full_sleep_dataset",
            fn=lambda: predict_temporal_ensemble(model_bundle, sleep_features),
            n_epochs=len(sleep_features),
            n_records=sleep_features["record_id"].nunique(),
            repetitions=REPETITIONS,
        )
    ]

    per_record_rows = []
    for _, record_features in sleep_features.groupby("record_id", sort=True):
        row = benchmark_callable(
            name="single_record",
            fn=lambda record_features=record_features: predict_temporal_ensemble(
                model_bundle,
                record_features,
            ),
            n_epochs=len(record_features),
            n_records=1,
            repetitions=max(5, REPETITIONS // 3),
        )
        per_record_rows.append(row)

    per_record_table = pd.DataFrame(per_record_rows)
    median_record = per_record_table.median(numeric_only=True).to_dict()
    median_record.update(
        {
            "task": "median_single_record",
            "n_epochs": int(per_record_table["n_epochs"].median()),
            "n_records": 1,
            "repetitions": int(per_record_table["repetitions"].median()),
        }
    )
    rows.append(median_record)

    output = pd.DataFrame(rows)
    output.to_csv(BENCHMARK_TABLE_PATH, index=False)
    PERFORMANCE_REPORT_PATH.write_text(
        build_report(output, model_artifact_size_mb),
        encoding="utf-8",
    )

    print("Inference benchmark complete")
    print(f"  Table: {BENCHMARK_TABLE_PATH}")
    print(f"  Report: {PERFORMANCE_REPORT_PATH}")
    print(f"  Model artifact: {MODEL_ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
