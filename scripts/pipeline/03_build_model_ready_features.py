from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, REPORTS_TABLES_DIR  # noqa: E402


INPUT_PATH = DATA_PROCESSED_DIR / "features_all.csv"
OUTPUT_PATH = DATA_PROCESSED_DIR / "features_model_ready.csv"
SUMMARY_PATH = REPORTS_TABLES_DIR / "model_ready_feature_summary.csv"

SERVICE_COLUMNS = {
    "record_id",
    "epoch_id",
    "start_sec",
    "end_sec",
    "sleep_stage",
    "label",
    "label_binary",
    "is_sleep_epoch",
}

LEAK_COLUMNS = {
    "label",
    "label_binary",
    "event_types",
    "n_events",
    "event_start_sec",
    "event_duration_sec",
}

CONTEXT_BASE_FEATURES = [
    "flow_low_amplitude_ratio",
    "flow_energy",
    "spo2_min",
    "spo2_mean",
    "spo2_below_90_ratio",
    "spo2_below_92_ratio",
    "effort_corr",
    "ribcage_energy",
    "abdo_energy",
]


def get_numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    excluded_columns = SERVICE_COLUMNS | LEAK_COLUMNS
    numeric_columns = features.select_dtypes(include=[np.number]).columns

    return [
        column
        for column in numeric_columns
        if column not in excluded_columns and not column.startswith("rn_")
    ]


def add_is_sleep_epoch(features: pd.DataFrame) -> pd.DataFrame:
    result = features.copy()
    sleep_stage = result["sleep_stage"]

    result["is_sleep_epoch"] = (
        sleep_stage.notna() & ~sleep_stage.isin([0, 8])
    ).astype(int)

    return result


def robust_record_zscore(values: pd.Series) -> pd.Series:
    median = values.median(skipna=True)
    q75 = values.quantile(0.75)
    q25 = values.quantile(0.25)
    iqr = q75 - q25

    scale = iqr
    if pd.isna(scale) or scale == 0:
        scale = values.std(skipna=True)

    if pd.isna(scale) or scale == 0:
        return pd.Series(np.nan, index=values.index)

    return (values - median) / scale


def add_record_normalized_features(
    features: pd.DataFrame,
    numeric_feature_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    result = features.copy()
    created_columns = []

    for column in numeric_feature_columns:
        new_column = f"rn_{column}"
        result[new_column] = result.groupby("record_id", sort=False)[column].transform(
            robust_record_zscore
        )
        created_columns.append(new_column)

    return result, created_columns


def add_context_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = features.copy()
    context_columns = []

    original_index_name = "__original_index"
    result[original_index_name] = np.arange(len(result))
    sorted_result = result.sort_values(["record_id", "epoch_id"]).copy()
    grouped = sorted_result.groupby("record_id", sort=False)

    for feature in CONTEXT_BASE_FEATURES:
        if feature not in sorted_result.columns:
            continue

        prev_column = f"prev1_{feature}"
        next_column = f"next1_{feature}"
        roll_mean_column = f"roll3_mean_{feature}"
        roll_min_column = f"roll3_min_{feature}"
        roll_max_column = f"roll3_max_{feature}"

        sorted_result[prev_column] = grouped[feature].shift(1)
        sorted_result[next_column] = grouped[feature].shift(-1)
        sorted_result[roll_mean_column] = grouped[feature].transform(
            lambda values: values.rolling(window=3, center=True, min_periods=1).mean()
        )
        sorted_result[roll_min_column] = grouped[feature].transform(
            lambda values: values.rolling(window=3, center=True, min_periods=1).min()
        )
        sorted_result[roll_max_column] = grouped[feature].transform(
            lambda values: values.rolling(window=3, center=True, min_periods=1).max()
        )

        context_columns.extend(
            [
                prev_column,
                next_column,
                roll_mean_column,
                roll_min_column,
                roll_max_column,
            ]
        )

    sorted_result = sorted_result.sort_values(original_index_name)
    sorted_result = sorted_result.drop(columns=[original_index_name])

    return sorted_result, context_columns


def calculate_positive_rate(features: pd.DataFrame) -> float:
    if features.empty:
        return np.nan

    return float(features["label_binary"].mean())


def build_summary(
    model_ready_features: pd.DataFrame,
    original_feature_columns: list[str],
    record_normalized_columns: list[str],
    context_columns: list[str],
) -> pd.DataFrame:
    n_sleep_epochs = int((model_ready_features["is_sleep_epoch"] == 1).sum())
    n_wake_or_unknown_epochs = int((model_ready_features["is_sleep_epoch"] == 0).sum())
    sleep_only = model_ready_features[model_ready_features["is_sleep_epoch"] == 1]

    summary = {
        "n_rows": len(model_ready_features),
        "n_columns": model_ready_features.shape[1],
        "n_original_features": len(original_feature_columns),
        "n_record_normalized_features": len(record_normalized_columns),
        "n_context_features": len(context_columns),
        "n_sleep_epochs": n_sleep_epochs,
        "n_wake_or_unknown_epochs": n_wake_or_unknown_epochs,
        "positive_rate_all": calculate_positive_rate(model_ready_features),
        "positive_rate_sleep_only": calculate_positive_rate(sleep_only),
    }

    return pd.DataFrame([summary])


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Feature table not found: {INPUT_PATH}. "
            "Run: python scripts/pipeline/03_extract_features.py"
        )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(INPUT_PATH)
    original_shape = features.shape

    model_ready_features = add_is_sleep_epoch(features)
    original_feature_columns = get_numeric_feature_columns(model_ready_features)
    model_ready_features, record_normalized_columns = add_record_normalized_features(
        model_ready_features,
        original_feature_columns,
    )
    model_ready_features, context_columns = add_context_features(model_ready_features)

    summary = build_summary(
        model_ready_features=model_ready_features,
        original_feature_columns=original_feature_columns,
        record_normalized_columns=record_normalized_columns,
        context_columns=context_columns,
    )

    model_ready_features.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    n_sleep_epochs = int(summary["n_sleep_epochs"].iloc[0])
    positive_rate_all = float(summary["positive_rate_all"].iloc[0])
    positive_rate_sleep_only = float(summary["positive_rate_sleep_only"].iloc[0])

    print("\nModel-ready feature build summary")
    print(f"  Source shape: {original_shape[0]} rows x {original_shape[1]} columns")
    print(
        f"  Output shape: {model_ready_features.shape[0]} rows x "
        f"{model_ready_features.shape[1]} columns"
    )
    print(f"  Sleep epochs: {n_sleep_epochs}")
    print(f"  positive_rate_all: {positive_rate_all:.4f}")
    print(f"  positive_rate_sleep_only: {positive_rate_sleep_only:.4f}")
    print(f"  Saved features: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
