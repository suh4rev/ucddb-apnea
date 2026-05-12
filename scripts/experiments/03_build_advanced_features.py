from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, REPORTS_TABLES_DIR  # noqa: E402


warnings.simplefilter("ignore", PerformanceWarning)

INPUT_PATH = DATA_PROCESSED_DIR / "features_model_ready.csv"
OUTPUT_PATH = DATA_PROCESSED_DIR / "features_advanced.csv"
SUMMARY_PATH = REPORTS_TABLES_DIR / "advanced_feature_summary.csv"

LEAK_COLUMNS = {
    "label",
    "label_binary",
    "n_events",
    "event_types",
    "event_start_sec",
    "event_duration_sec",
}

CONTEXT_FEATURES = [
    "spo2_min",
    "spo2_mean",
    "spo2_below_90_ratio",
    "spo2_below_92_ratio",
    "spo2_drop_from_median",
    "flow_low_amplitude_ratio",
    "flow_energy",
    "flow_abs_mean",
    "ribcage_energy",
    "abdo_energy",
    "effort_corr",
    "effort_diff_std",
]

RATIO_FEATURES = {
    "flow_energy_to_ribcage_energy": ("flow_energy", "ribcage_energy"),
    "flow_energy_to_abdo_energy": ("flow_energy", "abdo_energy"),
    "flow_abs_to_ribcage_abs": ("flow_abs_mean", "ribcage_abs_mean"),
    "flow_abs_to_abdo_abs": ("flow_abs_mean", "abdo_abs_mean"),
    "ribcage_to_abdo_energy_ratio": ("ribcage_energy", "abdo_energy"),
}


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def add_column(
    features: pd.DataFrame,
    column_name: str,
    values: pd.Series,
    added_columns: list[str],
) -> None:
    if column_name in LEAK_COLUMNS:
        return

    is_new_column = column_name not in features.columns
    features[column_name] = values

    if is_new_column:
        added_columns.append(column_name)


def add_context_features(features: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    result = features.copy()
    added_columns: list[str] = []

    original_index_column = "__original_index"
    result[original_index_column] = np.arange(len(result))
    sorted_result = result.sort_values(["record_id", "epoch_id"]).copy()
    grouped = sorted_result.groupby("record_id", sort=False)

    for feature in CONTEXT_FEATURES:
        if feature not in sorted_result.columns or feature in LEAK_COLUMNS:
            continue

        add_column(
            sorted_result,
            f"lag1_{feature}",
            grouped[feature].shift(1),
            added_columns,
        )
        add_column(
            sorted_result,
            f"lag2_{feature}",
            grouped[feature].shift(2),
            added_columns,
        )
        add_column(
            sorted_result,
            f"lead1_{feature}",
            grouped[feature].shift(-1),
            added_columns,
        )
        add_column(
            sorted_result,
            f"lead2_{feature}",
            grouped[feature].shift(-2),
            added_columns,
        )

        for window in [3, 5]:
            add_column(
                sorted_result,
                f"roll{window}_mean_{feature}",
                grouped[feature].transform(
                    lambda values: values.rolling(
                        window=window,
                        center=True,
                        min_periods=1,
                    ).mean()
                ),
                added_columns,
            )
            add_column(
                sorted_result,
                f"roll{window}_min_{feature}",
                grouped[feature].transform(
                    lambda values: values.rolling(
                        window=window,
                        center=True,
                        min_periods=1,
                    ).min()
                ),
                added_columns,
            )
            add_column(
                sorted_result,
                f"roll{window}_max_{feature}",
                grouped[feature].transform(
                    lambda values: values.rolling(
                        window=window,
                        center=True,
                        min_periods=1,
                    ).max()
                ),
                added_columns,
            )

    for feature in ["spo2_min", "spo2_mean"]:
        if feature not in sorted_result.columns:
            continue

        add_column(
            sorted_result,
            f"roll5_median_{feature}",
            grouped[feature].transform(
                lambda values: values.rolling(
                    window=5,
                    center=True,
                    min_periods=1,
                ).median()
            ),
            added_columns,
        )

    sorted_result = sorted_result.sort_values(original_index_column)
    sorted_result = sorted_result.drop(columns=[original_index_column])

    return sorted_result, added_columns


def add_spo2_delta_features(
    features: pd.DataFrame,
    added_columns: list[str],
) -> pd.DataFrame:
    result = features.copy()

    required_columns = {
        "spo2_min",
        "spo2_mean",
        "lag1_spo2_min",
        "lag2_spo2_min",
        "lead1_spo2_min",
        "lead2_spo2_min",
        "roll5_median_spo2_min",
        "roll5_median_spo2_mean",
    }
    missing_columns = required_columns - set(result.columns)
    if missing_columns:
        print(f"Warning: skipping some SpO2 deltas, missing columns: {sorted(missing_columns)}")

    if {"spo2_min", "lag1_spo2_min"}.issubset(result.columns):
        add_column(
            result,
            "delta1_spo2_min",
            result["spo2_min"] - result["lag1_spo2_min"],
            added_columns,
        )
    if {"spo2_min", "lag2_spo2_min"}.issubset(result.columns):
        add_column(
            result,
            "delta2_spo2_min",
            result["spo2_min"] - result["lag2_spo2_min"],
            added_columns,
        )
    if {"lead1_spo2_min", "spo2_min"}.issubset(result.columns):
        add_column(
            result,
            "lead_delta1_spo2_min",
            result["lead1_spo2_min"] - result["spo2_min"],
            added_columns,
        )
    if {"lead2_spo2_min", "spo2_min"}.issubset(result.columns):
        add_column(
            result,
            "lead_delta2_spo2_min",
            result["lead2_spo2_min"] - result["spo2_min"],
            added_columns,
        )
    if {"roll5_median_spo2_min", "spo2_min"}.issubset(result.columns):
        add_column(
            result,
            "drop_from_roll5_median_spo2_min",
            result["roll5_median_spo2_min"] - result["spo2_min"],
            added_columns,
        )
    if {"roll5_median_spo2_mean", "spo2_mean"}.issubset(result.columns):
        add_column(
            result,
            "drop_from_roll5_median_spo2_mean",
            result["roll5_median_spo2_mean"] - result["spo2_mean"],
            added_columns,
        )

    return result


def add_ratio_features(
    features: pd.DataFrame,
    added_columns: list[str],
) -> pd.DataFrame:
    result = features.copy()

    for new_column, (numerator, denominator) in RATIO_FEATURES.items():
        if numerator not in result.columns or denominator not in result.columns:
            print(f"Warning: skipping {new_column}, missing source columns.")
            continue

        add_column(
            result,
            new_column,
            safe_divide(result[numerator], result[denominator]),
            added_columns,
        )

    return result


def add_low_flow_interactions(
    features: pd.DataFrame,
    added_columns: list[str],
) -> pd.DataFrame:
    result = features.copy()
    interactions = {
        "flow_low_amp_x_spo2_drop": (
            "flow_low_amplitude_ratio",
            "spo2_drop_from_median",
        ),
        "flow_low_amp_x_spo2_below90": (
            "flow_low_amplitude_ratio",
            "spo2_below_90_ratio",
        ),
        "flow_low_amp_x_effort_diff": (
            "flow_low_amplitude_ratio",
            "effort_diff_std",
        ),
    }

    for new_column, (left, right) in interactions.items():
        if left not in result.columns or right not in result.columns:
            print(f"Warning: skipping {new_column}, missing source columns.")
            continue

        add_column(result, new_column, result[left] * result[right], added_columns)

    return result


def calculate_positive_rate(features: pd.DataFrame) -> float:
    if features.empty:
        return np.nan

    return float(features["label_binary"].mean())


def build_summary(
    features: pd.DataFrame,
    added_columns: list[str],
) -> pd.DataFrame:
    numeric_features = features.select_dtypes(include=[np.number])
    nan_ratios = numeric_features.isna().mean()
    max_nan_ratio = float(nan_ratios.max()) if not nan_ratios.empty else np.nan
    features_with_nan_over_20 = int((nan_ratios > 0.20).sum())
    sleep_only = features[features["is_sleep_epoch"] == 1]

    return pd.DataFrame(
        [
            {
                "n_rows": len(features),
                "n_columns": features.shape[1],
                "n_added_features": len(added_columns),
                "positive_rate_all": calculate_positive_rate(features),
                "positive_rate_sleep_only": calculate_positive_rate(sleep_only),
                "max_nan_ratio": max_nan_ratio,
                "features_with_nan_over_20_percent": features_with_nan_over_20,
            }
        ]
    )


def main() -> None:
    if not INPUT_PATH.exists():
        raise SystemExit(
            f"Model-ready feature table not found: {INPUT_PATH}. "
            "Run: python scripts/pipeline/03_build_model_ready_features.py"
        )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(INPUT_PATH)
    original_shape = features.shape

    features, added_context_columns = add_context_features(features)
    added_columns = list(added_context_columns)
    features = add_spo2_delta_features(features, added_columns)
    features = add_ratio_features(features, added_columns)
    features = add_low_flow_interactions(features, added_columns)

    summary = build_summary(features, added_columns)

    features.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    max_nan_ratio = float(summary["max_nan_ratio"].iloc[0])

    print("\nAdvanced feature build summary")
    print(f"  Source shape: {original_shape[0]} rows x {original_shape[1]} columns")
    print(f"  Output shape: {features.shape[0]} rows x {features.shape[1]} columns")
    print(f"  Added features: {len(added_columns)}")
    print(f"  max_nan_ratio: {max_nan_ratio:.4f}")
    print(f"  Saved features: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
