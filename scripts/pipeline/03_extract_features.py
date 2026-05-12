from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, REPORTS_TABLES_DIR  # noqa: E402


ID_COLUMNS = [
    "record_id",
    "epoch_id",
    "start_sec",
    "end_sec",
    "sleep_stage",
    "label",
    "label_binary",
]

FLOW_FEATURE_COLUMNS = [
    "flow_mean",
    "flow_std",
    "flow_min",
    "flow_max",
    "flow_range",
    "flow_rms",
    "flow_energy",
    "flow_abs_mean",
    "flow_zero_crossing_rate",
    "flow_low_amplitude_ratio",
]

RIBCAGE_FEATURE_COLUMNS = [
    "ribcage_mean",
    "ribcage_std",
    "ribcage_min",
    "ribcage_max",
    "ribcage_range",
    "ribcage_rms",
    "ribcage_energy",
    "ribcage_abs_mean",
    "ribcage_zero_crossing_rate",
]

ABDO_FEATURE_COLUMNS = [
    "abdo_mean",
    "abdo_std",
    "abdo_min",
    "abdo_max",
    "abdo_range",
    "abdo_rms",
    "abdo_energy",
    "abdo_abs_mean",
    "abdo_zero_crossing_rate",
]

EFFORT_FEATURE_COLUMNS = [
    "effort_corr",
    "effort_diff_mean",
    "effort_diff_std",
    "effort_sum_energy",
]

SPO2_FEATURE_COLUMNS = [
    "spo2_mean",
    "spo2_std",
    "spo2_min",
    "spo2_max",
    "spo2_range",
    "spo2_drop_from_median",
    "spo2_below_90_ratio",
    "spo2_below_92_ratio",
    "spo2_slope",
]

ECG_FEATURE_COLUMNS = [
    "ecg_mean",
    "ecg_std",
    "ecg_min",
    "ecg_max",
    "ecg_range",
    "ecg_rms",
    "ecg_energy",
    "ecg_n_peaks",
    "ecg_hr_mean",
    "ecg_rr_mean",
    "ecg_rr_std",
    "ecg_rmssd",
]

FEATURE_GROUPS = {
    "ecg": ECG_FEATURE_COLUMNS,
    "flow": FLOW_FEATURE_COLUMNS,
    "ribcage": RIBCAGE_FEATURE_COLUMNS,
    "abdo": ABDO_FEATURE_COLUMNS,
    "effort": EFFORT_FEATURE_COLUMNS,
    "spo2": SPO2_FEATURE_COLUMNS,
}

FEATURE_COLUMNS = [
    *FLOW_FEATURE_COLUMNS,
    *RIBCAGE_FEATURE_COLUMNS,
    *ABDO_FEATURE_COLUMNS,
    *EFFORT_FEATURE_COLUMNS,
    *SPO2_FEATURE_COLUMNS,
    *ECG_FEATURE_COLUMNS,
]

SUMMARY_COLUMNS = [
    "record_id",
    "n_epochs",
    "n_feature_rows",
    "n_missing_ecg",
    "n_missing_flow",
    "n_missing_ribcage",
    "n_missing_abdo",
    "n_missing_spo2",
    "n_failed_epochs",
]

CHANNEL_ALIASES = {
    "ecg": {"ecg"},
    "flow": {"flow"},
    "ribcage": {"ribcage"},
    "abdo": {"abdo"},
    "spo2": {"spo2", "sp02"},
}


def import_pyedflib():
    try:
        import pyedflib
    except ImportError as exc:
        raise SystemExit(
            "pyedflib is not installed. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    return pyedflib


def normalize_channel_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )


def rms(signal: np.ndarray) -> float:
    values = finite_values(signal)
    if values.size == 0:
        return np.nan

    return float(np.sqrt(np.mean(values**2)))


def energy(signal: np.ndarray) -> float:
    values = finite_values(signal)
    if values.size == 0:
        return np.nan

    return float(np.sum(values**2))


def zero_crossing_rate(signal: np.ndarray) -> float:
    values = finite_values(signal)
    if values.size < 2:
        return np.nan

    signs = np.sign(values)
    signs[signs == 0] = 1
    return float(np.mean(signs[1:] != signs[:-1]))


def finite_values(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    return values[np.isfinite(values)]


def clean_spo2_values(signal: np.ndarray) -> np.ndarray:
    values = finite_values(signal)
    return values[(values >= 50) & (values <= 100)]


def empty_features(columns: list[str]) -> dict[str, float]:
    return {column: np.nan for column in columns}


def basic_signal_features(signal: np.ndarray, prefix: str) -> dict[str, float]:
    values = finite_values(signal)
    if values.size == 0:
        return empty_features(
            [
                f"{prefix}_mean",
                f"{prefix}_std",
                f"{prefix}_min",
                f"{prefix}_max",
                f"{prefix}_range",
                f"{prefix}_rms",
                f"{prefix}_energy",
                f"{prefix}_abs_mean",
                f"{prefix}_zero_crossing_rate",
            ]
        )

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": minimum,
        f"{prefix}_max": maximum,
        f"{prefix}_range": maximum - minimum,
        f"{prefix}_rms": rms(values),
        f"{prefix}_energy": energy(values),
        f"{prefix}_abs_mean": float(np.mean(np.abs(values))),
        f"{prefix}_zero_crossing_rate": zero_crossing_rate(values),
    }


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    x_values = finite_values(x)
    y_values = finite_values(y)
    n = min(x_values.size, y_values.size)

    if n < 2:
        return np.nan

    x_values = x_values[:n]
    y_values = y_values[:n]

    if np.std(x_values) == 0 or np.std(y_values) == 0:
        return np.nan

    return float(np.corrcoef(x_values, y_values)[0, 1])


def extract_spo2_features(signal: np.ndarray) -> dict[str, float]:
    values = clean_spo2_values(signal)
    if values.size == 0:
        return empty_features(SPO2_FEATURE_COLUMNS)

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    if values.size >= 2:
        x = np.arange(values.size, dtype=float)
        slope = float(np.polyfit(x, values, deg=1)[0])
    else:
        slope = np.nan

    return {
        "spo2_mean": float(np.mean(values)),
        "spo2_std": float(np.std(values)),
        "spo2_min": minimum,
        "spo2_max": maximum,
        "spo2_range": maximum - minimum,
        "spo2_drop_from_median": float(np.median(values) - minimum),
        "spo2_below_90_ratio": float(np.mean(values < 90)),
        "spo2_below_92_ratio": float(np.mean(values < 92)),
        "spo2_slope": slope,
    }


def extract_ecg_features(signal: np.ndarray, fs: float) -> dict[str, float]:
    values = finite_values(signal)
    features = empty_features(ECG_FEATURE_COLUMNS)

    if values.size == 0:
        return features

    basic_features = basic_signal_features(values, "ecg")
    for column in ECG_FEATURE_COLUMNS[:7]:
        features[column] = basic_features[column]

    if fs <= 0 or np.std(values) == 0:
        return features

    normalized = values - np.median(values)
    if abs(np.min(normalized)) > abs(np.max(normalized)):
        normalized = -normalized

    scale = np.std(normalized)
    if scale == 0:
        return features

    normalized = normalized / scale
    min_distance_samples = max(1, int(round(0.3 * fs)))

    try:
        peaks, _ = find_peaks(
            normalized,
            distance=min_distance_samples,
            prominence=0.5,
        )
    except Exception:  # noqa: BLE001
        return features

    features["ecg_n_peaks"] = int(len(peaks))

    if len(peaks) < 3:
        return features

    rr_intervals = np.diff(peaks) / fs
    if rr_intervals.size == 0:
        return features

    min_rr = 60.0 / 220.0
    max_rr = 60.0 / 30.0
    valid_rr_intervals = rr_intervals[
        (rr_intervals >= min_rr) & (rr_intervals <= max_rr)
    ]

    if valid_rr_intervals.size < 2:
        return features

    rr_mean = float(np.mean(valid_rr_intervals))
    hr_mean = float(60.0 / rr_mean)
    if hr_mean < 30 or hr_mean > 220:
        return features

    features["ecg_rr_mean"] = rr_mean
    features["ecg_rr_std"] = float(np.std(valid_rr_intervals))
    features["ecg_hr_mean"] = hr_mean
    features["ecg_rmssd"] = float(np.sqrt(np.mean(np.diff(valid_rr_intervals) ** 2)))

    return features


def extract_flow_features(
    signal: np.ndarray,
    low_amplitude_threshold: float | None,
) -> dict[str, float]:
    features = basic_signal_features(signal, "flow")
    values = finite_values(signal)

    if values.size == 0 or low_amplitude_threshold is None:
        features["flow_low_amplitude_ratio"] = np.nan
    elif not np.isfinite(low_amplitude_threshold):
        features["flow_low_amplitude_ratio"] = np.nan
    else:
        features["flow_low_amplitude_ratio"] = float(
            np.mean(np.abs(values) < low_amplitude_threshold)
        )

    return features


def extract_effort_features(
    ribcage_signal: np.ndarray | None,
    abdo_signal: np.ndarray | None,
) -> dict[str, float]:
    if ribcage_signal is None or abdo_signal is None:
        return empty_features(EFFORT_FEATURE_COLUMNS)

    ribcage_values = finite_values(ribcage_signal)
    abdo_values = finite_values(abdo_signal)
    n = min(ribcage_values.size, abdo_values.size)

    if n == 0:
        return empty_features(EFFORT_FEATURE_COLUMNS)

    ribcage_values = ribcage_values[:n]
    abdo_values = abdo_values[:n]
    diff = ribcage_values - abdo_values
    effort_sum = ribcage_values + abdo_values

    return {
        "effort_corr": safe_corr(ribcage_values, abdo_values),
        "effort_diff_mean": float(np.mean(diff)),
        "effort_diff_std": float(np.std(diff)),
        "effort_sum_energy": energy(effort_sum),
    }


def find_channel_indices(labels: list[str]) -> dict[str, int]:
    normalized_labels = {
        normalize_channel_name(label): index for index, label in enumerate(labels)
    }
    indices: dict[str, int] = {}

    for modality, aliases in CHANNEL_ALIASES.items():
        for alias in aliases:
            if alias in normalized_labels:
                indices[modality] = normalized_labels[alias]
                break

    return indices


def read_record_signals(signal_path: Path) -> dict[str, dict[str, object]]:
    pyedflib = import_pyedflib()
    signals: dict[str, dict[str, object]] = {}

    with pyedflib.EdfReader(str(signal_path)) as reader:
        labels = reader.getSignalLabels()
        frequencies = reader.getSampleFrequencies()
        channel_indices = find_channel_indices(labels)

        for modality, channel_index in channel_indices.items():
            signals[modality] = {
                "signal": reader.readSignal(channel_index),
                "fs": float(frequencies[channel_index]),
                "label": labels[channel_index],
            }

    return signals


def get_epoch_signal(
    record_signals: dict[str, dict[str, object]],
    modality: str,
    start_sec: float,
    end_sec: float,
) -> np.ndarray | None:
    if modality not in record_signals:
        return None

    signal = np.asarray(record_signals[modality]["signal"], dtype=float)
    fs = float(record_signals[modality]["fs"])

    start_index = max(0, int(round(start_sec * fs)))
    end_index = min(signal.size, int(round(end_sec * fs)))

    if end_index <= start_index:
        return np.array([], dtype=float)

    return signal[start_index:end_index]


def get_channel_fs(
    record_signals: dict[str, dict[str, object]],
    modality: str,
) -> float | None:
    if modality not in record_signals:
        return None

    return float(record_signals[modality]["fs"])


def calculate_flow_threshold(record_signals: dict[str, dict[str, object]]) -> float | None:
    if "flow" not in record_signals:
        return None

    flow_values = finite_values(np.asarray(record_signals["flow"]["signal"], dtype=float))
    if flow_values.size == 0:
        return np.nan

    return float(0.10 * np.percentile(np.abs(flow_values), 95))


def build_empty_feature_row(epoch_row: pd.Series) -> dict[str, object]:
    row = {column: epoch_row[column] for column in ID_COLUMNS}
    row.update(empty_features(FEATURE_COLUMNS))
    return row


def build_feature_row(
    epoch_row: pd.Series,
    record_signals: dict[str, dict[str, object]],
    flow_low_amplitude_threshold: float | None,
) -> dict[str, object]:
    start_sec = float(epoch_row["start_sec"])
    end_sec = float(epoch_row["end_sec"])

    row = {column: epoch_row[column] for column in ID_COLUMNS}

    flow_signal = get_epoch_signal(record_signals, "flow", start_sec, end_sec)
    if flow_signal is None:
        row.update(empty_features(FLOW_FEATURE_COLUMNS))
    else:
        row.update(extract_flow_features(flow_signal, flow_low_amplitude_threshold))

    ribcage_signal = get_epoch_signal(record_signals, "ribcage", start_sec, end_sec)
    if ribcage_signal is None:
        row.update(empty_features(RIBCAGE_FEATURE_COLUMNS))
    else:
        row.update(basic_signal_features(ribcage_signal, "ribcage"))

    abdo_signal = get_epoch_signal(record_signals, "abdo", start_sec, end_sec)
    if abdo_signal is None:
        row.update(empty_features(ABDO_FEATURE_COLUMNS))
    else:
        row.update(basic_signal_features(abdo_signal, "abdo"))

    row.update(extract_effort_features(ribcage_signal, abdo_signal))

    spo2_signal = get_epoch_signal(record_signals, "spo2", start_sec, end_sec)
    if spo2_signal is None:
        row.update(empty_features(SPO2_FEATURE_COLUMNS))
    else:
        row.update(extract_spo2_features(spo2_signal))

    ecg_signal = get_epoch_signal(record_signals, "ecg", start_sec, end_sec)
    ecg_fs = get_channel_fs(record_signals, "ecg")
    if ecg_signal is None or ecg_fs is None:
        row.update(empty_features(ECG_FEATURE_COLUMNS))
    else:
        row.update(extract_ecg_features(ecg_signal, ecg_fs))

    return row


def group_is_missing(row: dict[str, object], columns: list[str]) -> bool:
    return all(pd.isna(row[column]) for column in columns)


def summarize_record_features(
    record_id: str,
    n_epochs: int,
    feature_rows: list[dict[str, object]],
    n_failed_epochs: int,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "n_epochs": n_epochs,
        "n_feature_rows": len(feature_rows),
        "n_missing_ecg": sum(
            group_is_missing(row, ECG_FEATURE_COLUMNS) for row in feature_rows
        ),
        "n_missing_flow": sum(
            group_is_missing(row, FLOW_FEATURE_COLUMNS) for row in feature_rows
        ),
        "n_missing_ribcage": sum(
            group_is_missing(row, RIBCAGE_FEATURE_COLUMNS) for row in feature_rows
        ),
        "n_missing_abdo": sum(
            group_is_missing(row, ABDO_FEATURE_COLUMNS) for row in feature_rows
        ),
        "n_missing_spo2": sum(
            group_is_missing(row, SPO2_FEATURE_COLUMNS) for row in feature_rows
        ),
        "n_failed_epochs": n_failed_epochs,
    }


def count_group_nan_values(features_table: pd.DataFrame) -> dict[str, int]:
    counts = {}

    for group_name, columns in FEATURE_GROUPS.items():
        counts[group_name] = int(features_table[columns].isna().sum().sum())

    return counts


def main() -> None:
    epochs_path = DATA_PROCESSED_DIR / "epochs.csv"
    features_path = DATA_PROCESSED_DIR / "features_all.csv"
    summary_path = REPORTS_TABLES_DIR / "feature_extraction_summary.csv"

    if not epochs_path.exists():
        raise SystemExit(
            f"Epoch table not found: {epochs_path}. "
            "Run: python scripts/pipeline/02_build_epochs.py"
        )

    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    epochs = pd.read_csv(epochs_path)
    if epochs.empty:
        features_table = pd.DataFrame(columns=[*ID_COLUMNS, *FEATURE_COLUMNS])
        summary_table = pd.DataFrame(columns=SUMMARY_COLUMNS)
        features_table.to_csv(features_path, index=False)
        summary_table.to_csv(summary_path, index=False)
        print("No epochs found. Empty feature table was created.")
        return

    all_feature_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    record_groups = list(epochs.groupby("record_id", sort=True))

    for record_id, record_epochs in tqdm(record_groups, desc="Extracting features"):
        signal_path = DATA_RAW_DIR / f"{record_id}.rec"
        record_feature_rows: list[dict[str, object]] = []
        n_failed_epochs = 0

        try:
            if not signal_path.exists():
                raise FileNotFoundError(f"Signal file not found: {signal_path}")

            record_signals = read_record_signals(signal_path)
            flow_low_amplitude_threshold = calculate_flow_threshold(record_signals)

            for _, epoch_row in record_epochs.iterrows():
                try:
                    feature_row = build_feature_row(
                        epoch_row,
                        record_signals,
                        flow_low_amplitude_threshold,
                    )
                except Exception:  # noqa: BLE001
                    n_failed_epochs += 1
                    feature_row = build_empty_feature_row(epoch_row)

                record_feature_rows.append(feature_row)

        except Exception as exc:  # noqa: BLE001
            print(f"Could not read {record_id}: {type(exc).__name__}: {exc}")
            n_failed_epochs = len(record_epochs)
            record_feature_rows = [
                build_empty_feature_row(epoch_row)
                for _, epoch_row in record_epochs.iterrows()
            ]

        all_feature_rows.extend(record_feature_rows)
        summary_rows.append(
            summarize_record_features(
                record_id=record_id,
                n_epochs=len(record_epochs),
                feature_rows=record_feature_rows,
                n_failed_epochs=n_failed_epochs,
            )
        )

    features_table = pd.DataFrame(all_feature_rows, columns=[*ID_COLUMNS, *FEATURE_COLUMNS])
    summary_table = pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)

    features_table.to_csv(features_path, index=False)
    summary_table.to_csv(summary_path, index=False)

    n_feature_columns = len(FEATURE_COLUMNS)
    group_nan_counts = count_group_nan_values(features_table)

    print("\nFeature extraction summary")
    print(f"  Processed records: {len(record_groups)}")
    print(f"  Feature rows: {len(features_table)}")
    print(f"  Feature columns: {n_feature_columns}")
    print("  NaN values by group:")
    for group_name, nan_count in group_nan_counts.items():
        print(f"    {group_name}: {nan_count}")
    print(f"  Saved features: {features_path}")


if __name__ == "__main__":
    main()
