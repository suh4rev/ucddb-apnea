from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, REPORTS_TABLES_DIR  # noqa: E402


WINDOW_CONFIGS = [
    {"window_sec": 60, "stride_sec": 10},
    {"window_sec": 10, "stride_sec": 5},
]

POSITIVE_LABEL = "apnea_hypopnea"
NEGATIVE_LABEL = "normal"
RESPIRATORY_EVENT_PREFIXES = ("APNEA", "HYP")
TIME_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")
NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
RECORD_ID_PATTERN = re.compile(r"ucddb\d+", re.IGNORECASE)

CHANNEL_ALIASES = {
    "flow": {"flow"},
    "spo2": {"spo2", "sp02"},
    "ribcage": {"ribcage"},
    "abdo": {"abdo"},
}

METADATA_COLUMNS = [
    "record_id",
    "segment_id",
    "window_sec",
    "stride_sec",
    "start_sec",
    "end_sec",
    "sleep_stage_majority",
    "is_sleep_segment",
    "label_binary",
    "label",
    "overlap_seconds",
    "max_event_overlap_seconds",
]

FLOW_FEATURE_COLUMNS = [
    "flow_mean",
    "flow_std",
    "flow_min",
    "flow_max",
    "flow_range",
    "flow_abs_mean",
    "flow_rms",
    "flow_energy",
    "flow_p05",
    "flow_p10",
    "flow_p25",
    "flow_p50",
    "flow_p75",
    "flow_p90",
    "flow_p95",
    "flow_low_amp_ratio_10",
    "flow_low_amp_ratio_20",
    "flow_low_amp_ratio_30",
    "flow_longest_low_amp_sec_10",
    "flow_longest_low_amp_sec_20",
    "flow_longest_low_amp_sec_30",
    "flow_zero_crossing_rate",
    "flow_flatness_ratio",
    "flow_resp_rate_estimate",
    "flow_band_power_resp",
]

SPO2_FEATURE_COLUMNS = [
    "spo2_mean",
    "spo2_std",
    "spo2_min",
    "spo2_max",
    "spo2_range",
    "spo2_p05",
    "spo2_p10",
    "spo2_p50",
    "spo2_below_90_ratio",
    "spo2_below_92_ratio",
    "spo2_below_94_ratio",
    "spo2_slope",
    "spo2_drop_from_record_median",
    "spo2_drop_from_prev_30s_median",
    "spo2_drop_from_prev_60s_median",
    "spo2_next_30s_min",
    "spo2_next_60s_min",
    "spo2_next_30s_drop",
    "spo2_next_60s_drop",
    "spo2_area_below_90",
    "spo2_area_below_92",
]

EFFORT_FEATURE_COLUMNS = [
    "ribcage_energy",
    "abdo_energy",
    "ribcage_abs_mean",
    "abdo_abs_mean",
    "effort_corr",
    "effort_diff_std",
    "effort_sum_energy",
    "flow_to_ribcage_energy_ratio",
    "flow_to_abdo_energy_ratio",
    "ribcage_to_abdo_energy_ratio",
    "flow_low_amp_x_spo2_drop",
]

FEATURE_COLUMNS = [
    *FLOW_FEATURE_COLUMNS,
    *SPO2_FEATURE_COLUMNS,
    *EFFORT_FEATURE_COLUMNS,
]

OUTPUT_PATH = DATA_PROCESSED_DIR / "segment_features.csv"
SUMMARY_PATH = REPORTS_TABLES_DIR / "segment_feature_summary.csv"


def import_pyedflib():
    try:
        import pyedflib
    except ImportError as exc:
        raise SystemExit(
            "pyedflib is not installed. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    return pyedflib


def get_record_id(path: Path) -> str | None:
    match = RECORD_ID_PATTERN.search(path.name)
    if match:
        return match.group(0).lower()

    return None


def find_signal_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []

    return sorted(
        path
        for path in raw_dir.glob("ucddb*.rec")
        if path.is_file() and "lifecard" not in path.name.lower()
    )


def find_optional_file(record_id: str, suffix: str) -> Path | None:
    matches = sorted(DATA_RAW_DIR.glob(f"{record_id}{suffix}"))
    if matches:
        return matches[0]

    return None


def normalize_channel_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )


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


def read_record_signals(signal_path: Path) -> tuple[dict[str, dict[str, object]], float]:
    pyedflib = import_pyedflib()
    signals: dict[str, dict[str, object]] = {}

    with pyedflib.EdfReader(str(signal_path)) as reader:
        labels = reader.getSignalLabels()
        frequencies = reader.getSampleFrequencies()
        channel_indices = find_channel_indices(labels)
        duration_seconds = float(reader.file_duration)

        for modality, channel_index in channel_indices.items():
            signals[modality] = {
                "signal": np.asarray(reader.readSignal(channel_index), dtype=float),
                "fs": float(frequencies[channel_index]),
                "label": labels[channel_index],
            }

    return signals, duration_seconds


def time_to_seconds(value: str) -> int:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def is_number(value: str) -> bool:
    return bool(NUMBER_PATTERN.match(value))


def parse_duration(tokens: list[str]) -> float | None:
    for token in tokens:
        cleaned_token = token.strip().rstrip(",;")
        if is_number(cleaned_token):
            return float(cleaned_token)

    return None


def is_respiratory_event(event_type: str) -> bool:
    normalized_event_type = event_type.upper()
    return normalized_event_type.startswith(RESPIRATORY_EVENT_PREFIXES)


def parse_resp_event_file(event_path: Path | None) -> list[dict[str, object]]:
    if event_path is None:
        return []

    events: list[dict[str, object]] = []
    with event_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            tokens = line.split()
            if not tokens or not TIME_PATTERN.match(tokens[0]) or len(tokens) < 3:
                continue

            event_type = tokens[1]
            if not is_respiratory_event(event_type):
                continue

            duration_sec = parse_duration(tokens[2:])
            if duration_sec is None:
                continue

            start_sec = float(time_to_seconds(tokens[0]))
            events.append(
                {
                    "start_sec": start_sec,
                    "end_sec": start_sec + float(duration_sec),
                    "event_type": event_type,
                    "duration_sec": float(duration_sec),
                }
            )

    return events


def parse_stage_value(line: str) -> int | float | None:
    token = line.strip().split()[0] if line.strip() else ""
    if not token:
        return None

    value = float(token)
    if value.is_integer():
        return int(value)

    return value


def read_stage_file(stage_path: Path | None) -> list[int | float | None]:
    if stage_path is None:
        return []

    stages: list[int | float | None] = []
    with stage_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            try:
                stages.append(parse_stage_value(line))
            except (ValueError, IndexError):
                stages.append(None)

    return stages


def finite_values(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    return values[np.isfinite(values)]


def empty_features(columns: list[str]) -> dict[str, float]:
    return {column: np.nan for column in columns}


def safe_nanmedian(values: np.ndarray) -> float:
    finite = finite_values(values)
    if finite.size == 0:
        return np.nan

    return float(np.median(finite))


def safe_nanmin(values: np.ndarray) -> float:
    finite = finite_values(values)
    if finite.size == 0:
        return np.nan

    return float(np.min(finite))


def robust_normalize_signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=float)
    finite = finite_values(values)
    if finite.size == 0:
        return np.full_like(values, np.nan, dtype=float)

    median = float(np.median(finite))
    q75, q25 = np.percentile(finite, [75, 25])
    scale = float(q75 - q25)
    if not np.isfinite(scale) or scale == 0:
        scale = float(np.std(finite))

    if not np.isfinite(scale) or scale == 0:
        return values

    return (values - median) / scale


def clean_spo2_signal(signal: np.ndarray) -> np.ndarray:
    values = np.asarray(signal, dtype=float).copy()
    values[(values < 50) | (values > 100)] = np.nan
    return values


def prepare_record_signals(
    record_signals: dict[str, dict[str, object]],
) -> dict[str, dict[str, object]]:
    prepared: dict[str, dict[str, object]] = {}

    for modality, payload in record_signals.items():
        signal = np.asarray(payload["signal"], dtype=float)
        fs = float(payload["fs"])
        label = str(payload["label"])

        if modality in {"flow", "ribcage", "abdo"}:
            signal = robust_normalize_signal(signal)
        elif modality == "spo2":
            signal = clean_spo2_signal(signal)

        prepared[modality] = {
            "signal": signal,
            "fs": fs,
            "label": label,
        }

    return prepared


def get_segment_signal(
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


def get_fs(record_signals: dict[str, dict[str, object]], modality: str) -> float | None:
    if modality not in record_signals:
        return None

    return float(record_signals[modality]["fs"])


def percentile(values: np.ndarray, q: float) -> float:
    finite = finite_values(values)
    if finite.size == 0:
        return np.nan

    return float(np.percentile(finite, q))


def rms(values: np.ndarray) -> float:
    finite = finite_values(values)
    if finite.size == 0:
        return np.nan

    return float(np.sqrt(np.mean(finite**2)))


def energy(values: np.ndarray) -> float:
    finite = finite_values(values)
    if finite.size == 0:
        return np.nan

    return float(np.sum(finite**2))


def zero_crossing_rate(values: np.ndarray) -> float:
    finite = finite_values(values)
    if finite.size < 2:
        return np.nan

    signs = np.sign(finite)
    signs[signs == 0] = 1
    return float(np.mean(signs[1:] != signs[:-1]))


def longest_true_duration(mask: np.ndarray, fs: float | None) -> float:
    if fs is None or fs <= 0 or mask.size == 0:
        return np.nan

    max_run = 0
    current_run = 0
    for value in mask:
        if bool(value):
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 0

    return float(max_run / fs)


def flow_spectral_features(values: np.ndarray, fs: float | None) -> dict[str, float]:
    if fs is None or fs <= 0:
        return {"flow_resp_rate_estimate": np.nan, "flow_band_power_resp": np.nan}

    finite = finite_values(values)
    if finite.size < max(8, int(fs * 5)):
        return {"flow_resp_rate_estimate": np.nan, "flow_band_power_resp": np.nan}

    centered = finite - np.mean(finite)
    if np.std(centered) == 0:
        return {"flow_resp_rate_estimate": np.nan, "flow_band_power_resp": 0.0}

    nperseg = min(finite.size, max(8, int(round(fs * 30))))
    freqs, psd = welch(centered, fs=fs, nperseg=nperseg)
    band_mask = (freqs >= 0.10) & (freqs <= 0.50)
    if not np.any(band_mask):
        return {"flow_resp_rate_estimate": np.nan, "flow_band_power_resp": np.nan}

    band_freqs = freqs[band_mask]
    band_psd = psd[band_mask]
    band_power = float(np.trapz(band_psd, band_freqs))

    if band_psd.size == 0 or np.all(~np.isfinite(band_psd)):
        resp_rate = np.nan
    else:
        resp_rate = float(band_freqs[int(np.nanargmax(band_psd))] * 60.0)

    return {
        "flow_resp_rate_estimate": resp_rate,
        "flow_band_power_resp": band_power,
    }


def extract_flow_features(
    signal: np.ndarray | None,
    fs: float | None,
    baseline_amp: float,
) -> dict[str, float]:
    features = empty_features(FLOW_FEATURE_COLUMNS)
    if signal is None:
        return features

    values = finite_values(signal)
    if values.size == 0:
        return features

    minimum = float(np.min(values))
    maximum = float(np.max(values))

    features.update(
        {
            "flow_mean": float(np.mean(values)),
            "flow_std": float(np.std(values)),
            "flow_min": minimum,
            "flow_max": maximum,
            "flow_range": maximum - minimum,
            "flow_abs_mean": float(np.mean(np.abs(values))),
            "flow_rms": rms(values),
            "flow_energy": energy(values),
            "flow_p05": percentile(values, 5),
            "flow_p10": percentile(values, 10),
            "flow_p25": percentile(values, 25),
            "flow_p50": percentile(values, 50),
            "flow_p75": percentile(values, 75),
            "flow_p90": percentile(values, 90),
            "flow_p95": percentile(values, 95),
            "flow_zero_crossing_rate": zero_crossing_rate(values),
        }
    )

    if np.isfinite(baseline_amp) and baseline_amp > 0:
        for percent in [10, 20, 30]:
            threshold = (percent / 100.0) * baseline_amp
            valid = np.isfinite(signal)
            low_amp_mask = valid & (np.abs(signal) < threshold)
            features[f"flow_low_amp_ratio_{percent}"] = float(np.mean(low_amp_mask[valid]))
            features[f"flow_longest_low_amp_sec_{percent}"] = longest_true_duration(
                low_amp_mask,
                fs,
            )

        diffs = np.abs(np.diff(values))
        flat_threshold = max(1e-6, 0.01 * baseline_amp)
        features["flow_flatness_ratio"] = (
            float(np.mean(diffs < flat_threshold)) if diffs.size else np.nan
        )

    features.update(flow_spectral_features(values, fs))
    return features


def get_interval_values(
    record_signals: dict[str, dict[str, object]],
    modality: str,
    start_sec: float,
    end_sec: float,
) -> np.ndarray:
    if end_sec <= start_sec:
        return np.array([], dtype=float)

    values = get_segment_signal(record_signals, modality, start_sec, end_sec)
    if values is None:
        return np.array([], dtype=float)

    return values


def extract_spo2_features(
    signal: np.ndarray | None,
    fs: float | None,
    start_sec: float,
    end_sec: float,
    record_signals: dict[str, dict[str, object]],
    record_median: float,
) -> dict[str, float]:
    features = empty_features(SPO2_FEATURE_COLUMNS)
    if signal is None:
        return features

    values = finite_values(signal)
    if values.size == 0:
        return features

    minimum = float(np.min(values))
    maximum = float(np.max(values))
    current_median = float(np.median(values))

    features.update(
        {
            "spo2_mean": float(np.mean(values)),
            "spo2_std": float(np.std(values)),
            "spo2_min": minimum,
            "spo2_max": maximum,
            "spo2_range": maximum - minimum,
            "spo2_p05": percentile(values, 5),
            "spo2_p10": percentile(values, 10),
            "spo2_p50": percentile(values, 50),
            "spo2_below_90_ratio": float(np.mean(values < 90)),
            "spo2_below_92_ratio": float(np.mean(values < 92)),
            "spo2_below_94_ratio": float(np.mean(values < 94)),
            "spo2_drop_from_record_median": record_median - minimum
            if np.isfinite(record_median)
            else np.nan,
            "spo2_area_below_90": float(np.sum(np.maximum(90 - values, 0)) / fs)
            if fs and fs > 0
            else np.nan,
            "spo2_area_below_92": float(np.sum(np.maximum(92 - values, 0)) / fs)
            if fs and fs > 0
            else np.nan,
        }
    )

    if fs is not None and fs > 0:
        valid_mask = np.isfinite(signal)
        if valid_mask.sum() >= 2:
            x = np.arange(signal.size, dtype=float)[valid_mask] / fs
            y = signal[valid_mask]
            features["spo2_slope"] = float(np.polyfit(x, y, deg=1)[0])

    prev_30 = get_interval_values(record_signals, "spo2", max(0, start_sec - 30), start_sec)
    prev_60 = get_interval_values(record_signals, "spo2", max(0, start_sec - 60), start_sec)
    next_30 = get_interval_values(record_signals, "spo2", end_sec, end_sec + 30)
    next_60 = get_interval_values(record_signals, "spo2", end_sec, end_sec + 60)

    prev_30_median = safe_nanmedian(prev_30)
    prev_60_median = safe_nanmedian(prev_60)
    next_30_min = safe_nanmin(next_30)
    next_60_min = safe_nanmin(next_60)

    features["spo2_drop_from_prev_30s_median"] = (
        prev_30_median - minimum if np.isfinite(prev_30_median) else np.nan
    )
    features["spo2_drop_from_prev_60s_median"] = (
        prev_60_median - minimum if np.isfinite(prev_60_median) else np.nan
    )
    features["spo2_next_30s_min"] = next_30_min
    features["spo2_next_60s_min"] = next_60_min
    features["spo2_next_30s_drop"] = (
        current_median - next_30_min if np.isfinite(next_30_min) else np.nan
    )
    features["spo2_next_60s_drop"] = (
        current_median - next_60_min if np.isfinite(next_60_min) else np.nan
    )

    return features


def safe_corr(x: np.ndarray | None, y: np.ndarray | None) -> float:
    if x is None or y is None:
        return np.nan

    n = min(x.size, y.size)
    if n < 2:
        return np.nan

    x_values = np.asarray(x[:n], dtype=float)
    y_values = np.asarray(y[:n], dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    if mask.sum() < 2:
        return np.nan

    x_values = x_values[mask]
    y_values = y_values[mask]
    if np.std(x_values) == 0 or np.std(y_values) == 0:
        return np.nan

    return float(np.corrcoef(x_values, y_values)[0, 1])


def pairwise_values(x: np.ndarray | None, y: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    if x is None or y is None:
        return np.array([], dtype=float), np.array([], dtype=float)

    n = min(x.size, y.size)
    if n == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    x_values = np.asarray(x[:n], dtype=float)
    y_values = np.asarray(y[:n], dtype=float)
    mask = np.isfinite(x_values) & np.isfinite(y_values)
    return x_values[mask], y_values[mask]


def safe_divide(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator) or denominator == 0:
        return np.nan

    return float(numerator / denominator)


def extract_effort_features(
    flow_features: dict[str, float],
    spo2_features: dict[str, float],
    ribcage_signal: np.ndarray | None,
    abdo_signal: np.ndarray | None,
) -> dict[str, float]:
    features = empty_features(EFFORT_FEATURE_COLUMNS)

    ribcage_energy = energy(ribcage_signal) if ribcage_signal is not None else np.nan
    abdo_energy = energy(abdo_signal) if abdo_signal is not None else np.nan
    ribcage_values = finite_values(ribcage_signal) if ribcage_signal is not None else np.array([])
    abdo_values = finite_values(abdo_signal) if abdo_signal is not None else np.array([])

    features["ribcage_energy"] = ribcage_energy
    features["abdo_energy"] = abdo_energy
    features["ribcage_abs_mean"] = (
        float(np.mean(np.abs(ribcage_values))) if ribcage_values.size else np.nan
    )
    features["abdo_abs_mean"] = (
        float(np.mean(np.abs(abdo_values))) if abdo_values.size else np.nan
    )
    features["effort_corr"] = safe_corr(ribcage_signal, abdo_signal)

    ribcage_pair, abdo_pair = pairwise_values(ribcage_signal, abdo_signal)
    if ribcage_pair.size:
        diff = ribcage_pair - abdo_pair
        effort_sum = ribcage_pair + abdo_pair
        features["effort_diff_std"] = float(np.std(diff))
        features["effort_sum_energy"] = energy(effort_sum)

    flow_energy = flow_features.get("flow_energy", np.nan)
    features["flow_to_ribcage_energy_ratio"] = safe_divide(flow_energy, ribcage_energy)
    features["flow_to_abdo_energy_ratio"] = safe_divide(flow_energy, abdo_energy)
    features["ribcage_to_abdo_energy_ratio"] = safe_divide(ribcage_energy, abdo_energy)
    features["flow_low_amp_x_spo2_drop"] = (
        flow_features.get("flow_low_amp_ratio_20", np.nan)
        * spo2_features.get("spo2_drop_from_record_median", np.nan)
    )

    return features


def stage_majority(
    stages: list[int | float | None],
    start_sec: float,
    end_sec: float,
) -> int | float | None:
    weights: dict[int | float, float] = {}

    first_epoch = max(0, int(start_sec // 30))
    last_epoch = min(len(stages) - 1, int((end_sec - 1e-9) // 30))
    if last_epoch < first_epoch:
        return None

    for epoch_index in range(first_epoch, last_epoch + 1):
        stage = stages[epoch_index] if epoch_index < len(stages) else None
        if stage is None or pd.isna(stage):
            continue

        epoch_start = epoch_index * 30.0
        epoch_end = epoch_start + 30.0
        overlap = max(0.0, min(end_sec, epoch_end) - max(start_sec, epoch_start))
        if overlap > 0:
            weights[stage] = weights.get(stage, 0.0) + overlap

    if not weights:
        return None

    return max(weights.items(), key=lambda item: (item[1], -float(item[0])))[0]


def is_sleep_stage(stage: int | float | None) -> int:
    if stage is None or pd.isna(stage):
        return 0

    return int(stage not in {0, 8})


def event_overlap_stats(
    events: list[dict[str, object]],
    start_sec: float,
    end_sec: float,
) -> tuple[int, float, float]:
    if not events:
        return 0, 0.0, 0.0

    starts = np.asarray([float(event["start_sec"]) for event in events], dtype=float)
    ends = np.asarray([float(event["end_sec"]) for event in events], dtype=float)
    mask = (starts < end_sec) & (ends > start_sec)
    if not np.any(mask):
        return 0, 0.0, 0.0

    overlaps = np.minimum(ends[mask], end_sec) - np.maximum(starts[mask], start_sec)
    overlaps = overlaps[overlaps > 0]
    if overlaps.size == 0:
        return 0, 0.0, 0.0

    return 1, float(np.sum(overlaps)), float(np.max(overlaps))


def calculate_baseline_amp(record_signals: dict[str, dict[str, object]]) -> float:
    if "flow" not in record_signals:
        return np.nan

    flow = np.asarray(record_signals["flow"]["signal"], dtype=float)
    values = finite_values(flow)
    if values.size == 0:
        return np.nan

    return float(np.percentile(np.abs(values), 95))


def build_segment_row(
    record_id: str,
    segment_id: str,
    window_sec: int,
    stride_sec: int,
    start_sec: float,
    end_sec: float,
    events: list[dict[str, object]],
    stages: list[int | float | None],
    record_signals: dict[str, dict[str, object]],
    baseline_amp: float,
    spo2_record_median: float,
) -> dict[str, object]:
    label_binary, overlap_seconds, max_overlap_seconds = event_overlap_stats(
        events,
        start_sec,
        end_sec,
    )
    majority_stage = stage_majority(stages, start_sec, end_sec)
    flow_signal = get_segment_signal(record_signals, "flow", start_sec, end_sec)
    spo2_signal = get_segment_signal(record_signals, "spo2", start_sec, end_sec)
    ribcage_signal = get_segment_signal(record_signals, "ribcage", start_sec, end_sec)
    abdo_signal = get_segment_signal(record_signals, "abdo", start_sec, end_sec)

    flow_features = extract_flow_features(
        flow_signal,
        get_fs(record_signals, "flow"),
        baseline_amp,
    )
    spo2_features = extract_spo2_features(
        spo2_signal,
        get_fs(record_signals, "spo2"),
        start_sec,
        end_sec,
        record_signals,
        spo2_record_median,
    )
    effort_features = extract_effort_features(
        flow_features,
        spo2_features,
        ribcage_signal,
        abdo_signal,
    )

    row: dict[str, object] = {
        "record_id": record_id,
        "segment_id": segment_id,
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "start_sec": start_sec,
        "end_sec": end_sec,
        "sleep_stage_majority": majority_stage,
        "is_sleep_segment": is_sleep_stage(majority_stage),
        "label_binary": label_binary,
        "label": POSITIVE_LABEL if label_binary else NEGATIVE_LABEL,
        "overlap_seconds": overlap_seconds,
        "max_event_overlap_seconds": max_overlap_seconds,
    }
    row.update(flow_features)
    row.update(spo2_features)
    row.update(effort_features)
    return row


def build_record_segments(signal_path: Path) -> list[dict[str, object]]:
    record_id = get_record_id(signal_path)
    if record_id is None:
        return []

    event_path = find_optional_file(record_id, "_respevt.txt")
    stage_path = find_optional_file(record_id, "_stage.txt")
    events = parse_resp_event_file(event_path)
    stages = read_stage_file(stage_path)

    record_signals_raw, duration_seconds = read_record_signals(signal_path)
    record_signals = prepare_record_signals(record_signals_raw)
    baseline_amp = calculate_baseline_amp(record_signals)
    spo2_record_median = (
        safe_nanmedian(np.asarray(record_signals["spo2"]["signal"], dtype=float))
        if "spo2" in record_signals
        else np.nan
    )

    rows: list[dict[str, object]] = []
    for config in WINDOW_CONFIGS:
        window_sec = int(config["window_sec"])
        stride_sec = int(config["stride_sec"])
        max_start = int(np.floor(duration_seconds - window_sec))
        if max_start < 0:
            continue

        starts = range(0, max_start + 1, stride_sec)
        for start_sec in starts:
            end_sec = float(start_sec + window_sec)
            segment_id = f"w{window_sec}_s{int(start_sec):06d}"
            rows.append(
                build_segment_row(
                    record_id=record_id,
                    segment_id=segment_id,
                    window_sec=window_sec,
                    stride_sec=stride_sec,
                    start_sec=float(start_sec),
                    end_sec=end_sec,
                    events=events,
                    stages=stages,
                    record_signals=record_signals,
                    baseline_amp=baseline_amp,
                    spo2_record_median=spo2_record_median,
                )
            )

    return rows


def calculate_positive_rate(features: pd.DataFrame) -> float:
    if features.empty:
        return np.nan

    return float(features["label_binary"].mean())


def build_summary(features: pd.DataFrame) -> pd.DataFrame:
    feature_nan_ratios = features[FEATURE_COLUMNS].isna().mean()
    without_ucddb005 = features[features["record_id"] != "ucddb005"]
    sleep_only = features[features["is_sleep_segment"] == 1]

    return pd.DataFrame(
        [
            {
                "n_rows": len(features),
                "n_records": int(features["record_id"].nunique()) if not features.empty else 0,
                "n_60s_segments": int((features["window_sec"] == 60).sum())
                if not features.empty
                else 0,
                "n_10s_segments": int((features["window_sec"] == 10).sum())
                if not features.empty
                else 0,
                "positive_rate_all": calculate_positive_rate(features),
                "positive_rate_sleep_only": calculate_positive_rate(sleep_only),
                "max_nan_ratio": float(feature_nan_ratios.max())
                if not feature_nan_ratios.empty
                else np.nan,
                "features_with_nan_over_20_percent": int((feature_nan_ratios > 0.20).sum())
                if not feature_nan_ratios.empty
                else 0,
                "n_segments_ucddb005": int((features["record_id"] == "ucddb005").sum())
                if not features.empty
                else 0,
                "positive_rate_without_ucddb005": calculate_positive_rate(without_ucddb005),
            }
        ]
    )


def main() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    signal_files = find_signal_files(DATA_RAW_DIR)
    if not signal_files:
        raise SystemExit(
            f"No UCDDB .rec files found in {DATA_RAW_DIR}. "
            "Run: python scripts/00_download_ucddb.py"
        )

    all_rows: list[dict[str, object]] = []
    print(f"Found UCDDB .rec files: {len(signal_files)}")

    for signal_path in tqdm(signal_files, desc="Building segment features"):
        record_id = get_record_id(signal_path) or signal_path.stem
        try:
            all_rows.extend(build_record_segments(signal_path))
        except Exception as exc:  # noqa: BLE001
            print(f"Could not build segments for {record_id}: {type(exc).__name__}: {exc}")

    features = pd.DataFrame(all_rows, columns=[*METADATA_COLUMNS, *FEATURE_COLUMNS])
    summary = build_summary(features)

    features.to_csv(OUTPUT_PATH, index=False)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("\nSegment feature summary")
    print(f"  rows: {len(features)}")
    print(f"  records: {features['record_id'].nunique() if not features.empty else 0}")
    print(
        "  60s / 10s segments: "
        f"{int((features['window_sec'] == 60).sum()) if not features.empty else 0} / "
        f"{int((features['window_sec'] == 10).sum()) if not features.empty else 0}"
    )
    print(f"  positive_rate_all: {float(summary['positive_rate_all'].iloc[0]):.4f}")
    print(f"  positive_rate_sleep_only: {float(summary['positive_rate_sleep_only'].iloc[0]):.4f}")
    print(
        "  features_with_nan_over_20_percent: "
        f"{int(summary['features_with_nan_over_20_percent'].iloc[0])}"
    )
    print(f"  Saved features: {OUTPUT_PATH}")
    print("  Note: spo2_next_* features use offline future signal context.")


if __name__ == "__main__":
    main()
