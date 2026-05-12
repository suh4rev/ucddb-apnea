from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, REPORTS_TABLES_DIR  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"
PREDICTIONS_PATH = REPORTS_TABLES_DIR / "temporal_ensemble_cv_predictions.csv"
THRESHOLDS_PATH = REPORTS_TABLES_DIR / "temporal_ensemble_best_thresholds.csv"
FEATURES_PATH = DATA_PROCESSED_DIR / "features_model_ready.csv"

REPORT_PATH = REPORTS_DIR / "error_analysis_report.md"
BY_RECORD_PATH = REPORTS_TABLES_DIR / "error_analysis_by_record.csv"
BY_SLEEP_STAGE_PATH = REPORTS_TABLES_DIR / "error_analysis_by_sleep_stage.csv"
FEATURE_SUMMARY_PATH = REPORTS_TABLES_DIR / "error_analysis_feature_summary.csv"
TOP_FP_PATH = REPORTS_TABLES_DIR / "error_analysis_top_fp.csv"
TOP_FN_PATH = REPORTS_TABLES_DIR / "error_analysis_top_fn.csv"

FINAL_REGIME = "sleep_only"
FINAL_FEATURE_SET = "spo2_flow_spo2"
FINAL_MODEL = "MeanEnsemble"
FINAL_POSTPROCESSING = "rolling_mean_centered"
FINAL_SMOOTHING_WINDOW = 61
FINAL_SMOOTHING_CENTERED = True
FINAL_SELECTION_RULE = "max_f1"

DIAGNOSTIC_COLUMNS = [
    "spo2_mean",
    "spo2_min",
    "spo2_std",
    "spo2_drop_from_median",
    "spo2_below_90_ratio",
    "spo2_below_92_ratio",
    "flow_low_amplitude_ratio",
    "roll3_max_flow_low_amplitude_ratio",
    "roll3_min_spo2_min",
    "roll3_mean_spo2_mean",
    "effort_corr",
    "ecg_hr_mean",
]


def markdown_table(frame: pd.DataFrame, float_digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"

    view = frame.copy()
    for column in ["sleep_stage", "dominant_sleep_stage"]:
        if column in view.columns:
            view[column] = view[column].map(
                lambda value: "" if pd.isna(value) else str(int(float(value)))
            )
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.{float_digits}f}"
            )
    headers = list(view.columns)
    rows = view.astype(str).values.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def final_prediction_mask(frame: pd.DataFrame) -> pd.Series:
    return (
        (frame["regime"] == FINAL_REGIME)
        & (frame["feature_set"] == FINAL_FEATURE_SET)
        & (frame["model"] == FINAL_MODEL)
        & (frame["postprocessing"] == FINAL_POSTPROCESSING)
        & (frame["smoothing_window_epochs"].astype(int) == FINAL_SMOOTHING_WINDOW)
        & (frame["smoothing_centered"].astype(bool) == FINAL_SMOOTHING_CENTERED)
    )


def load_threshold(default: float = 0.45) -> float:
    if not THRESHOLDS_PATH.exists():
        return default

    thresholds = pd.read_csv(THRESHOLDS_PATH)
    mask = final_prediction_mask(thresholds) & (
        thresholds["selection_rule"] == FINAL_SELECTION_RULE
    )
    selected = thresholds.loc[mask]
    if selected.empty:
        return default
    return float(selected.iloc[0]["threshold"])


def load_final_predictions() -> tuple[pd.DataFrame, float]:
    predictions = pd.read_csv(PREDICTIONS_PATH)
    selected = predictions.loc[final_prediction_mask(predictions)].copy()
    if selected.empty:
        raise RuntimeError(
            "Final temporal ensemble predictions were not found. "
            "Run scripts/pipeline/04_train_temporal_ensemble.py first."
        )

    threshold = load_threshold()
    selected["y_pred"] = (selected["y_proba"] >= threshold).astype(int)
    return selected, threshold


def add_error_type(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    y_true = output["label_binary"].astype(int)
    y_pred = output["y_pred"].astype(int)
    conditions = [
        (y_true == 1) & (y_pred == 1),
        (y_true == 0) & (y_pred == 1),
        (y_true == 1) & (y_pred == 0),
        (y_true == 0) & (y_pred == 0),
    ]
    output["error_type"] = np.select(conditions, ["TP", "FP", "FN", "TN"])
    return output


def confusion_summary(frame: pd.DataFrame) -> dict[str, float]:
    y_true = frame["label_binary"].astype(int)
    y_pred = frame["y_pred"].astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    return {
        "n_epochs": int(len(frame)),
        "n_positive": int(y_true.sum()),
        "n_negative": int((y_true == 0).sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "fp_rate_among_actual_negative": fp / max(fp + tn, 1),
        "fn_rate_among_actual_positive": fn / max(fn + tp, 1),
        "positive_label_rate": float(y_true.mean()) if len(frame) else np.nan,
        "predicted_positive_rate": float(y_pred.mean()) if len(frame) else np.nan,
        "mean_y_proba": float(frame["y_proba"].mean()) if len(frame) else np.nan,
    }


def summarize_grouped(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group_value, group in frame.groupby(group_column, dropna=False, sort=True):
        row = {group_column: group_value}
        row.update(confusion_summary(group))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_by_record(frame: pd.DataFrame) -> pd.DataFrame:
    summary = summarize_grouped(frame, "record_id")
    sleep_stage_mode = (
        frame.groupby("record_id")["sleep_stage"]
        .agg(lambda values: values.mode().iloc[0] if not values.mode().empty else "")
        .rename("dominant_sleep_stage")
        .reset_index()
    )
    return summary.merge(sleep_stage_mode, on="record_id", how="left")


def feature_summary(frame: pd.DataFrame) -> pd.DataFrame:
    available_columns = [col for col in DIAGNOSTIC_COLUMNS if col in frame.columns]
    if not available_columns:
        return pd.DataFrame()

    rows = []
    for error_type, group in frame.groupby("error_type", sort=True):
        row = {"error_type": error_type, "n_epochs": int(len(group))}
        for column in available_columns:
            row[f"{column}_mean"] = float(group[column].mean())
            row[f"{column}_median"] = float(group[column].median())
        rows.append(row)
    return pd.DataFrame(rows)


def top_errors(frame: pd.DataFrame, error_type: str, limit: int = 30) -> pd.DataFrame:
    columns = [
        "record_id",
        "epoch_id",
        "start_sec",
        "end_sec",
        "sleep_stage",
        "label",
        "label_binary",
        "fold",
        "y_proba",
        "y_pred",
        "error_type",
    ]
    columns.extend(column for column in DIAGNOSTIC_COLUMNS if column in frame.columns)
    columns = [column for column in columns if column in frame.columns]

    errors = frame.loc[frame["error_type"] == error_type, columns].copy()
    if error_type == "FP":
        return errors.sort_values("y_proba", ascending=False).head(limit)
    if error_type == "FN":
        return errors.sort_values("y_proba", ascending=True).head(limit)
    return errors.head(limit)


def build_report(
    merged: pd.DataFrame,
    threshold: float,
    by_record: pd.DataFrame,
    by_sleep_stage: pd.DataFrame,
    features: pd.DataFrame,
) -> str:
    overall = confusion_summary(merged)
    top_fp_records = by_record.sort_values(
        ["fp", "fp_rate_among_actual_negative"], ascending=False
    ).head(5)
    top_fn_records = by_record.sort_values(
        ["fn", "fn_rate_among_actual_positive"], ascending=False
    ).head(5)

    feature_means = features.set_index("error_type") if not features.empty else pd.DataFrame()

    fp_signal_note = ""
    fn_signal_note = ""
    if {"FP", "TN"}.issubset(feature_means.index) and "spo2_min_mean" in feature_means:
        fp_signal_note = (
            f"- FP epochs have mean SpO2 min "
            f"{feature_means.loc['FP', 'spo2_min_mean']:.2f} versus "
            f"{feature_means.loc['TN', 'spo2_min_mean']:.2f} for TN, which suggests "
            "that desaturation-like patterns contribute to false alarms."
        )
    if {"FN", "TP"}.issubset(feature_means.index) and "spo2_min_mean" in feature_means:
        fn_signal_note = (
            f"- FN epochs have mean SpO2 min "
            f"{feature_means.loc['FN', 'spo2_min_mean']:.2f} versus "
            f"{feature_means.loc['TP', 'spo2_min_mean']:.2f} for TP, so missed events "
            "are often less separable by oxygen saturation alone."
        )

    top_fp_display = top_fp_records[
        [
            "record_id",
            "n_epochs",
            "fp",
            "fp_rate_among_actual_negative",
            "dominant_sleep_stage",
            "mean_y_proba",
        ]
    ]
    top_fn_display = top_fn_records[
        [
            "record_id",
            "n_epochs",
            "fn",
            "fn_rate_among_actual_positive",
            "dominant_sleep_stage",
            "mean_y_proba",
        ]
    ]
    sleep_display = by_sleep_stage[
        [
            "sleep_stage",
            "n_epochs",
            "tp",
            "fp",
            "fn",
            "tn",
            "fp_rate_among_actual_negative",
            "fn_rate_among_actual_positive",
        ]
    ]

    notes = "\n".join(
        note for note in [fp_signal_note, fn_signal_note] if note
    )
    if not notes:
        notes = "- Diagnostic feature means are saved in the feature summary table."

    return f"""# Error Analysis Report

## Scope

This report analyzes out-of-fold predictions for the final temporal ensemble:
`{FINAL_FEATURE_SET}` / `{FINAL_MODEL}` / `{FINAL_POSTPROCESSING}` with a
{FINAL_SMOOTHING_WINDOW}-epoch centered smoothing window. The decision threshold
comes from `{FINAL_SELECTION_RULE}` and equals {threshold:.2f}.

## Confusion Summary

- Total analyzed epochs: {overall['n_epochs']}
- TP: {overall['tp']}
- FP: {overall['fp']}
- FN: {overall['fn']}
- TN: {overall['tn']}
- FP rate among actual negative epochs: {overall['fp_rate_among_actual_negative']:.4f}
- FN rate among actual positive epochs: {overall['fn_rate_among_actual_positive']:.4f}

## Main Observations

{notes}
- Record-level concentration is visible: several records account for a large
  share of FP or FN epochs, so individual physiology and sensor quality should
  be discussed alongside aggregate metrics.
- Sleep-stage grouping is included because apnea manifestations and signal
  artifacts can differ across sleep stages and wake epochs.

## Records With Most FP

{markdown_table(top_fp_display)}

## Records With Most FN

{markdown_table(top_fn_display)}

## Sleep Stage Breakdown

{markdown_table(sleep_display)}

## Output Tables

- `{BY_RECORD_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{BY_SLEEP_STAGE_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{FEATURE_SUMMARY_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{TOP_FP_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- `{TOP_FN_PATH.relative_to(PROJECT_ROOT).as_posix()}`
"""


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    predictions, threshold = load_final_predictions()
    features = pd.read_csv(FEATURES_PATH)
    feature_columns = [
        column
        for column in [
            "record_id",
            "epoch_id",
            "start_sec",
            "end_sec",
            "sleep_stage",
            "label",
            *DIAGNOSTIC_COLUMNS,
        ]
        if column in features.columns
    ]
    merged = predictions.merge(
        features[feature_columns],
        on=["record_id", "epoch_id"],
        how="left",
    )
    merged = add_error_type(merged)

    by_record = summarize_by_record(merged).sort_values(
        ["fp", "fn"], ascending=False
    )
    by_sleep_stage = summarize_grouped(merged, "sleep_stage")
    diagnostics = feature_summary(merged)
    top_fp = top_errors(merged, "FP")
    top_fn = top_errors(merged, "FN")

    by_record.to_csv(BY_RECORD_PATH, index=False)
    by_sleep_stage.to_csv(BY_SLEEP_STAGE_PATH, index=False)
    diagnostics.to_csv(FEATURE_SUMMARY_PATH, index=False)
    top_fp.to_csv(TOP_FP_PATH, index=False)
    top_fn.to_csv(TOP_FN_PATH, index=False)
    REPORT_PATH.write_text(
        build_report(merged, threshold, by_record, by_sleep_stage, diagnostics),
        encoding="utf-8",
    )

    overall = confusion_summary(merged)
    print("Error analysis complete")
    print(
        "  Confusion: "
        f"TP={overall['tp']} FP={overall['fp']} "
        f"FN={overall['fn']} TN={overall['tn']}"
    )
    print(f"  Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
