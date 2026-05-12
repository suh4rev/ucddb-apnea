from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import REPORTS_TABLES_DIR  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"

PREVIOUS_BEST_AUC = 0.6725
PREVIOUS_BEST_F1 = 0.4152
PREVIOUS_THRESHOLD_TUNED_F1 = 0.4311

RESULTS_SUMMARY_PATH = REPORTS_TABLES_DIR / "segment_model_results_summary.csv"
BEST_RESULTS_PATH = REPORTS_TABLES_DIR / "segment_best_results.csv"
BEST_THRESHOLDS_PATH = REPORTS_TABLES_DIR / "segment_best_thresholds.csv"
FEATURE_SUMMARY_PATH = REPORTS_TABLES_DIR / "segment_feature_summary.csv"

FINAL_TABLE_CSV_PATH = REPORTS_TABLES_DIR / "segment_final_results_table.csv"
FINAL_TABLE_MD_PATH = REPORTS_TABLES_DIR / "segment_final_results_table.md"
REPORT_PATH = REPORTS_DIR / "segment_analysis_report.md"

DISPLAY_COLUMNS = [
    "rank",
    "regime",
    "window_set",
    "feature_set",
    "model",
    "config_name",
    "postprocessing",
    "roc_auc_mean",
    "f1_mean",
    "sensitivity_mean",
    "specificity_mean",
    "average_precision_mean",
    "max_f1_threshold",
    "max_f1",
    "sens70_threshold",
    "sens70_specificity",
]


def format_metric(value: object) -> str:
    if pd.isna(value):
        return "NA"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def make_markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    if table.empty:
        return "_No rows available._"

    display = table[columns].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(format_metric)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in display.iterrows():
        body.append("| " + " | ".join(str(row[column]) for column in columns) + " |")

    return "\n".join([header, separator, *body])


def describe_row(row: pd.Series | None) -> str:
    if row is None:
        return "not available"

    parts = []
    for column in ["regime", "window_set", "feature_set", "model", "config_name", "postprocessing"]:
        if column in row.index:
            parts.append(str(row[column]))

    return (
        f"{' / '.join(parts)}: "
        f"ROC-AUC={format_metric(row.get('roc_auc_mean', np.nan))}, "
        f"F1@0.5={format_metric(row.get('f1_mean', np.nan))}, "
        f"sensitivity={format_metric(row.get('sensitivity_mean', np.nan))}, "
        f"specificity={format_metric(row.get('specificity_mean', np.nan))}"
    )


def best_by_metric(table: pd.DataFrame, metric: str) -> pd.Series | None:
    if table.empty or metric not in table.columns:
        return None

    ranked = table.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if ranked.empty:
        return None

    return ranked.iloc[0]


def read_required_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Required table not found: {path}")

    return pd.read_csv(path)


def build_final_table(
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
) -> pd.DataFrame:
    max_f1 = thresholds[thresholds["selection_rule"] == "max_f1"][
        ["experiment", "threshold", "f1"]
    ].rename(columns={"threshold": "max_f1_threshold", "f1": "max_f1"})
    sens70 = thresholds[
        thresholds["selection_rule"] == "sensitivity_ge_0.70_max_specificity"
    ][["experiment", "threshold", "specificity"]].rename(
        columns={
            "threshold": "sens70_threshold",
            "specificity": "sens70_specificity",
        }
    )

    final = summary.merge(max_f1, on="experiment", how="left")
    final = final.merge(sens70, on="experiment", how="left")
    final = final.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)
    final.insert(0, "rank", np.arange(1, len(final) + 1))
    return final


def best_rows_by_group(summary: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for value, group in summary.groupby(group_column, sort=True):
        best = best_by_metric(group, "roc_auc_mean")
        if best is not None:
            rows.append(best)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def smoothing_delta(summary: pd.DataFrame) -> tuple[float, int]:
    key_columns = [
        "regime",
        "window_set",
        "feature_set",
        "model",
        "config_name",
    ]
    if not set([*key_columns, "postprocessing", "roc_auc_mean"]).issubset(summary.columns):
        return np.nan, 0

    pivot = summary.pivot_table(
        index=key_columns,
        columns="postprocessing",
        values="roc_auc_mean",
        aggfunc="first",
    )
    if "raw" not in pivot.columns or "smoothed" not in pivot.columns:
        return np.nan, 0

    deltas = pivot["smoothed"] - pivot["raw"]
    deltas = deltas.dropna()
    if deltas.empty:
        return np.nan, 0

    return float(deltas.mean()), int(len(deltas))


def threshold_best(thresholds: pd.DataFrame, selection_rule: str, metric: str) -> pd.Series | None:
    subset = thresholds[thresholds["selection_rule"] == selection_rule].copy()
    if subset.empty or metric not in subset.columns:
        return None

    subset = subset.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if subset.empty:
        return None

    return subset.iloc[0]


def threshold_descriptor(row: pd.Series | None) -> str:
    if row is None:
        return "not available"

    return (
        f"{row['experiment']} @ threshold={format_metric(row.get('threshold', np.nan))}: "
        f"F1={format_metric(row.get('f1', np.nan))}, "
        f"sensitivity={format_metric(row.get('sensitivity', np.nan))}, "
        f"specificity={format_metric(row.get('specificity', np.nan))}"
    )


def build_report(
    summary: pd.DataFrame,
    thresholds: pd.DataFrame,
    final_table: pd.DataFrame,
    feature_summary: pd.DataFrame | None,
) -> str:
    best_auc = best_by_metric(summary, "roc_auc_mean")
    best_f1 = best_by_metric(summary, "f1_mean")
    best_tuned_f1 = threshold_best(thresholds, "max_f1", "f1")
    best_sens70 = threshold_best(
        thresholds[thresholds["sensitivity"] >= 0.70],
        "sensitivity_ge_0.70_max_specificity",
        "specificity",
    )

    window_rows = best_rows_by_group(summary, "window_set")
    feature_rows = best_rows_by_group(summary, "feature_set")
    regime_rows = best_rows_by_group(summary, "regime")
    post_rows = best_rows_by_group(summary, "postprocessing")
    mean_smoothing_delta, n_smoothing_pairs = smoothing_delta(summary)

    best_auc_value = float(best_auc["roc_auc_mean"]) if best_auc is not None else np.nan
    best_f1_value = float(best_f1["f1_mean"]) if best_f1 is not None else np.nan
    best_tuned_f1_value = (
        float(best_tuned_f1["f1"]) if best_tuned_f1 is not None and pd.notna(best_tuned_f1["f1"]) else np.nan
    )
    reached_auc_08 = bool(np.isfinite(best_auc_value) and best_auc_value >= 0.80)
    suspiciously_high = bool(np.isfinite(best_auc_value) and best_auc_value >= 0.85)

    feature_summary_text = "Feature summary table is unavailable."
    if feature_summary is not None and not feature_summary.empty:
        row = feature_summary.iloc[0]
        feature_summary_text = (
            f"Segment rows={int(row['n_rows'])}, records={int(row['n_records'])}, "
            f"positive_rate_all={format_metric(row['positive_rate_all'])}, "
            f"positive_rate_sleep_only={format_metric(row['positive_rate_sleep_only'])}, "
            f"features_with_nan_over_20_percent={int(row['features_with_nan_over_20_percent'])}."
        )

    auc_delta_text = (
        "not available"
        if not np.isfinite(best_auc_value)
        else f"{best_auc_value - PREVIOUS_BEST_AUC:+.4f}"
    )
    f1_delta_text = (
        "not available"
        if not np.isfinite(best_f1_value)
        else f"{best_f1_value - PREVIOUS_BEST_F1:+.4f}"
    )
    tuned_delta_text = (
        "not available"
        if not np.isfinite(best_tuned_f1_value)
        else f"{best_tuned_f1_value - PREVIOUS_THRESHOLD_TUNED_F1:+.4f}"
    )

    leakage_note = ""
    if suspiciously_high:
        leakage_note = (
            "\n\nThe best ROC-AUC is above 0.85, so leakage was re-checked in the "
            "training design: record_id, segment_id, time columns, sleep stage, labels, "
            "overlap_seconds, max_event_overlap_seconds, and event_* columns are excluded "
            "from X; validation is grouped by record_id. The high score should still be "
            "treated cautiously and manually inspected."
        )

    if reached_auc_08:
        auc_08_text = "AUC 0.8 was reached in this offline segment-level experiment."
    else:
        auc_08_text = (
            "AUC 0.8 was not reached. This is plausible for UCDDB under subject-level CV: "
            "there are few subjects, record-to-record physiology varies strongly, tabular "
            "segment features still discard raw waveform morphology, and apnea/hypopnea "
            "labels are temporally fuzzy around onset and recovery. A deep raw-signal model "
            "or sequence model is the more realistic route to a large jump."
        )

    report = f"""# Segment-Level Experiment Analysis

## Summary

{feature_summary_text}

Best ROC-AUC: {describe_row(best_auc)}.

Best F1 at threshold 0.5: {describe_row(best_f1)}.

Best tuned F1: {threshold_descriptor(best_tuned_f1)}.

Best sensitivity>=0.70 variant: {threshold_descriptor(best_sens70)}.

Compared with the previous reference, best ROC-AUC delta={auc_delta_text}, best F1@0.5 delta={f1_delta_text}, and best tuned F1 delta={tuned_delta_text}.

## Top Results

{make_markdown_table(final_table.head(10), DISPLAY_COLUMNS)}

## Window Comparison

{make_markdown_table(window_rows, ["window_set", "regime", "feature_set", "model", "config_name", "postprocessing", "roc_auc_mean", "f1_mean"])}

## Feature Set Comparison

{make_markdown_table(feature_rows, ["feature_set", "regime", "window_set", "model", "config_name", "postprocessing", "roc_auc_mean", "f1_mean"])}

## ucddb005 Exclusion

{make_markdown_table(regime_rows, ["regime", "window_set", "feature_set", "model", "config_name", "postprocessing", "roc_auc_mean", "f1_mean"])}

## Smoothing

{make_markdown_table(post_rows, ["postprocessing", "regime", "window_set", "feature_set", "model", "config_name", "roc_auc_mean", "f1_mean"])}

Mean paired ROC-AUC delta for smoothed minus raw over {n_smoothing_pairs} matched experiment settings: {format_metric(mean_smoothing_delta)}.

## Offline Context

The segment feature table includes `spo2_next_30s_*` and `spo2_next_60s_*` features, and smoothing uses neighboring probabilities inside the same record. These operations do not use labels or event metadata, but they use future signal context. They are suitable only for offline retrospective PSG analysis, not real-time detection.

## AUC 0.8

{auc_08_text}{leakage_note}

## Files

- Final CSV table: `{FINAL_TABLE_CSV_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- Final Markdown table: `{FINAL_TABLE_MD_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- Full summary: `{RESULTS_SUMMARY_PATH.relative_to(PROJECT_ROOT).as_posix()}`
- Threshold sweep summary: `{BEST_THRESHOLDS_PATH.relative_to(PROJECT_ROOT).as_posix()}`
"""

    return report


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    summary = read_required_table(RESULTS_SUMMARY_PATH)
    thresholds = read_required_table(BEST_THRESHOLDS_PATH)
    feature_summary = pd.read_csv(FEATURE_SUMMARY_PATH) if FEATURE_SUMMARY_PATH.exists() else None

    final_table = build_final_table(summary, thresholds)
    final_table.to_csv(FINAL_TABLE_CSV_PATH, index=False)

    markdown_table = make_markdown_table(final_table.head(30), DISPLAY_COLUMNS)
    FINAL_TABLE_MD_PATH.write_text(markdown_table + "\n", encoding="utf-8")

    report = build_report(summary, thresholds, final_table, feature_summary)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print("Segment analysis summary")
    best_auc = best_by_metric(summary, "roc_auc_mean")
    best_f1 = best_by_metric(summary, "f1_mean")
    best_tuned_f1 = threshold_best(thresholds, "max_f1", "f1")
    print(f"  Best ROC-AUC: {describe_row(best_auc)}")
    print(f"  Best F1 @0.5: {describe_row(best_f1)}")
    print(f"  Best tuned F1: {threshold_descriptor(best_tuned_f1)}")
    print(f"  Report: {REPORT_PATH}")
    print(f"  Final table: {FINAL_TABLE_CSV_PATH}")


if __name__ == "__main__":
    main()
