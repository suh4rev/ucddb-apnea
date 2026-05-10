from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import REPORTS_FIGURES_DIR, REPORTS_TABLES_DIR  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"

EPOCHS_OVERALL_PATH = REPORTS_TABLES_DIR / "epochs_summary_overall.csv"
MODEL_READY_SUMMARY_PATH = REPORTS_TABLES_DIR / "model_ready_feature_summary.csv"
FINAL_RESULTS_PATH = REPORTS_TABLES_DIR / "final_results_table.csv"
IMPROVED_BEST_PATH = REPORTS_TABLES_DIR / "improved_best_results.csv"
TEMPORAL_BEST_PATH = REPORTS_TABLES_DIR / "temporal_ensemble_best_results.csv"
TEMPORAL_SUMMARY_PATH = REPORTS_TABLES_DIR / "temporal_ensemble_results_summary.csv"
TEMPORAL_THRESHOLDS_PATH = REPORTS_TABLES_DIR / "temporal_ensemble_best_thresholds.csv"
ADVANCED_BEST_PATH = REPORTS_TABLES_DIR / "advanced_best_results.csv"
SEGMENT_REDUCED_BEST_PATH = REPORTS_TABLES_DIR / "segment_reduced_best_results.csv"
TEMPORAL_REPORT_PATH = REPORTS_DIR / "temporal_ensemble_report.md"
SEGMENT_REPORT_PATH = REPORTS_DIR / "segment_reduced_analysis_report.md"
AUDIT_REPORT_PATH = REPORTS_DIR / "pipeline_audit_report.md"

OUTPUT_INDEX_PATH = REPORTS_DIR / "final_artifact_index.md"
MASTER_RESULTS_CSV_PATH = REPORTS_TABLES_DIR / "final_master_results_table.csv"
MASTER_RESULTS_MD_PATH = REPORTS_TABLES_DIR / "final_master_results_table.md"
DATASET_SUMMARY_PATH = REPORTS_TABLES_DIR / "final_dataset_summary.csv"
PRACTICAL_METRICS_PATH = REPORTS_TABLES_DIR / "final_practical_process_metrics.csv"
NEGATIVE_EXPERIMENTS_PATH = REPORTS_TABLES_DIR / "final_negative_experiments_summary.csv"

ROC_FIGURE_PATH = REPORTS_FIGURES_DIR / "final_roc_curves_with_temporal.png"
CONFUSION_FIGURE_PATH = REPORTS_FIGURES_DIR / "final_confusion_matrix_temporal_best.png"
MODEL_COMPARISON_FIGURE_PATH = REPORTS_FIGURES_DIR / "final_model_comparison_auc.png"
THRESHOLD_TRADEOFF_FIGURE_PATH = REPORTS_FIGURES_DIR / "final_threshold_tradeoff.png"

PREVIOUS_BEST_AUC = 0.6725
MASTER_COLUMNS = [
    "approach",
    "regime",
    "model",
    "postprocessing",
    "roc_auc",
    "f1",
    "sensitivity",
    "specificity",
    "average_precision",
    "comment",
]


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise SystemExit(f"Required input file not found: {path}")
        return pd.DataFrame()

    return pd.read_csv(path)


def relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def first_row(table: pd.DataFrame) -> pd.Series:
    if table.empty:
        raise ValueError("Expected non-empty table.")

    return table.iloc[0]


def best_by_auc(table: pd.DataFrame) -> pd.Series | None:
    if table.empty or "roc_auc_mean" not in table.columns:
        return None

    ranked = table.dropna(subset=["roc_auc_mean"]).sort_values(
        "roc_auc_mean",
        ascending=False,
    )
    if ranked.empty:
        return None

    return ranked.iloc[0]


def select_first(table: pd.DataFrame, mask: pd.Series) -> pd.Series | None:
    subset = table[mask].copy()
    if subset.empty:
        return None

    if "roc_auc_mean" in subset.columns:
        subset = subset.sort_values("roc_auc_mean", ascending=False)

    return subset.iloc[0]


def metric_value(row: pd.Series | None, column: str) -> float:
    if row is None or column not in row.index or pd.isna(row[column]):
        return np.nan

    return float(row[column])


def make_result_row(
    approach: str,
    row: pd.Series | None,
    model: str,
    postprocessing: str,
    comment: str,
) -> dict[str, object]:
    return {
        "approach": approach,
        "regime": row.get("regime", "") if row is not None else "",
        "model": model,
        "postprocessing": postprocessing,
        "roc_auc": metric_value(row, "roc_auc_mean"),
        "f1": metric_value(row, "f1_mean"),
        "sensitivity": metric_value(row, "sensitivity_mean"),
        "specificity": metric_value(row, "specificity_mean"),
        "average_precision": metric_value(row, "average_precision_mean"),
        "comment": comment,
    }


def format_metric(value: object) -> str:
    if pd.isna(value):
        return "NA"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    if table.empty:
        return "_No rows available._\n"

    display = table[columns].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(format_metric)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *rows]) + "\n"


def build_dataset_summary() -> pd.DataFrame:
    epochs = first_row(read_csv(EPOCHS_OVERALL_PATH))
    model_ready = first_row(read_csv(MODEL_READY_SUMMARY_PATH))

    n_sleep_epochs = int(model_ready["n_sleep_epochs"])
    sleep_positive_rate = float(model_ready["positive_rate_sleep_only"])
    n_sleep_positive = int(round(n_sleep_epochs * sleep_positive_rate))

    summary = {
        "n_records": int(epochs["n_records"]),
        "n_epochs": int(epochs["n_epochs"]),
        "n_normal": int(epochs["n_normal"]),
        "n_apnea_hypopnea": int(epochs["n_apnea_hypopnea"]),
        "positive_rate": float(epochs["positive_rate"]),
        "n_sleep_epochs": n_sleep_epochs,
        "n_sleep_positive": n_sleep_positive,
        "sleep_positive_rate": sleep_positive_rate,
    }
    return pd.DataFrame([summary])


def build_master_results_table() -> pd.DataFrame:
    final_results = read_csv(FINAL_RESULTS_PATH)
    improved = read_csv(IMPROVED_BEST_PATH)
    temporal = read_csv(TEMPORAL_SUMMARY_PATH)

    baseline_flow = select_first(
        final_results,
        (final_results["feature_variant"] == "base")
        & (final_results["experiment"] == "flow_only"),
    )
    baseline_resp_spo2 = select_first(
        final_results,
        (final_results["feature_variant"] == "base")
        & (final_results["experiment"] == "respiratory_spo2_fusion"),
    )
    improved_spo2 = select_first(improved, improved["experiment"] == "spo2_only")
    improved_resp_spo2 = select_first(
        improved,
        improved["experiment"] == "respiratory_spo2_fusion",
    )
    improved_late = select_first(improved, improved["experiment"] == "late_fusion")
    temporal_raw = select_first(
        temporal,
        (temporal["model"] == "MeanEnsemble") & (temporal["postprocessing"] == "raw"),
    )
    temporal_causal = select_first(
        temporal,
        (temporal["model"] == "MeanEnsemble")
        & (temporal["postprocessing"] == "rolling_mean_causal"),
    )
    temporal_centered = select_first(
        temporal,
        (temporal["model"] == "MeanEnsemble")
        & (temporal["postprocessing"] == "rolling_mean_centered"),
    )

    rows = [
        make_result_row(
            "baseline flow_only",
            baseline_flow,
            "XGBoost",
            "none",
            "Single Flow modality baseline.",
        ),
        make_result_row(
            "baseline respiratory_spo2_fusion",
            baseline_resp_spo2,
            "XGBoost",
            "none",
            "Baseline respiratory + SpO2 fusion.",
        ),
        make_result_row(
            "improved spo2_only",
            improved_spo2,
            "XGBoost",
            "enhanced epoch context",
            "Best pre-temporal single-modality result.",
        ),
        make_result_row(
            "improved respiratory_spo2_fusion",
            improved_resp_spo2,
            "XGBoost",
            "enhanced epoch context",
            "Best improved classical multimodal fusion.",
        ),
        make_result_row(
            "improved late_fusion",
            improved_late,
            "XGBoost",
            "late fusion",
            "Late fusion of modality-specific models.",
        ),
        make_result_row(
            "temporal ensemble raw",
            temporal_raw,
            "MeanEnsemble",
            "none",
            "SpO2 XGBoost + Flow/SpO2 HistGradientBoosting.",
        ),
        make_result_row(
            "temporal ensemble causal smoothing",
            temporal_causal,
            "MeanEnsemble",
            "rolling mean causal",
            "Label-free temporal smoothing without future probabilities.",
        ),
        make_result_row(
            "temporal ensemble centered smoothing",
            temporal_centered,
            "MeanEnsemble",
            "rolling mean centered",
            "Offline retrospective smoothing; uses future probabilities.",
        ),
    ]
    return pd.DataFrame(rows, columns=MASTER_COLUMNS)


def select_temporal_best_threshold() -> pd.Series:
    thresholds = read_csv(TEMPORAL_THRESHOLDS_PATH)
    mask = (
        (thresholds["feature_set"] == "spo2_flow_spo2")
        & (thresholds["model"] == "MeanEnsemble")
        & (thresholds["postprocessing"] == "rolling_mean_centered")
        & (thresholds["smoothing_window_epochs"] == 61)
        & (thresholds["smoothing_centered"] == True)
        & (thresholds["selection_rule"] == "max_f1")
    )
    row = select_first(thresholds, mask)
    if row is not None:
        return row

    fallback = thresholds[thresholds["selection_rule"] == "max_f1"].sort_values(
        "f1",
        ascending=False,
    )
    return fallback.iloc[0]


def build_practical_metrics(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    row = select_temporal_best_threshold()
    n_sleep_epochs = int(dataset_summary["n_sleep_epochs"].iloc[0])
    tp = int(row["tp"])
    fp = int(row["fp"])
    fn = int(row["fn"])
    tn = int(row["tn"])
    n_predicted_positive = tp + fp

    metrics = {
        "threshold": float(row["threshold"]),
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "sensitivity": float(row["sensitivity"]),
        "specificity": float(row["specificity"]),
        "precision": float(row["precision"]),
        "f1": float(row["f1"]),
        "n_sleep_epochs": n_sleep_epochs,
        "n_predicted_positive": n_predicted_positive,
        "predicted_positive_rate": n_predicted_positive / n_sleep_epochs
        if n_sleep_epochs
        else np.nan,
        "covered_positive_epochs_rate": tp / (tp + fn) if (tp + fn) else np.nan,
    }
    return pd.DataFrame([metrics])


def build_negative_experiments_summary() -> pd.DataFrame:
    advanced_best = best_by_auc(read_csv(ADVANCED_BEST_PATH, required=False))
    segment_best = best_by_auc(read_csv(SEGMENT_REDUCED_BEST_PATH, required=False))

    rows = []
    if advanced_best is not None:
        rows.append(
            {
                "experiment_group": "advanced contextual features",
                "best_experiment": advanced_best.get("experiment", ""),
                "best_regime": advanced_best.get("regime", ""),
                "best_model": advanced_best.get("model", ""),
                "best_roc_auc": float(advanced_best["roc_auc_mean"]),
                "best_f1": float(advanced_best["f1_mean"]),
                "comparison_reference_auc": PREVIOUS_BEST_AUC,
                "conclusion": "Did not improve the improved SpO2 baseline or temporal ensemble.",
            }
        )
    if segment_best is not None:
        rows.append(
            {
                "experiment_group": "segment-level reduced experiment",
                "best_experiment": segment_best.get("feature_set", ""),
                "best_regime": segment_best.get("regime", ""),
                "best_model": segment_best.get("model", ""),
                "best_roc_auc": float(segment_best["roc_auc_mean"]),
                "best_f1": float(segment_best["f1_mean"]),
                "comparison_reference_auc": PREVIOUS_BEST_AUC,
                "conclusion": "Did not improve the improved SpO2 baseline or temporal ensemble.",
            }
        )

    return pd.DataFrame(rows)


def save_master_markdown(master: pd.DataFrame) -> None:
    MASTER_RESULTS_MD_PATH.write_text(
        markdown_table(master, MASTER_COLUMNS),
        encoding="utf-8",
    )


def plot_auc_bar(
    table: pd.DataFrame,
    path: Path,
    title: str,
    label_column: str = "approach",
) -> None:
    plot_table = table.dropna(subset=["roc_auc"]).copy()
    if plot_table.empty:
        return

    labels = plot_table[label_column].astype(str).tolist()
    values = plot_table["roc_auc"].astype(float).tolist()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(labels, values)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(title)
    ax.set_ylim(0.0, max(0.85, max(values) + 0.05))
    ax.tick_params(axis="x", rotation=35)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_final_roc_placeholder(master: pd.DataFrame) -> str:
    selected = master[
        master["approach"].isin(
            [
                "improved spo2_only",
                "improved respiratory_spo2_fusion",
                "temporal ensemble raw",
                "temporal ensemble causal smoothing",
                "temporal ensemble centered smoothing",
            ]
        )
    ].copy()
    plot_auc_bar(
        selected,
        ROC_FIGURE_PATH,
        "ROC-AUC comparison (ROC curves unavailable for temporal ensemble)",
    )
    return (
        "Temporal ensemble OOF probability table was not saved, so "
        "`final_roc_curves_with_temporal.png` is a ROC-AUC bar chart rather than ROC curves."
    )


def plot_confusion_matrix(practical_metrics: pd.DataFrame) -> None:
    row = first_row(practical_metrics)
    matrix = np.array(
        [
            [int(row["TN"]), int(row["FP"])],
            [int(row["FN"]), int(row["TP"])],
        ]
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    image = ax.imshow(matrix)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title("Temporal ensemble confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks([0, 1], labels=["normal", "apnea_hypopnea"])
    ax.set_yticks([0, 1], labels=["normal", "apnea_hypopnea"])

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(CONFUSION_FIGURE_PATH, dpi=200)
    plt.close(fig)


def plot_model_comparison(master: pd.DataFrame) -> None:
    plot_auc_bar(
        master,
        MODEL_COMPARISON_FIGURE_PATH,
        "Final model comparison by ROC-AUC",
    )


def plot_threshold_tradeoff(
    temporal_summary: pd.DataFrame,
    temporal_thresholds: pd.DataFrame,
) -> None:
    best_summary = select_first(
        temporal_summary,
        (temporal_summary["feature_set"] == "spo2_flow_spo2")
        & (temporal_summary["model"] == "MeanEnsemble")
        & (temporal_summary["postprocessing"] == "rolling_mean_centered")
        & (temporal_summary["smoothing_window_epochs"] == 61)
        & (temporal_summary["smoothing_centered"] == True),
    )
    best_threshold = select_temporal_best_threshold()

    rows = []
    if best_summary is not None:
        rows.append(
            {
                "threshold": "0.50",
                "f1": float(best_summary["f1_mean"]),
                "sensitivity": float(best_summary["sensitivity_mean"]),
                "specificity": float(best_summary["specificity_mean"]),
            }
        )
    rows.append(
        {
            "threshold": format_metric(best_threshold["threshold"]),
            "f1": float(best_threshold["f1"]),
            "sensitivity": float(best_threshold["sensitivity"]),
            "specificity": float(best_threshold["specificity"]),
        }
    )
    tradeoff = pd.DataFrame(rows)

    x = np.arange(len(tradeoff))
    width = 0.25
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x - width, tradeoff["f1"], width, label="F1")
    ax.bar(x, tradeoff["sensitivity"], width, label="Sensitivity")
    ax.bar(x + width, tradeoff["specificity"], width, label="Specificity")
    ax.set_xticks(x, labels=[f"threshold {value}" for value in tradeoff["threshold"]])
    ax.set_ylim(0, 1)
    ax.set_title("Temporal ensemble threshold tradeoff")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(THRESHOLD_TRADEOFF_FIGURE_PATH, dpi=200)
    plt.close(fig)


def build_index(
    dataset_summary: pd.DataFrame,
    master: pd.DataFrame,
    negative: pd.DataFrame,
    roc_note: str,
) -> str:
    best_final = master.dropna(subset=["roc_auc"]).sort_values(
        "roc_auc",
        ascending=False,
    ).iloc[0]
    best_classical_multimodal = master[
        master["approach"] == "improved respiratory_spo2_fusion"
    ].iloc[0]
    temporal_centered = master[
        master["approach"] == "temporal ensemble centered smoothing"
    ].iloc[0]
    temporal_causal = master[
        master["approach"] == "temporal ensemble causal smoothing"
    ].iloc[0]
    ds = first_row(dataset_summary)

    negative_lines = []
    for _, row in negative.iterrows():
        negative_lines.append(
            f"- {row['experiment_group']}: best ROC-AUC={format_metric(row['best_roc_auc'])}; "
            f"{row['conclusion']}"
        )
    negative_text = "\n".join(negative_lines) if negative_lines else "- No negative experiment summary available."

    return f"""# Final Artifact Index

## Summary

Dataset: {int(ds['n_records'])} records, {int(ds['n_epochs'])} 30-second epochs, positive rate {format_metric(ds['positive_rate'])}. Sleep-only subset: {int(ds['n_sleep_epochs'])} epochs, positive rate {format_metric(ds['sleep_positive_rate'])}.

Final best result: {best_final['approach']} with ROC-AUC={format_metric(best_final['roc_auc'])}, F1={format_metric(best_final['f1'])}, sensitivity={format_metric(best_final['sensitivity'])}, specificity={format_metric(best_final['specificity'])}.

Best classical multimodal result: improved respiratory_spo2_fusion with ROC-AUC={format_metric(best_classical_multimodal['roc_auc'])}.

Best temporal multimodal/offline result: centered temporal ensemble with ROC-AUC={format_metric(temporal_centered['roc_auc'])}. The causal temporal ensemble reached ROC-AUC={format_metric(temporal_causal['roc_auc'])}.

Advanced and segment-level checks:

{negative_text}

## Figures

- `{relative(ROC_FIGURE_PATH)}`: {roc_note}
- `{relative(CONFUSION_FIGURE_PATH)}`: confusion matrix for temporal ensemble threshold 0.45.
- `{relative(MODEL_COMPARISON_FIGURE_PATH)}`: ROC-AUC bar chart for key models.
- `{relative(THRESHOLD_TRADEOFF_FIGURE_PATH)}`: threshold 0.50 vs 0.45 tradeoff for temporal ensemble.

## Tables

- `{relative(DATASET_SUMMARY_PATH)}`: final dataset summary.
- `{relative(MASTER_RESULTS_CSV_PATH)}` and `{relative(MASTER_RESULTS_MD_PATH)}`: compact final results table for the thesis.
- `{relative(PRACTICAL_METRICS_PATH)}`: process metrics for temporal ensemble threshold 0.45.
- `{relative(NEGATIVE_EXPERIMENTS_PATH)}`: experiments that did not improve the final baseline.

## Source Reports

- `{relative(AUDIT_REPORT_PATH)}`: pipeline audit and leakage checks.
- `{relative(TEMPORAL_REPORT_PATH)}`: temporal ensemble analysis.
- `{relative(SEGMENT_REPORT_PATH)}`: reduced segment-level analysis.

## Thesis Usage Note

Use the centered temporal ensemble only as an offline retrospective PSG analysis result, because centered smoothing uses future probabilities inside a record. For a stricter non-future post-processing variant, cite the causal temporal ensemble.
"""


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    dataset_summary = build_dataset_summary()
    master = build_master_results_table()
    practical_metrics = build_practical_metrics(dataset_summary)
    negative = build_negative_experiments_summary()

    dataset_summary.to_csv(DATASET_SUMMARY_PATH, index=False)
    master.to_csv(MASTER_RESULTS_CSV_PATH, index=False)
    save_master_markdown(master)
    practical_metrics.to_csv(PRACTICAL_METRICS_PATH, index=False)
    negative.to_csv(NEGATIVE_EXPERIMENTS_PATH, index=False)

    roc_note = plot_final_roc_placeholder(master)
    plot_confusion_matrix(practical_metrics)
    plot_model_comparison(master)
    plot_threshold_tradeoff(
        read_csv(TEMPORAL_SUMMARY_PATH),
        read_csv(TEMPORAL_THRESHOLDS_PATH),
    )

    OUTPUT_INDEX_PATH.write_text(
        build_index(dataset_summary, master, negative, roc_note),
        encoding="utf-8",
    )

    best = master.dropna(subset=["roc_auc"]).sort_values("roc_auc", ascending=False).iloc[0]
    print("Final artifact package created")
    print(f"  Best result: {best['approach']} ROC-AUC={best['roc_auc']:.4f}")
    print(f"  Index: {OUTPUT_INDEX_PATH}")
    print(f"  Master table: {MASTER_RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()
