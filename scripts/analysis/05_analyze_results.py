from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import REPORTS_FIGURES_DIR, REPORTS_TABLES_DIR  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"

SUMMARY_PATH = REPORTS_TABLES_DIR / "improved_model_results_summary.csv"
BY_FOLD_PATH = REPORTS_TABLES_DIR / "improved_model_results_by_fold.csv"
CONFUSION_PATH = REPORTS_TABLES_DIR / "improved_model_confusion_matrices.csv"
PREDICTIONS_PATH = REPORTS_TABLES_DIR / "improved_cv_predictions.csv"

FINAL_RESULTS_CSV_PATH = REPORTS_TABLES_DIR / "final_results_table.csv"
FINAL_RESULTS_MD_PATH = REPORTS_TABLES_DIR / "final_results_table.md"
THRESHOLD_SWEEP_PATH = REPORTS_TABLES_DIR / "threshold_sweep_improved.csv"
BEST_THRESHOLDS_PATH = REPORTS_TABLES_DIR / "best_thresholds_improved.csv"
FINAL_REPORT_PATH = REPORTS_DIR / "final_analysis_report.md"

ROC_FIGURE_PATH = REPORTS_FIGURES_DIR / "roc_curves_final.png"
CONFUSION_FIGURE_PATH = REPORTS_FIGURES_DIR / "confusion_matrix_best.png"
MODALITY_FIGURE_PATH = REPORTS_FIGURES_DIR / "modality_comparison_auc.png"
FUSION_FIGURE_PATH = REPORTS_FIGURES_DIR / "fusion_comparison_auc.png"
BEST_METRICS_FIGURE_PATH = REPORTS_FIGURES_DIR / "f1_sensitivity_specificity_best.png"

THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)

KEY_RESULT_ROWS = [
    ("all_epochs", "base", "flow_only"),
    ("all_epochs", "base", "respiratory_spo2_fusion"),
    ("all_epochs", "base", "late_fusion"),
    ("all_epochs", "enhanced", "flow_only"),
    ("all_epochs", "enhanced", "spo2_only"),
    ("all_epochs", "enhanced", "respiratory_spo2_fusion"),
    ("all_epochs", "enhanced", "late_fusion"),
    ("sleep_only", "enhanced", "spo2_only"),
    ("sleep_only", "enhanced", "respiratory_spo2_fusion"),
    ("sleep_only", "enhanced", "late_fusion"),
]

FINAL_RESULTS_COLUMNS = [
    "regime",
    "feature_variant",
    "experiment",
    "n_features",
    "roc_auc_mean",
    "roc_auc_std",
    "f1_mean",
    "sensitivity_mean",
    "specificity_mean",
    "average_precision_mean",
]

ROC_CURVE_MODELS = [
    ("all_epochs", "base", "late_fusion"),
    ("all_epochs", "enhanced", "late_fusion"),
    ("all_epochs", "enhanced", "respiratory_spo2_fusion"),
    ("sleep_only", "enhanced", "spo2_only"),
    ("sleep_only", "enhanced", "respiratory_spo2_fusion"),
]

CONFUSION_TARGET = ("sleep_only", "enhanced", "spo2_only")

MODALITY_COMPARISON = [
    ("all_epochs", "enhanced", "flow_only"),
    ("all_epochs", "enhanced", "effort_only"),
    ("all_epochs", "enhanced", "spo2_only"),
    ("all_epochs", "enhanced", "respiratory_spo2_fusion"),
    ("all_epochs", "enhanced", "full_core_fusion"),
    ("all_epochs", "enhanced", "late_fusion"),
]

FUSION_COMPARISON = [
    ("all_epochs", "base", "respiratory_spo2_fusion"),
    ("all_epochs", "enhanced", "respiratory_spo2_fusion"),
    ("all_epochs", "base", "late_fusion"),
    ("all_epochs", "enhanced", "late_fusion"),
    ("sleep_only", "enhanced", "spo2_only"),
    ("sleep_only", "enhanced", "late_fusion"),
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Required file not found: {path}")


def label_for_key(regime: str, feature_variant: str, experiment: str) -> str:
    return f"{regime}\n{feature_variant}\n{experiment}"


def filter_summary_row(
    summary: pd.DataFrame,
    key: tuple[str, str, str],
) -> pd.DataFrame:
    regime, feature_variant, experiment = key
    return summary[
        (summary["regime"] == regime)
        & (summary["feature_variant"] == feature_variant)
        & (summary["experiment"] == experiment)
    ]


def build_final_results_table(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for key in KEY_RESULT_ROWS:
        match = filter_summary_row(summary, key)
        if match.empty:
            print(f"Warning: missing final table row: {key}")
            continue
        rows.append(match.iloc[0][FINAL_RESULTS_COLUMNS].to_dict())

    return pd.DataFrame(rows, columns=FINAL_RESULTS_COLUMNS)


def dataframe_to_markdown(table: pd.DataFrame, decimals: int = 3) -> str:
    if table.empty:
        return "No rows available.\n"

    rounded = table.copy()
    numeric_columns = rounded.select_dtypes(include=[np.number]).columns
    rounded[numeric_columns] = rounded[numeric_columns].round(decimals)

    headers = list(rounded.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for _, row in rounded.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.{decimals}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines) + "\n"


def specificity_from_confusion(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def calculate_threshold_metrics(
    y_true: pd.Series,
    y_proba: pd.Series,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = float(recall_score(y_true, y_pred, zero_division=0))
    specificity = float(specificity_from_confusion(int(tn), int(fp)))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "youden_index": sensitivity + specificity - 1,
    }


def build_threshold_sweep(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["regime", "feature_variant", "experiment", "model"]

    for keys, group in predictions.groupby(group_columns, sort=True):
        y_true = group["label_binary"].astype(int)
        y_proba = group["y_proba"].astype(float)
        key_values = dict(zip(group_columns, keys))

        for threshold in THRESHOLDS:
            rows.append(
                {
                    **key_values,
                    "threshold": float(threshold),
                    **calculate_threshold_metrics(y_true, y_proba, float(threshold)),
                }
            )

    return pd.DataFrame(rows)


def best_threshold_row(
    key_values: dict[str, object],
    selection_rule: str,
    row: pd.Series | None,
    details: str = "",
) -> dict[str, object]:
    if row is None:
        return {
            **key_values,
            "selection_rule": selection_rule,
            "threshold": np.nan,
            "accuracy": np.nan,
            "precision": np.nan,
            "sensitivity": np.nan,
            "specificity": np.nan,
            "f1": np.nan,
            "tn": np.nan,
            "fp": np.nan,
            "fn": np.nan,
            "tp": np.nan,
            "youden_index": np.nan,
            "details": details,
        }

    return {
        **key_values,
        "selection_rule": selection_rule,
        "threshold": float(row["threshold"]),
        "accuracy": float(row["accuracy"]),
        "precision": float(row["precision"]),
        "sensitivity": float(row["sensitivity"]),
        "specificity": float(row["specificity"]),
        "f1": float(row["f1"]),
        "tn": int(row["tn"]),
        "fp": int(row["fp"]),
        "fn": int(row["fn"]),
        "tp": int(row["tp"]),
        "youden_index": float(row["youden_index"]),
        "details": details,
    }


def build_best_thresholds(threshold_sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = ["regime", "feature_variant", "experiment", "model"]

    for keys, group in threshold_sweep.groupby(group_columns, sort=True):
        key_values = dict(zip(group_columns, keys))

        by_f1 = group.sort_values(
            ["f1", "specificity", "sensitivity"],
            ascending=[False, False, False],
        ).iloc[0]
        rows.append(best_threshold_row(key_values, "max_f1", by_f1))

        sensitivity_candidates = group[group["sensitivity"] >= 0.70]
        if sensitivity_candidates.empty:
            rows.append(
                best_threshold_row(
                    key_values,
                    "sensitivity_ge_0.70_max_specificity",
                    None,
                    "No threshold reached sensitivity >= 0.70.",
                )
            )
        else:
            sensitivity_row = sensitivity_candidates.sort_values(
                ["specificity", "f1"],
                ascending=[False, False],
            ).iloc[0]
            rows.append(
                best_threshold_row(
                    key_values,
                    "sensitivity_ge_0.70_max_specificity",
                    sensitivity_row,
                )
            )

        by_youden = group.sort_values(
            ["youden_index", "f1"],
            ascending=[False, False],
        ).iloc[0]
        rows.append(best_threshold_row(key_values, "max_youden", by_youden))

    return pd.DataFrame(rows)


def filter_predictions(
    predictions: pd.DataFrame,
    key: tuple[str, str, str],
) -> pd.DataFrame:
    regime, feature_variant, experiment = key
    return predictions[
        (predictions["regime"] == regime)
        & (predictions["feature_variant"] == feature_variant)
        & (predictions["experiment"] == experiment)
    ]


def plot_roc_curves(predictions: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(9, 7))

    for key in ROC_CURVE_MODELS:
        subset = filter_predictions(predictions, key)
        if subset.empty or subset["label_binary"].nunique() < 2:
            print(f"Warning: cannot plot ROC for {key}")
            continue

        y_true = subset["label_binary"].astype(int)
        y_proba = subset["y_proba"].astype(float)
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc_value = roc_auc_score(y_true, y_proba)
        plt.plot(fpr, tpr, label=f"{' / '.join(key)} (AUC={auc_value:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Random")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Final Models")
    plt.legend(fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_confusion_matrix(predictions: pd.DataFrame, output_path: Path) -> None:
    subset = filter_predictions(predictions, CONFUSION_TARGET)
    if subset.empty:
        raise ValueError(f"No predictions found for confusion target: {CONFUSION_TARGET}")

    y_true = subset["label_binary"].astype(int)
    y_pred = (subset["y_proba"].astype(float) >= 0.5).astype(int)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.title("Confusion Matrix: sleep_only / enhanced / spo2_only")
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.xticks([0, 1], ["normal", "apnea_hypopnea"])
    plt.yticks([0, 1], ["normal", "apnea_hypopnea"])

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def values_for_keys(
    summary: pd.DataFrame,
    keys: list[tuple[str, str, str]],
    value_column: str,
) -> tuple[list[str], list[float]]:
    labels = []
    values = []

    for key in keys:
        match = filter_summary_row(summary, key)
        if match.empty:
            print(f"Warning: missing bar-chart row: {key}")
            continue
        labels.append(label_for_key(*key))
        values.append(float(match.iloc[0][value_column]))

    return labels, values


def plot_auc_bar(
    summary: pd.DataFrame,
    keys: list[tuple[str, str, str]],
    output_path: Path,
    title: str,
) -> None:
    labels, values = values_for_keys(summary, keys, "roc_auc_mean")

    plt.figure(figsize=(10, 6))
    positions = np.arange(len(labels))
    plt.bar(positions, values)
    plt.xticks(positions, labels, rotation=35, ha="right")
    plt.ylabel("Mean ROC-AUC")
    plt.title(title)
    plt.ylim(0, max(values) + 0.08 if values else 1)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_top5_metrics(summary: pd.DataFrame, output_path: Path) -> None:
    top5 = summary.sort_values("roc_auc_mean", ascending=False).head(5).copy()
    labels = [
        label_for_key(row["regime"], row["feature_variant"], row["experiment"])
        for _, row in top5.iterrows()
    ]
    positions = np.arange(len(top5))
    width = 0.25

    plt.figure(figsize=(11, 6))
    plt.bar(positions - width, top5["f1_mean"], width, label="F1")
    plt.bar(positions, top5["sensitivity_mean"], width, label="Sensitivity")
    plt.bar(positions + width, top5["specificity_mean"], width, label="Specificity")
    plt.xticks(positions, labels, rotation=35, ha="right")
    plt.ylabel("Metric value")
    plt.title("Top-5 Models by ROC-AUC: F1, Sensitivity, Specificity")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def write_final_report(
    summary: pd.DataFrame,
    by_fold: pd.DataFrame,
    final_results: pd.DataFrame,
    output_path: Path,
) -> None:
    best_auc = summary.sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_f1 = summary.sort_values("f1_mean", ascending=False).iloc[0]

    base_late = filter_summary_row(summary, ("all_epochs", "base", "late_fusion"))
    enhanced_late = filter_summary_row(summary, ("all_epochs", "enhanced", "late_fusion"))
    baseline_delta_text = "Baseline vs enhanced comparison is unavailable."
    if not base_late.empty and not enhanced_late.empty:
        delta_auc = float(enhanced_late.iloc[0]["roc_auc_mean"] - base_late.iloc[0]["roc_auc_mean"])
        delta_f1 = float(enhanced_late.iloc[0]["f1_mean"] - base_late.iloc[0]["f1_mean"])
        baseline_delta_text = (
            "For all-epoch late fusion, enhanced features changed ROC-AUC by "
            f"{delta_auc:+.3f} and F1 by {delta_f1:+.3f} compared with base features."
        )

    enhanced_all = summary[
        (summary["regime"] == "all_epochs")
        & (summary["feature_variant"] == "enhanced")
    ].sort_values("roc_auc_mean", ascending=False)
    modality_text = "Modality comparison is unavailable."
    if not enhanced_all.empty:
        top_modality = enhanced_all.iloc[0]
        modality_text = (
            "Among enhanced all-epoch experiments, the strongest ROC-AUC was "
            f"`{top_modality['experiment']}` with {top_modality['roc_auc_mean']:.3f}."
        )

    fusion_rows = summary[summary["experiment"].isin(["respiratory_spo2_fusion", "late_fusion"])]
    fusion_text = "Fusion comparison is unavailable."
    if not fusion_rows.empty:
        top_fusion = fusion_rows.sort_values("roc_auc_mean", ascending=False).iloc[0]
        fusion_text = (
            "The strongest fusion result was "
            f"`{top_fusion['regime']} / {top_fusion['feature_variant']} / "
            f"{top_fusion['experiment']}` with ROC-AUC {top_fusion['roc_auc_mean']:.3f}."
        )

    max_auc_std = float(by_fold.groupby(["regime", "feature_variant", "experiment"])["roc_auc"].std().max())

    lines = [
        "# Final Analysis Report",
        "",
        "This report summarizes improved-model results without retraining.",
        "",
        "## Best Models",
        "",
        (
            f"- Best by AUC-ROC: `{best_auc['regime']} / {best_auc['feature_variant']} / "
            f"{best_auc['experiment']}` with mean ROC-AUC {best_auc['roc_auc_mean']:.3f}."
        ),
        (
            f"- Best by F1: `{best_f1['regime']} / {best_f1['feature_variant']} / "
            f"{best_f1['experiment']}` with mean F1 {best_f1['f1_mean']:.3f}."
        ),
        "",
        "## Baseline vs Enhanced",
        "",
        baseline_delta_text,
        "",
        "## Modality Findings",
        "",
        modality_text,
        "",
        "## Fusion Findings",
        "",
        fusion_text,
        "",
        "## Thresholding",
        "",
        (
            "A threshold of 0.5 is not necessarily optimal under class imbalance. "
            "The threshold sweep tables include alternatives that maximize F1, "
            "Youden index, or specificity under sensitivity >= 0.70."
        ),
        "",
        "## Limitations",
        "",
        (
            "The analysis uses 25 UCDDB records with subject-level cross-validation. "
            f"Fold variability is moderate: the maximum ROC-AUC fold std across "
            f"experiments is {max_auc_std:.3f}. Results should therefore be reported "
            "as comparative evidence rather than final clinical performance."
        ),
        "",
        "## Final Table",
        "",
        dataframe_to_markdown(final_results, decimals=3),
    ]

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    for path in [SUMMARY_PATH, BY_FOLD_PATH, CONFUSION_PATH, PREDICTIONS_PATH]:
        require_file(path)

    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    summary = pd.read_csv(SUMMARY_PATH)
    by_fold = pd.read_csv(BY_FOLD_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    final_results = build_final_results_table(summary)
    final_results.to_csv(FINAL_RESULTS_CSV_PATH, index=False)
    FINAL_RESULTS_MD_PATH.write_text(
        dataframe_to_markdown(final_results, decimals=3),
        encoding="utf-8",
    )

    threshold_sweep = build_threshold_sweep(predictions)
    best_thresholds = build_best_thresholds(threshold_sweep)
    threshold_sweep.to_csv(THRESHOLD_SWEEP_PATH, index=False)
    best_thresholds.to_csv(BEST_THRESHOLDS_PATH, index=False)

    plot_roc_curves(predictions, ROC_FIGURE_PATH)
    plot_confusion_matrix(predictions, CONFUSION_FIGURE_PATH)
    plot_auc_bar(
        summary,
        MODALITY_COMPARISON,
        MODALITY_FIGURE_PATH,
        "Enhanced All-Epoch Modality Comparison",
    )
    plot_auc_bar(
        summary,
        FUSION_COMPARISON,
        FUSION_FIGURE_PATH,
        "Fusion Strategy Comparison",
    )
    plot_top5_metrics(summary, BEST_METRICS_FIGURE_PATH)
    write_final_report(summary, by_fold, final_results, FINAL_REPORT_PATH)

    best_auc = summary.sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_f1 = summary.sort_values("f1_mean", ascending=False).iloc[0]

    print("\nFinal analysis summary")
    print(
        f"  Best AUC-ROC: {best_auc['regime']} / {best_auc['feature_variant']} / "
        f"{best_auc['experiment']} = {best_auc['roc_auc_mean']:.4f}"
    )
    print(
        f"  Best F1: {best_f1['regime']} / {best_f1['feature_variant']} / "
        f"{best_f1['experiment']} = {best_f1['f1_mean']:.4f}"
    )
    print(f"  Final table: {FINAL_RESULTS_CSV_PATH}")
    print(f"  Final report: {FINAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
