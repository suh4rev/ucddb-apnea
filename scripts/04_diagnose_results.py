from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, REPORTS_TABLES_DIR  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"

FEATURES_PATH = DATA_PROCESSED_DIR / "features_all.csv"
RESULTS_BY_FOLD_PATH = REPORTS_TABLES_DIR / "model_results_by_fold.csv"
RESULTS_SUMMARY_PATH = REPORTS_TABLES_DIR / "model_results_summary.csv"
CV_PREDICTIONS_PATH = REPORTS_TABLES_DIR / "cv_predictions.csv"

FOLD_LABEL_DISTRIBUTION_PATH = REPORTS_TABLES_DIR / "fold_label_distribution.csv"
STAGE_LABEL_DISTRIBUTION_PATH = REPORTS_TABLES_DIR / "stage_label_distribution.csv"
THRESHOLD_SWEEP_PATH = REPORTS_TABLES_DIR / "threshold_sweep.csv"
BEST_THRESHOLDS_PATH = REPORTS_TABLES_DIR / "best_thresholds_by_experiment.csv"
TOP_EXPERIMENTS_PATH = REPORTS_TABLES_DIR / "top_experiments_for_report.csv"
DIAGNOSTIC_REPORT_PATH = REPORTS_DIR / "diagnostic_report.md"

THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)

TOP_EXPERIMENT_COLUMNS = [
    "experiment",
    "model",
    "roc_auc_mean",
    "f1_mean",
    "sensitivity_mean",
    "specificity_mean",
    "average_precision_mean",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(f"Required file not found: {path}")


def specificity_from_confusion(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def calculate_binary_metrics(
    y_true: pd.Series,
    y_proba: pd.Series,
    threshold: float,
) -> dict[str, float | int]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity_from_confusion(int(tn), int(fp))),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def build_fold_label_distribution(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (experiment, model, fold), group in predictions.groupby(
        ["experiment", "model", "fold"],
        sort=True,
    ):
        label_binary = group["label_binary"].astype(int)
        n_positive = int((label_binary == 1).sum())
        n_negative = int((label_binary == 0).sum())
        n_valid = int(len(group))
        record_ids = sorted(group["record_id"].astype(str).unique())

        rows.append(
            {
                "experiment": experiment,
                "model": model,
                "fold": fold,
                "n_valid": n_valid,
                "n_positive": n_positive,
                "n_negative": n_negative,
                "positive_rate": n_positive / n_valid if n_valid else 0.0,
                "record_ids": "; ".join(record_ids),
            }
        )

    return pd.DataFrame(rows)


def build_stage_label_distribution(features: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for sleep_stage, group in features.groupby("sleep_stage", dropna=False, sort=True):
        label_binary = group["label_binary"].astype(int)
        n_epochs = int(len(group))
        n_apnea_hypopnea = int((label_binary == 1).sum())
        n_normal = int((label_binary == 0).sum())

        rows.append(
            {
                "sleep_stage": sleep_stage,
                "n_epochs": n_epochs,
                "n_normal": n_normal,
                "n_apnea_hypopnea": n_apnea_hypopnea,
                "positive_rate": n_apnea_hypopnea / n_epochs if n_epochs else 0.0,
            }
        )

    return pd.DataFrame(rows)


def build_threshold_sweep(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (experiment, model), group in predictions.groupby(
        ["experiment", "model"],
        sort=True,
    ):
        y_true = group["label_binary"].astype(int)
        y_proba = group["y_proba"].astype(float)

        for threshold in THRESHOLDS:
            metrics = calculate_binary_metrics(y_true, y_proba, float(threshold))
            rows.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "threshold": float(threshold),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def select_best_thresholds(threshold_sweep: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for (experiment, model), group in threshold_sweep.groupby(
        ["experiment", "model"],
        sort=True,
    ):
        by_f1 = group.sort_values(
            ["f1", "specificity", "sensitivity"],
            ascending=[False, False, False],
        ).iloc[0]

        sensitivity_candidates = group[group["sensitivity"] >= 0.70]
        if sensitivity_candidates.empty:
            sensitivity_row = None
        else:
            sensitivity_row = sensitivity_candidates.sort_values(
                ["specificity", "f1"],
                ascending=[False, False],
            ).iloc[0]

        youden_group = group.copy()
        youden_group["youden_index"] = (
            youden_group["sensitivity"] + youden_group["specificity"] - 1
        )
        by_youden = youden_group.sort_values(
            ["youden_index", "f1"],
            ascending=[False, False],
        ).iloc[0]

        rows.append(make_best_threshold_row(experiment, model, "max_f1", by_f1))

        if sensitivity_row is None:
            rows.append(
                {
                    "experiment": experiment,
                    "model": model,
                    "selection_rule": "sensitivity_ge_0.70_max_specificity",
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
                    "details": "No threshold reached sensitivity >= 0.70.",
                }
            )
        else:
            rows.append(
                make_best_threshold_row(
                    experiment,
                    model,
                    "sensitivity_ge_0.70_max_specificity",
                    sensitivity_row,
                )
            )

        rows.append(make_best_threshold_row(experiment, model, "max_youden", by_youden))

    return pd.DataFrame(rows)


def make_best_threshold_row(
    experiment: str,
    model: str,
    selection_rule: str,
    row: pd.Series,
) -> dict[str, object]:
    return {
        "experiment": experiment,
        "model": model,
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
        "youden_index": float(row["sensitivity"] + row["specificity"] - 1),
        "details": "",
    }


def build_top_experiments(results_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        results_summary[TOP_EXPERIMENT_COLUMNS]
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )


def describe_fold_variability(results_summary: pd.DataFrame) -> str:
    if results_summary.empty or "roc_auc_std" not in results_summary.columns:
        return "Fold variability could not be estimated."

    max_roc_auc_std = float(results_summary["roc_auc_std"].max())
    max_f1_std = float(results_summary["f1_std"].max())

    if max_roc_auc_std >= 0.10 or max_f1_std >= 0.15:
        return (
            "There is strong variability between folds "
            f"(max ROC-AUC std={max_roc_auc_std:.3f}, max F1 std={max_f1_std:.3f})."
        )

    if max_roc_auc_std >= 0.05 or max_f1_std >= 0.10:
        return (
            "There is moderate variability between folds "
            f"(max ROC-AUC std={max_roc_auc_std:.3f}, max F1 std={max_f1_std:.3f})."
        )

    return (
        "Fold variability is relatively low "
        f"(max ROC-AUC std={max_roc_auc_std:.3f}, max F1 std={max_f1_std:.3f})."
    )


def should_consider_sleep_only(stage_distribution: pd.DataFrame) -> str:
    if stage_distribution.empty:
        return "Sleep-stage analysis is unavailable, so sleep_only cannot be assessed."

    positive_rates = stage_distribution["positive_rate"].dropna()
    if positive_rates.empty:
        return "Sleep-stage positive rates are unavailable."

    spread = float(positive_rates.max() - positive_rates.min())
    if spread >= 0.20:
        return (
            "A sleep_only diagnostic baseline is worth considering because "
            f"positive rates vary by sleep_stage with spread={spread:.3f}. "
            "It should be used only as analysis, not as a primary deployment model."
        )

    return (
        "A sleep_only diagnostic baseline is optional; current sleep_stage "
        f"positive-rate spread is {spread:.3f}."
    )


def dataframe_to_markdown(table: pd.DataFrame) -> str:
    if table.empty:
        return "No rows available."

    table = table.copy()
    headers = list(table.columns)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]

    for _, row in table.iterrows():
        values = []
        for column in headers:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")

    return "\n".join(lines)


def write_diagnostic_report(
    report_path: Path,
    results_summary: pd.DataFrame,
    stage_distribution: pd.DataFrame,
    top_experiments: pd.DataFrame,
) -> None:
    best_roc_auc = results_summary.sort_values(
        "roc_auc_mean",
        ascending=False,
    ).iloc[0]
    best_f1 = results_summary.sort_values("f1_mean", ascending=False).iloc[0]
    sleep_stages = [str(value) for value in stage_distribution["sleep_stage"].tolist()]

    lines = [
        "# Diagnostic Report",
        "",
        "This report analyzes already trained baseline models without retraining.",
        "",
        "## Best Experiments",
        "",
        (
            f"- Best by ROC-AUC: `{best_roc_auc['experiment']}` / "
            f"`{best_roc_auc['model']}` with mean ROC-AUC "
            f"{best_roc_auc['roc_auc_mean']:.4f}."
        ),
        (
            f"- Best by F1: `{best_f1['experiment']}` / `{best_f1['model']}` "
            f"with mean F1 {best_f1['f1_mean']:.4f}."
        ),
        "",
        "## Fold Variability",
        "",
        describe_fold_variability(results_summary),
        "",
        "## Sleep Stages",
        "",
        f"Observed sleep_stage values: {', '.join(sleep_stages)}.",
        should_consider_sleep_only(stage_distribution),
        "",
        "## Thresholding",
        "",
        (
            "A fixed threshold of 0.5 may be suboptimal because the positive "
            "class is imbalanced and the project may prefer higher sensitivity "
            "for screening. The threshold sweep tables show alternatives that "
            "maximize F1, Youden index, or sensitivity-constrained specificity."
        ),
        "",
        "## Top Experiments",
        "",
        dataframe_to_markdown(top_experiments.head(10)),
        "",
        "## Output Tables",
        "",
        f"- `{FOLD_LABEL_DISTRIBUTION_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{STAGE_LABEL_DISTRIBUTION_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{THRESHOLD_SWEEP_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{BEST_THRESHOLDS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{TOP_EXPERIMENTS_PATH.relative_to(PROJECT_ROOT)}`",
    ]

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    for path in [
        FEATURES_PATH,
        RESULTS_BY_FOLD_PATH,
        RESULTS_SUMMARY_PATH,
        CV_PREDICTIONS_PATH,
    ]:
        require_file(path)

    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURES_PATH)
    results_summary = pd.read_csv(RESULTS_SUMMARY_PATH)
    predictions = pd.read_csv(CV_PREDICTIONS_PATH)

    fold_distribution = build_fold_label_distribution(predictions)
    stage_distribution = build_stage_label_distribution(features)
    threshold_sweep = build_threshold_sweep(predictions)
    best_thresholds = select_best_thresholds(threshold_sweep)
    top_experiments = build_top_experiments(results_summary)

    fold_distribution.to_csv(FOLD_LABEL_DISTRIBUTION_PATH, index=False)
    stage_distribution.to_csv(STAGE_LABEL_DISTRIBUTION_PATH, index=False)
    threshold_sweep.to_csv(THRESHOLD_SWEEP_PATH, index=False)
    best_thresholds.to_csv(BEST_THRESHOLDS_PATH, index=False)
    top_experiments.to_csv(TOP_EXPERIMENTS_PATH, index=False)
    write_diagnostic_report(
        report_path=DIAGNOSTIC_REPORT_PATH,
        results_summary=results_summary,
        stage_distribution=stage_distribution,
        top_experiments=top_experiments,
    )

    best_roc_auc = top_experiments.iloc[0]
    best_f1 = results_summary.sort_values("f1_mean", ascending=False).iloc[0]

    print("\nDiagnostic summary")
    print(
        f"  Best ROC-AUC: {best_roc_auc['experiment']} / "
        f"{best_roc_auc['model']} = {best_roc_auc['roc_auc_mean']:.4f}"
    )
    print(
        f"  Best F1: {best_f1['experiment']} / "
        f"{best_f1['model']} = {best_f1['f1_mean']:.4f}"
    )
    print(f"  Threshold rows: {len(threshold_sweep)}")
    print(f"  Report: {DIAGNOSTIC_REPORT_PATH}")


if __name__ == "__main__":
    main()
