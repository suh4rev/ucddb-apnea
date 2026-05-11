from __future__ import annotations

import sys
from pathlib import Path

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
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import REPORTS_TABLES_DIR  # noqa: E402


CNN_PREDICTIONS_PATH = REPORTS_TABLES_DIR / "cnn_cv_predictions.csv"
RESNET_PREDICTIONS_PATH = REPORTS_TABLES_DIR / "cnn_improved_cv_predictions.csv"

RESULTS_BY_FOLD_PATH = REPORTS_TABLES_DIR / "cnn_dl_improvement_results_by_fold.csv"
RESULTS_SUMMARY_PATH = REPORTS_TABLES_DIR / "cnn_dl_improvement_results_summary.csv"
BEST_THRESHOLDS_PATH = REPORTS_TABLES_DIR / "cnn_dl_improvement_best_thresholds.csv"
CV_PREDICTIONS_PATH = REPORTS_TABLES_DIR / "cnn_dl_improvement_cv_predictions.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "cnn_dl_improvement_report.md"

THRESHOLD = 0.5
THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)

PREVIOUS_CNN_AUC = 0.5953
PREVIOUS_CNN_F1 = 0.3453
PREVIOUS_RESNET_AUC = 0.5998
PREVIOUS_RESNET_F1 = 0.3756
TEMPORAL_ENSEMBLE_AUC = 0.7066
TEMPORAL_ENSEMBLE_F1 = 0.4349

METRIC_COLUMNS = [
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "average_precision",
]
CONFUSION_COLUMNS = ["tn", "fp", "fn", "tp"]
GROUP_COLUMNS = [
    "experiment",
    "components",
    "postprocessing",
    "smoothing_window_epochs",
    "smoothing_centered",
    "offline_retrospective",
]


def require_file(path: Path) -> None:
    if not path.exists():
        raise SystemExit(
            f"Missing required file: {path}. Run the CNN scripts first:\n"
            "  python scripts/04_train_1dcnn_raw_signals.py\n"
            "  python scripts/04_train_1dcnn_improved.py"
        )


def load_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    require_file(CNN_PREDICTIONS_PATH)
    require_file(RESNET_PREDICTIONS_PATH)

    cnn = pd.read_csv(CNN_PREDICTIONS_PATH)
    resnet = pd.read_csv(RESNET_PREDICTIONS_PATH)

    required_cnn = {"record_id", "epoch_id", "label_binary", "input_mode", "fold", "y_proba"}
    required_resnet = {
        "record_id",
        "epoch_id",
        "label_binary",
        "input_mode",
        "fold",
        "postprocessing",
        "smoothing_window_epochs",
        "y_proba",
    }
    missing_cnn = required_cnn - set(cnn.columns)
    missing_resnet = required_resnet - set(resnet.columns)
    if missing_cnn:
        raise SystemExit(f"Missing columns in CNN predictions: {sorted(missing_cnn)}")
    if missing_resnet:
        raise SystemExit(
            f"Missing columns in ResNet predictions: {sorted(missing_resnet)}"
        )

    cnn["record_id"] = cnn["record_id"].astype(str).str.lower()
    resnet["record_id"] = resnet["record_id"].astype(str).str.lower()
    return cnn, resnet


def smooth_probabilities(
    predictions: pd.DataFrame,
    proba_column: str,
    window: int,
    centered: bool = False,
) -> pd.DataFrame:
    smoothed = predictions.copy()
    smoothed = smoothed.sort_values(["record_id", "epoch_id"]).copy()
    smoothed[proba_column] = (
        smoothed.groupby("record_id", group_keys=False)[proba_column]
        .apply(
            lambda series: series.rolling(
                window=window,
                min_periods=1,
                center=centered,
            ).mean()
        )
        .to_numpy()
    )
    return smoothed


def get_cnn_component(
    cnn: pd.DataFrame,
    input_mode: str,
    component_name: str,
    smoothing_window_epochs: int = 0,
) -> pd.DataFrame:
    component = cnn[cnn["input_mode"] == input_mode][
        ["record_id", "epoch_id", "label_binary", "fold", "y_proba"]
    ].copy()
    component = component.rename(columns={"y_proba": component_name})
    if smoothing_window_epochs > 0:
        component = smooth_probabilities(
            component,
            proba_column=component_name,
            window=smoothing_window_epochs,
            centered=False,
        )
    return component


def get_resnet_component(
    resnet: pd.DataFrame,
    component_name: str,
    smoothing_window_epochs: int = 0,
) -> pd.DataFrame:
    raw = resnet[
        (resnet["postprocessing"] == "raw")
        & (resnet["smoothing_window_epochs"].astype(int) == 0)
    ][["record_id", "epoch_id", "label_binary", "fold", "y_proba"]].copy()
    raw = raw.rename(columns={"y_proba": component_name})
    if smoothing_window_epochs > 0:
        raw = smooth_probabilities(
            raw,
            proba_column=component_name,
            window=smoothing_window_epochs,
            centered=False,
        )
    return raw


def merge_components(components: list[pd.DataFrame], component_names: list[str]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    keys = ["record_id", "epoch_id", "label_binary", "fold"]

    for component in components:
        if merged is None:
            merged = component.copy()
        else:
            merged = merged.merge(component, on=keys, how="inner")

    if merged is None:
        raise ValueError("At least one component is required")
    if len(merged) == 0:
        raise SystemExit("Component merge produced zero rows")

    merged["y_proba"] = merged[component_names].mean(axis=1)
    merged["y_pred"] = (merged["y_proba"] >= THRESHOLD).astype(int)
    return merged[keys + ["y_proba", "y_pred"]]


def build_experiments(cnn: pd.DataFrame, resnet: pd.DataFrame) -> pd.DataFrame:
    specs = [
        {
            "experiment": "resnet_causal_61",
            "components": "resnet_150s",
            "postprocessing": "rolling_mean_causal",
            "smoothing_window_epochs": 61,
            "builders": [
                lambda: get_resnet_component(resnet, "resnet_150s_causal_61", 61),
            ],
            "component_names": ["resnet_150s_causal_61"],
        },
        {
            "experiment": "resnet_causal_121",
            "components": "resnet_150s",
            "postprocessing": "rolling_mean_causal",
            "smoothing_window_epochs": 121,
            "builders": [
                lambda: get_resnet_component(resnet, "resnet_150s_causal_121", 121),
            ],
            "component_names": ["resnet_150s_causal_121"],
        },
        {
            "experiment": "cnn90_resnet_causal_31_mean",
            "components": "cnn_90s_context + resnet_150s",
            "postprocessing": "equal_mean_causal",
            "smoothing_window_epochs": 31,
            "builders": [
                lambda: get_cnn_component(cnn, "cnn_90s_context", "cnn_90s_causal_31", 31),
                lambda: get_resnet_component(resnet, "resnet_150s_causal_31", 31),
            ],
            "component_names": ["cnn_90s_causal_31", "resnet_150s_causal_31"],
        },
        {
            "experiment": "cnn90_resnet_causal_121_mean",
            "components": "cnn_90s_context + resnet_150s",
            "postprocessing": "equal_mean_causal",
            "smoothing_window_epochs": 121,
            "builders": [
                lambda: get_cnn_component(
                    cnn,
                    "cnn_90s_context",
                    "cnn_90s_causal_121",
                    121,
                ),
                lambda: get_resnet_component(resnet, "resnet_150s_causal_121", 121),
            ],
            "component_names": ["cnn_90s_causal_121", "resnet_150s_causal_121"],
        },
        {
            "experiment": "cnn30_resnet_causal_121_mean",
            "components": "cnn_30s + resnet_150s",
            "postprocessing": "equal_mean_causal",
            "smoothing_window_epochs": 121,
            "builders": [
                lambda: get_cnn_component(cnn, "cnn_30s", "cnn_30s_causal_121", 121),
                lambda: get_resnet_component(resnet, "resnet_150s_causal_121", 121),
            ],
            "component_names": ["cnn_30s_causal_121", "resnet_150s_causal_121"],
        },
    ]

    frames: list[pd.DataFrame] = []
    for spec in specs:
        component_frames = [builder() for builder in spec["builders"]]
        predictions = merge_components(component_frames, spec["component_names"])
        predictions["experiment"] = spec["experiment"]
        predictions["components"] = spec["components"]
        predictions["postprocessing"] = spec["postprocessing"]
        predictions["smoothing_window_epochs"] = int(spec["smoothing_window_epochs"])
        predictions["smoothing_centered"] = False
        predictions["offline_retrospective"] = True
        frames.append(predictions)

    return pd.concat(frames, ignore_index=True)


def specificity_from_confusion(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def safe_roc_auc(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(roc_auc_score(y_true, y_proba))
    except ValueError:
        return np.nan


def safe_average_precision(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    try:
        return float(average_precision_score(y_true, y_proba))
    except ValueError:
        return np.nan


def calculate_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = THRESHOLD,
) -> dict[str, float | int]:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity_from_confusion(int(tn), int(fp))),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_proba),
        "average_precision": safe_average_precision(y_true, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def calculate_results_by_fold(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, group in predictions.groupby([*GROUP_COLUMNS, "fold"], sort=False):
        group_dict = dict(zip([*GROUP_COLUMNS, "fold"], group_values))
        y_true = group["label_binary"].to_numpy(dtype=int)
        y_proba = group["y_proba"].to_numpy(dtype=float)
        rows.append(
            {
                **group_dict,
                "n_valid": int(len(group)),
                "n_valid_records": int(group["record_id"].nunique()),
                **calculate_metrics(y_true, y_proba, THRESHOLD),
            }
        )

    return pd.DataFrame(rows)


def summarize_results(results_by_fold: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, group in results_by_fold.groupby(GROUP_COLUMNS, sort=False):
        group_dict = dict(zip(GROUP_COLUMNS, group_values))
        row: dict[str, object] = {
            **group_dict,
            "n_folds": int(group["fold"].nunique()),
            "n_epochs": int(group["n_valid"].sum()),
            "n_records": int(group["n_valid_records"].sum()),
        }
        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=0))
        for column in CONFUSION_COLUMNS:
            row[f"{column}_sum"] = int(group[column].sum())
        rows.append(row)

    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def sweep_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_values, group in predictions.groupby(GROUP_COLUMNS, sort=False):
        group_dict = dict(zip(GROUP_COLUMNS, group_values))
        y_true = group["label_binary"].to_numpy(dtype=int)
        y_proba = group["y_proba"].to_numpy(dtype=float)
        sweep_rows = []
        for threshold in THRESHOLDS:
            metrics = calculate_metrics(y_true, y_proba, float(threshold))
            sweep_rows.append(
                {
                    "threshold": float(threshold),
                    "youden_index": (
                        metrics["sensitivity"] + metrics["specificity"] - 1
                    ),
                    **metrics,
                }
            )

        sweep = pd.DataFrame(sweep_rows)
        selections = [
            ("max_f1", sweep.sort_values(["f1", "roc_auc"], ascending=False).iloc[0]),
            (
                "max_youden",
                sweep.sort_values(["youden_index", "f1"], ascending=False).iloc[0],
            ),
        ]
        sensitivity_candidates = sweep[sweep["sensitivity"] >= 0.70]
        if sensitivity_candidates.empty:
            row = sweep.sort_values(
                ["sensitivity", "specificity"],
                ascending=False,
            ).iloc[0]
            rule = "sensitivity_ge_0_70_max_specificity_unmet"
        else:
            row = sensitivity_candidates.sort_values(
                ["specificity", "f1"],
                ascending=False,
            ).iloc[0]
            rule = "sensitivity_ge_0_70_max_specificity"
        selections.append((rule, row))

        for selection_rule, row in selections:
            rows.append(
                {
                    **group_dict,
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
                }
            )

    return pd.DataFrame(rows)


def dataframe_to_markdown(table: pd.DataFrame, decimals: int = 4) -> str:
    if table.empty:
        return "No rows available."

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

    return "\n".join(lines)


def write_report(summary: pd.DataFrame, thresholds: pd.DataFrame) -> None:
    best_auc = summary.iloc[0]
    best_f1 = summary.sort_values("f1_mean", ascending=False).iloc[0]
    best_tuned = thresholds[thresholds["selection_rule"] == "max_f1"].sort_values(
        "f1",
        ascending=False,
    ).iloc[0]

    result_table = dataframe_to_markdown(
        summary[
            [
                "experiment",
                "components",
                "postprocessing",
                "smoothing_window_epochs",
                "roc_auc_mean",
                "f1_mean",
                "sensitivity_mean",
                "specificity_mean",
                "average_precision_mean",
            ]
        ],
        decimals=4,
    )
    threshold_table = dataframe_to_markdown(
        thresholds[
            [
                "experiment",
                "selection_rule",
                "threshold",
                "f1",
                "sensitivity",
                "specificity",
                "precision",
            ]
        ],
        decimals=4,
    )

    report = f"""# 1D-CNN DL Improvement Analysis

## Goal

Improve the DL part of the project without retraining or leakage by using saved subject-level out-of-fold predictions from:

- `reports/tables/cnn_cv_predictions.csv`
- `reports/tables/cnn_improved_cv_predictions.csv`

The analysis tests fixed, label-free probability post-processing:

- causal rolling mean inside each `record_id`;
- equal-weight probability averaging between simple CNN and ResNet1D components.

No labels are used for smoothing or probability averaging. Labels are used only for reporting metrics and threshold selection.

## Leakage Controls

- Predictions are out-of-fold under subject-level validation.
- Smoothing is applied only within the same `record_id`.
- The improvement variants here use causal smoothing only.
- The underlying CNN context models are still offline retrospective where their input uses future neighboring epochs.
- The longest causal smoothing window is 121 epochs, about 60.5 minutes. It should be interpreted as probability stabilization over a long PSG interval, not as a local event detector.
- This is an exploratory DL post-processing comparison on the same out-of-fold predictions, so the selected best variant can be mildly optimistic and should be reported as an internal validation result.

## Results At Threshold 0.5

{result_table}

## Threshold Sweep

{threshold_table}

## Comparison

Previous simple CNN: ROC-AUC={PREVIOUS_CNN_AUC:.4f}, F1={PREVIOUS_CNN_F1:.4f}.

Previous ResNet1D: ROC-AUC={PREVIOUS_RESNET_AUC:.4f}, F1={PREVIOUS_RESNET_F1:.4f}.

Temporal ensemble: ROC-AUC={TEMPORAL_ENSEMBLE_AUC:.4f}, F1={TEMPORAL_ENSEMBLE_F1:.4f}.

Best DL-improvement ROC-AUC: {float(best_auc["roc_auc_mean"]):.4f} (`{best_auc["experiment"]}`).

Best DL-improvement F1 at threshold 0.5: {float(best_f1["f1_mean"]):.4f} (`{best_f1["experiment"]}`).

Best tuned DL-improvement F1: {float(best_tuned["f1"]):.4f} (`{best_tuned["experiment"]}`, threshold={float(best_tuned["threshold"]):.2f}).

## Interpretation

The DL improvement layer improves the DL-only metrics compared with the simple CNN and the first ResNet1D run, mainly by stabilizing noisy epoch-level probabilities. It still does not beat the classical temporal ensemble. The long smoothing window means the best DL AUC should be described as a retrospective/post-processing result, not as real-time event detection. This supports the thesis conclusion that, on small UCDDB subject-level validation, simple deep models on raw signals need either stronger architectures, more data, or external pretraining to outperform carefully engineered temporal ML features.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")


def main() -> None:
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    cnn, resnet = load_predictions()
    predictions = build_experiments(cnn, resnet)
    results_by_fold = calculate_results_by_fold(predictions)
    summary = summarize_results(results_by_fold)
    thresholds = sweep_thresholds(predictions)

    predictions.to_csv(CV_PREDICTIONS_PATH, index=False)
    results_by_fold.to_csv(RESULTS_BY_FOLD_PATH, index=False)
    summary.to_csv(RESULTS_SUMMARY_PATH, index=False)
    thresholds.to_csv(BEST_THRESHOLDS_PATH, index=False)
    write_report(summary, thresholds)

    best_auc = summary.iloc[0]
    best_f1 = summary.sort_values("f1_mean", ascending=False).iloc[0]
    best_tuned = thresholds[thresholds["selection_rule"] == "max_f1"].sort_values(
        "f1",
        ascending=False,
    ).iloc[0]

    print("DL improvement analysis complete.")
    print(
        "Best ROC-AUC: "
        f"{best_auc['roc_auc_mean']:.4f} ({best_auc['experiment']})"
    )
    print(
        "Best F1 @ 0.5: "
        f"{best_f1['f1_mean']:.4f} ({best_f1['experiment']})"
    )
    print(
        "Best tuned F1: "
        f"{best_tuned['f1']:.4f} ({best_tuned['experiment']}, "
        f"threshold={best_tuned['threshold']:.2f})"
    )
    print(f"Saved: {RESULTS_SUMMARY_PATH}")
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
