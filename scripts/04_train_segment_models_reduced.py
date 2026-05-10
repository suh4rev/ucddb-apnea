from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, REPORTS_TABLES_DIR  # noqa: E402


RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5

PREVIOUS_BEST_AUC = 0.6725
PREVIOUS_BEST_F1 = 0.4152
PREVIOUS_TUNED_F1 = 0.4311

FEATURES_PATH = DATA_PROCESSED_DIR / "segment_features.csv"
RESULTS_BY_FOLD_PATH = REPORTS_TABLES_DIR / "segment_reduced_results_by_fold.csv"
RESULTS_SUMMARY_PATH = REPORTS_TABLES_DIR / "segment_reduced_results_summary.csv"
BEST_RESULTS_PATH = REPORTS_TABLES_DIR / "segment_reduced_best_results.csv"
BEST_THRESHOLDS_PATH = REPORTS_TABLES_DIR / "segment_reduced_best_thresholds.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "segment_reduced_analysis_report.md"

EXCLUDED_COLUMNS = {
    "record_id",
    "segment_id",
    "window_sec",
    "stride_sec",
    "start_sec",
    "end_sec",
    "sleep_stage_majority",
    "is_sleep_segment",
    "label",
    "label_binary",
    "overlap_seconds",
    "max_event_overlap_seconds",
}

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

METADATA_COLUMNS = [
    "record_id",
    "window_sec",
    "is_sleep_segment",
    "label_binary",
]

ALL_FEATURE_COLUMNS = [
    *FLOW_FEATURE_COLUMNS,
    *SPO2_FEATURE_COLUMNS,
    *EFFORT_FEATURE_COLUMNS,
]

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
    "regime",
    "window_set",
    "feature_set",
    "model",
    "config_name",
]

RESULT_BY_FOLD_COLUMNS = [
    *GROUP_COLUMNS,
    "fold",
    "n_features",
    "n_train",
    "n_valid",
    "n_train_records",
    "n_valid_records",
    *METRIC_COLUMNS,
    *CONFUSION_COLUMNS,
]

SUMMARY_COLUMNS = [
    *GROUP_COLUMNS,
    "n_features",
    *[f"{metric}_mean" for metric in METRIC_COLUMNS],
    *[f"{metric}_std" for metric in METRIC_COLUMNS],
    *[f"{column}_sum" for column in CONFUSION_COLUMNS],
]

THRESHOLD_COLUMNS = [
    *GROUP_COLUMNS,
    "selection_rule",
    "threshold",
    "accuracy",
    "precision",
    "sensitivity",
    "specificity",
    "f1",
    "tn",
    "fp",
    "fn",
    "tp",
    "youden_index",
    "details",
]

THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)

REGIMES = {
    "all_records_sleep_only": {
        "exclude_ucddb005": False,
    },
    "exclude_ucddb005_sleep_only": {
        "exclude_ucddb005": True,
    },
}

WINDOW_SETS = {
    "window_60s": 60,
    "window_10s": 10,
}

XGBOOST_CONFIGS = {
    "xgb_shallow": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 1,
        "reg_lambda": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "xgb_regularized": {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 5,
        "reg_lambda": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
}


def import_xgboost():
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise SystemExit(
            "xgboost is not installed. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    return XGBClassifier


def unique_preserve_order(columns: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for column in columns:
        if column not in seen:
            result.append(column)
            seen.add(column)

    return result


def valid_numeric_columns(features: pd.DataFrame, columns: list[str]) -> list[str]:
    result = []

    for column in unique_preserve_order(columns):
        if column not in features.columns:
            continue
        if column in EXCLUDED_COLUMNS or column.startswith("event_"):
            continue
        if not pd.api.types.is_numeric_dtype(features[column]):
            continue
        result.append(column)

    return result


def build_feature_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    flow = valid_numeric_columns(features, FLOW_FEATURE_COLUMNS)
    spo2 = valid_numeric_columns(features, SPO2_FEATURE_COLUMNS)
    effort = valid_numeric_columns(features, EFFORT_FEATURE_COLUMNS)

    return {
        "flow_spo2": unique_preserve_order([*flow, *spo2]),
        "respiratory_spo2": unique_preserve_order([*flow, *spo2, *effort]),
    }


def model_configs() -> list[tuple[str, str]]:
    return [
        ("XGBoost", "xgb_shallow"),
        ("XGBoost", "xgb_regularized"),
        ("ExtraTrees", "extra_trees"),
    ]


def read_segment_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise SystemExit(
            f"Segment feature table not found: {FEATURES_PATH}. "
            "Run: python scripts/03_build_segment_features.py"
        )

    usecols = unique_preserve_order([*METADATA_COLUMNS, *ALL_FEATURE_COLUMNS])
    features = pd.read_csv(FEATURES_PATH, usecols=lambda column: column in usecols)
    missing_columns = set(METADATA_COLUMNS) - set(features.columns)
    if missing_columns:
        raise SystemExit(f"Missing required columns: {sorted(missing_columns)}")

    return features


def filter_dataset(features: pd.DataFrame, regime: str, window_set: str) -> pd.DataFrame:
    result = features[
        (features["is_sleep_segment"] == 1)
        & (features["window_sec"] == WINDOW_SETS[window_set])
    ].copy()

    if REGIMES[regime]["exclude_ucddb005"]:
        result = result[result["record_id"] != "ucddb005"].copy()

    return result.reset_index(drop=True)


def make_cv_splits(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray]]:
    n_groups = int(groups.nunique())
    if n_groups < N_SPLITS:
        raise ValueError(f"Need at least {N_SPLITS} records for CV, got {n_groups}.")

    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        return list(splitter.split(X, y, groups))
    except ImportError:
        splitter = GroupKFold(n_splits=N_SPLITS)
        return list(splitter.split(X, y, groups))


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())

    if n_positive == 0:
        return 1.0

    return n_negative / n_positive


def make_xgboost_pipeline(config_name: str, y_train: pd.Series) -> Pipeline:
    XGBClassifier = import_xgboost()
    config = XGBOOST_CONFIGS[config_name]

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    **config,
                    eval_metric="logloss",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    tree_method="hist",
                    scale_pos_weight=calculate_scale_pos_weight(y_train),
                ),
            ),
        ]
    )


def make_extra_trees_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                ExtraTreesClassifier(
                    n_estimators=500,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_model_pipeline(
    model_name: str,
    config_name: str,
    y_train: pd.Series,
) -> Pipeline:
    if model_name == "XGBoost":
        return make_xgboost_pipeline(config_name, y_train)

    if model_name == "ExtraTrees":
        return make_extra_trees_pipeline()

    raise ValueError(f"Unsupported model: {model_name}")


def safe_roc_auc(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y_true, y_proba))
    except ValueError:
        return np.nan


def safe_average_precision(y_true: pd.Series | np.ndarray, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return float(average_precision_score(y_true, y_proba))
    except ValueError:
        return np.nan


def specificity_from_confusion(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = specificity_from_confusion(int(tn), int(fp))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": safe_roc_auc(y_true, y_proba),
        "average_precision": safe_average_precision(y_true, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def predict_positive_probability(model: Pipeline, X_valid: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X_valid)[:, 1]


def make_fold_result_row(
    regime: str,
    window_set: str,
    feature_set: str,
    model_name: str,
    config_name: str,
    fold: int,
    n_features: int,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    groups: pd.Series,
    metrics: dict[str, float | int],
) -> dict[str, object]:
    return {
        "regime": regime,
        "window_set": window_set,
        "feature_set": feature_set,
        "model": model_name,
        "config_name": config_name,
        "fold": fold,
        "n_features": n_features,
        "n_train": int(len(train_indices)),
        "n_valid": int(len(valid_indices)),
        "n_train_records": int(groups.iloc[train_indices].nunique()),
        "n_valid_records": int(groups.iloc[valid_indices].nunique()),
        **metrics,
    }


def calculate_threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
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


def build_threshold_row(
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


def build_best_thresholds_for_experiment(
    key_values: dict[str, object],
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> list[dict[str, object]]:
    threshold_rows = []
    for threshold in THRESHOLDS:
        threshold_rows.append(
            {
                "threshold": float(threshold),
                **calculate_threshold_metrics(y_true, y_proba, float(threshold)),
            }
        )

    threshold_table = pd.DataFrame(threshold_rows)
    by_f1 = threshold_table.sort_values(
        ["f1", "specificity", "sensitivity"],
        ascending=[False, False, False],
    ).iloc[0]
    by_youden = threshold_table.sort_values(
        ["youden_index", "f1"],
        ascending=[False, False],
    ).iloc[0]
    sensitivity_candidates = threshold_table[threshold_table["sensitivity"] >= 0.70]

    rows = [
        build_threshold_row(key_values, "max_f1", by_f1),
        build_threshold_row(key_values, "max_youden", by_youden),
    ]

    if sensitivity_candidates.empty:
        rows.append(
            build_threshold_row(
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
            build_threshold_row(
                key_values,
                "sensitivity_ge_0.70_max_specificity",
                sensitivity_row,
            )
        )

    return rows


def run_experiment(
    data: pd.DataFrame,
    regime: str,
    window_set: str,
    feature_set: str,
    feature_columns: list[str],
    model_name: str,
    config_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if not feature_columns:
        print(f"Skipping {regime}/{window_set}/{feature_set}/{model_name}/{config_name}: no features.")
        return [], []

    y = data["label_binary"].astype(int).reset_index(drop=True)
    groups = data["record_id"].astype(str).reset_index(drop=True)
    X = data[feature_columns].reset_index(drop=True)

    if y.nunique() < 2:
        print(f"Skipping {regime}/{window_set}/{feature_set}/{model_name}/{config_name}: one class.")
        return [], []

    splits = make_cv_splits(X, y, groups)
    fold_rows: list[dict[str, object]] = []
    oof_true: list[np.ndarray] = []
    oof_proba: list[np.ndarray] = []

    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        X_train = X.iloc[train_indices]
        X_valid = X.iloc[valid_indices]
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]

        if y_train.nunique() < 2:
            print(f"Skipping fold {fold}: train split has one class.")
            continue

        model = make_model_pipeline(model_name, config_name, y_train)
        model.fit(X_train, y_train)

        y_proba = predict_positive_probability(model, X_valid)
        y_pred = (y_proba >= THRESHOLD).astype(int)
        metrics = calculate_metrics(y_valid, y_proba, y_pred)
        fold_rows.append(
            make_fold_result_row(
                regime=regime,
                window_set=window_set,
                feature_set=feature_set,
                model_name=model_name,
                config_name=config_name,
                fold=fold,
                n_features=len(feature_columns),
                train_indices=train_indices,
                valid_indices=valid_indices,
                groups=groups,
                metrics=metrics,
            )
        )
        oof_true.append(y_valid.to_numpy(dtype=int))
        oof_proba.append(y_proba.astype(float))

    if not oof_true:
        return fold_rows, []

    key_values = {
        "regime": regime,
        "window_set": window_set,
        "feature_set": feature_set,
        "model": model_name,
        "config_name": config_name,
    }
    threshold_rows = build_best_thresholds_for_experiment(
        key_values,
        np.concatenate(oof_true),
        np.concatenate(oof_proba),
    )
    return fold_rows, threshold_rows


def summarize_results(results_by_fold: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for keys, group in results_by_fold.groupby(GROUP_COLUMNS, sort=True):
        row = dict(zip(GROUP_COLUMNS, keys))
        row["n_features"] = int(group["n_features"].iloc[0])

        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))

        for column in CONFUSION_COLUMNS:
            row[f"{column}_sum"] = int(group[column].sum())

        summary_rows.append(row)

    return pd.DataFrame(summary_rows, columns=SUMMARY_COLUMNS)


def format_metric(value: object) -> str:
    if pd.isna(value):
        return "NA"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def describe_result(row: pd.Series | None) -> str:
    if row is None:
        return "not available"

    return (
        f"{row['regime']} / {row['window_set']} / {row['feature_set']} / "
        f"{row['model']} / {row['config_name']} "
        f"(ROC-AUC={format_metric(row['roc_auc_mean'])}, "
        f"F1={format_metric(row['f1_mean'])}, "
        f"sensitivity={format_metric(row['sensitivity_mean'])}, "
        f"specificity={format_metric(row['specificity_mean'])})"
    )


def best_summary_row(summary: pd.DataFrame, metric: str) -> pd.Series | None:
    if summary.empty or metric not in summary.columns:
        return None

    ranked = summary.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if ranked.empty:
        return None

    return ranked.iloc[0]


def best_threshold_row(
    thresholds: pd.DataFrame,
    selection_rule: str,
    metric: str,
) -> pd.Series | None:
    subset = thresholds[thresholds["selection_rule"] == selection_rule].copy()
    if subset.empty or metric not in subset.columns:
        return None

    subset = subset.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if subset.empty:
        return None

    return subset.iloc[0]


def describe_threshold(row: pd.Series | None) -> str:
    if row is None:
        return "not available"

    return (
        f"{row['regime']} / {row['window_set']} / {row['feature_set']} / "
        f"{row['model']} / {row['config_name']} @ threshold={format_metric(row['threshold'])} "
        f"(F1={format_metric(row['f1'])}, sensitivity={format_metric(row['sensitivity'])}, "
        f"specificity={format_metric(row['specificity'])})"
    )


def best_group_rows(summary: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []

    for _, group in summary.groupby(group_column, sort=True):
        best = best_summary_row(group, "roc_auc_mean")
        if best is not None:
            rows.append(best)

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False)


def markdown_table(table: pd.DataFrame, columns: list[str]) -> str:
    if table.empty:
        return "_No rows available._"

    display = table[columns].copy()
    for column in display.columns:
        if pd.api.types.is_float_dtype(display[column]):
            display[column] = display[column].map(format_metric)

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = [
        "| " + " | ".join(str(row[column]) for column in columns) + " |"
        for _, row in display.iterrows()
    ]
    return "\n".join([header, separator, *body])


def build_report(
    results_summary: pd.DataFrame,
    best_results: pd.DataFrame,
    best_thresholds: pd.DataFrame,
) -> str:
    best_auc = best_summary_row(results_summary, "roc_auc_mean")
    best_f1 = best_summary_row(results_summary, "f1_mean")
    best_tuned_f1 = best_threshold_row(best_thresholds, "max_f1", "f1")

    best_auc_value = float(best_auc["roc_auc_mean"]) if best_auc is not None else np.nan
    best_f1_value = float(best_f1["f1_mean"]) if best_f1 is not None else np.nan
    best_tuned_value = (
        float(best_tuned_f1["f1"])
        if best_tuned_f1 is not None and pd.notna(best_tuned_f1["f1"])
        else np.nan
    )

    auc_delta = best_auc_value - PREVIOUS_BEST_AUC if np.isfinite(best_auc_value) else np.nan
    f1_delta = best_f1_value - PREVIOUS_BEST_F1 if np.isfinite(best_f1_value) else np.nan
    tuned_delta = (
        best_tuned_value - PREVIOUS_TUNED_F1
        if np.isfinite(best_tuned_value)
        else np.nan
    )

    segment_helped = bool(np.isfinite(auc_delta) and auc_delta > 0)
    auc_08_reached = bool(np.isfinite(best_auc_value) and best_auc_value >= 0.80)

    window_rows = best_group_rows(results_summary, "window_set")
    feature_rows = best_group_rows(results_summary, "feature_set")
    regime_rows = best_group_rows(results_summary, "regime")

    helped_text = (
        "Segment-level representation improved the best ROC-AUC over the previous reference."
        if segment_helped
        else "Segment-level representation did not improve the best ROC-AUC over the previous reference in this reduced honest run."
    )
    auc_08_text = (
        "AUC 0.8 was reached."
        if auc_08_reached
        else (
            "AUC 0.8 was not reached. This is plausible under subject-level CV on UCDDB: "
            "the number of subjects is small, inter-subject physiology varies, tabular "
            "segment features still discard raw waveform morphology, and apnea/hypopnea "
            "event labels are noisy around onset and recovery. A sequence model or raw-signal "
            "deep learning approach is a more realistic route to a large jump."
        )
    )
    tuned_caveat = (
        "The best tuned F1 is only marginally above the previous tuned F1, and it is "
        "achieved at very low specificity. This should not be interpreted as a clinically "
        "useful improvement."
        if np.isfinite(tuned_delta) and tuned_delta > 0
        else "Threshold tuning did not produce a meaningful F1 improvement over the previous reference."
    )

    return f"""# Reduced Segment-Level Experiment

## Summary

Best ROC-AUC: {describe_result(best_auc)}.

Best F1 at threshold 0.5: {describe_result(best_f1)}.

Best tuned F1: {describe_threshold(best_tuned_f1)}.

Comparison with previous references:

- Previous best AUC: {PREVIOUS_BEST_AUC:.4f}; reduced best delta: {format_metric(auc_delta)}
- Previous best F1: {PREVIOUS_BEST_F1:.4f}; reduced best delta: {format_metric(f1_delta)}
- Previous tuned F1: {PREVIOUS_TUNED_F1:.4f}; reduced tuned delta: {format_metric(tuned_delta)}

{helped_text}

{tuned_caveat}

{auc_08_text}

## Top Results

{markdown_table(best_results.head(10), ["regime", "window_set", "feature_set", "model", "config_name", "roc_auc_mean", "f1_mean", "sensitivity_mean", "specificity_mean"])}

## Window Comparison

{markdown_table(window_rows, ["window_set", "regime", "feature_set", "model", "config_name", "roc_auc_mean", "f1_mean"])}

## Feature Set Comparison

{markdown_table(feature_rows, ["feature_set", "regime", "window_set", "model", "config_name", "roc_auc_mean", "f1_mean"])}

## ucddb005 Exclusion

{markdown_table(regime_rows, ["regime", "window_set", "feature_set", "model", "config_name", "roc_auc_mean", "f1_mean"])}

## Leakage Controls

The reduced experiment uses only subject-level CV grouped by `record_id`. The feature matrix excludes `record_id`, segment IDs, time columns, sleep stage, labels, overlap diagnostics, and any `event_*` columns. Respiratory event annotations are used only through the already-built label in `segment_features.csv`.

SpO2 future-context features (`spo2_next_30s_*`, `spo2_next_60s_*`) are retained from the segment feature table. They do not use labels or event metadata, but they are valid only for offline retrospective PSG analysis, not real-time detection.
"""


def print_data_overview(features: pd.DataFrame) -> None:
    print("Reduced segment dataset")
    print(f"  rows loaded: {len(features)}")
    print(f"  records: {features['record_id'].nunique()}")
    print(f"  sleep-only rows: {int((features['is_sleep_segment'] == 1).sum())}")
    print("  No full cv_predictions table will be written.")


def main() -> None:
    import_xgboost()
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    features = read_segment_features()
    if features.empty:
        raise SystemExit("Segment feature table is empty.")

    print_data_overview(features)
    feature_sets = build_feature_sets(features)

    fold_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []

    for regime in REGIMES:
        for window_set in WINDOW_SETS:
            data = filter_dataset(features, regime, window_set)
            y = data["label_binary"].astype(int) if not data.empty else pd.Series(dtype=int)
            positive_rate_text = f"{float(y.mean()):.4f}" if len(y) else "NA"
            print(
                f"\nDataset {regime} / {window_set}: "
                f"rows={len(data)}, records={data['record_id'].nunique() if not data.empty else 0}, "
                f"positive_rate={positive_rate_text}"
            )

            if data.empty:
                continue

            for feature_set, feature_columns in feature_sets.items():
                for model_name, config_name in model_configs():
                    print(
                        f"Training {regime} / {window_set} / {feature_set} / "
                        f"{model_name} / {config_name}"
                    )
                    experiment_fold_rows, experiment_threshold_rows = run_experiment(
                        data=data,
                        regime=regime,
                        window_set=window_set,
                        feature_set=feature_set,
                        feature_columns=feature_columns,
                        model_name=model_name,
                        config_name=config_name,
                    )
                    fold_rows.extend(experiment_fold_rows)
                    threshold_rows.extend(experiment_threshold_rows)

    results_by_fold = pd.DataFrame(fold_rows, columns=RESULT_BY_FOLD_COLUMNS)
    if results_by_fold.empty:
        raise SystemExit("No reduced model results were produced.")

    results_summary = summarize_results(results_by_fold)
    best_results = (
        results_summary.sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )
    best_thresholds = pd.DataFrame(threshold_rows, columns=THRESHOLD_COLUMNS)

    results_by_fold.to_csv(RESULTS_BY_FOLD_PATH, index=False)
    results_summary.to_csv(RESULTS_SUMMARY_PATH, index=False)
    best_results.to_csv(BEST_RESULTS_PATH, index=False)
    best_thresholds.to_csv(BEST_THRESHOLDS_PATH, index=False)

    report = build_report(results_summary, best_results, best_thresholds)
    REPORT_PATH.write_text(report, encoding="utf-8")

    best_auc = best_summary_row(results_summary, "roc_auc_mean")
    best_f1 = best_summary_row(results_summary, "f1_mean")
    best_tuned_f1 = best_threshold_row(best_thresholds, "max_f1", "f1")

    print("\nReduced segment experiment summary")
    print(f"  Best ROC-AUC: {describe_result(best_auc)}")
    print(f"  Best F1 @0.5: {describe_result(best_f1)}")
    print(f"  Best tuned F1: {describe_threshold(best_tuned_f1)}")
    print(f"  Results summary: {RESULTS_SUMMARY_PATH}")
    print(f"  Best results: {BEST_RESULTS_PATH}")
    print(f"  Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
