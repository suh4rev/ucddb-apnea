from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, REPORTS_TABLES_DIR  # noqa: E402


RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5

PREVIOUS_BEST_AUC = 0.6725
PREVIOUS_BEST_F1 = 0.4152
PREVIOUS_THRESHOLD_TUNED_F1 = 0.4311

FEATURES_PATH = DATA_PROCESSED_DIR / "segment_features.csv"
RESULTS_BY_FOLD_PATH = REPORTS_TABLES_DIR / "segment_model_results_by_fold.csv"
RESULTS_SUMMARY_PATH = REPORTS_TABLES_DIR / "segment_model_results_summary.csv"
CONFUSION_MATRICES_PATH = REPORTS_TABLES_DIR / "segment_model_confusion_matrices.csv"
CV_PREDICTIONS_PATH = REPORTS_TABLES_DIR / "segment_cv_predictions.csv"
BEST_RESULTS_PATH = REPORTS_TABLES_DIR / "segment_best_results.csv"
BEST_THRESHOLDS_PATH = REPORTS_TABLES_DIR / "segment_best_thresholds.csv"

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
    "experiment",
    "model",
    "config_name",
    "postprocessing",
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

PREDICTION_COLUMNS = [
    "record_id",
    "segment_id",
    "window_sec",
    "start_sec",
    "end_sec",
    "label_binary",
    *GROUP_COLUMNS,
    "fold",
    "y_proba",
    "y_pred",
]

THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)

XGBOOST_CONFIGS = {
    "config_xgb_1": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 1,
        "reg_lambda": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "config_xgb_2": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 5,
        "reg_lambda": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "config_xgb_3": {
        "n_estimators": 600,
        "learning_rate": 0.02,
        "max_depth": 3,
        "min_child_weight": 3,
        "reg_lambda": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
}

REGIMES = {
    "all_records_all_segments": {
        "exclude_ucddb005": False,
        "sleep_only": False,
    },
    "all_records_sleep_only": {
        "exclude_ucddb005": False,
        "sleep_only": True,
    },
    "exclude_ucddb005_all_segments": {
        "exclude_ucddb005": True,
        "sleep_only": False,
    },
    "exclude_ucddb005_sleep_only": {
        "exclude_ucddb005": True,
        "sleep_only": True,
    },
}

WINDOW_SETS = {
    "window_60s": 60,
    "window_10s": 10,
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
        "flow_only": flow,
        "spo2_only": spo2,
        "flow_spo2": unique_preserve_order([*flow, *spo2]),
        "respiratory_spo2": unique_preserve_order([*flow, *spo2, *effort]),
    }


def filter_regime(features: pd.DataFrame, regime_name: str) -> pd.DataFrame:
    config = REGIMES[regime_name]
    result = features.copy()

    if config["exclude_ucddb005"]:
        result = result[result["record_id"] != "ucddb005"].copy()

    if config["sleep_only"]:
        result = result[result["is_sleep_segment"] == 1].copy()

    return result


def filter_window(features: pd.DataFrame, window_set: str) -> pd.DataFrame:
    window_sec = WINDOW_SETS[window_set]
    return features[features["window_sec"] == window_sec].copy()


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())

    if n_positive == 0:
        return 1.0

    return n_negative / n_positive


def make_xgboost_pipeline(config: dict[str, int | float], y_train: pd.Series) -> Pipeline:
    XGBClassifier = import_xgboost()

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
                    scale_pos_weight=calculate_scale_pos_weight(y_train),
                    tree_method="hist",
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
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def make_random_forest_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestClassifier(
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
        return make_xgboost_pipeline(XGBOOST_CONFIGS[config_name], y_train)

    if model_name == "ExtraTreesClassifier":
        return make_extra_trees_pipeline()

    if model_name == "RandomForestClassifier":
        return make_random_forest_pipeline()

    raise ValueError(f"Unsupported model: {model_name}")


def model_configs() -> list[tuple[str, str]]:
    configs = [("XGBoost", config_name) for config_name in XGBOOST_CONFIGS]
    configs.extend(
        [
            ("ExtraTreesClassifier", "default"),
            ("RandomForestClassifier", "default"),
        ]
    )
    return configs


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


def safe_roc_auc(y_true: pd.Series, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return float(roc_auc_score(y_true, y_proba))
    except ValueError:
        return np.nan


def safe_average_precision(y_true: pd.Series, y_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        return float(average_precision_score(y_true, y_proba))
    except ValueError:
        return np.nan


def specificity_from_confusion(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


def calculate_metrics(
    y_true: pd.Series,
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


def make_experiment_name(
    regime: str,
    window_set: str,
    feature_set: str,
    model_name: str,
    config_name: str,
    postprocessing: str,
) -> str:
    base = f"{regime}__{window_set}__{feature_set}__{model_name}__{config_name}"
    if postprocessing == "smoothed":
        return f"{base}_smoothed"

    return base


def make_result_row(
    regime: str,
    window_set: str,
    feature_set: str,
    model_name: str,
    config_name: str,
    postprocessing: str,
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
        "experiment": make_experiment_name(
            regime,
            window_set,
            feature_set,
            model_name,
            config_name,
            postprocessing,
        ),
        "model": model_name,
        "config_name": config_name,
        "postprocessing": postprocessing,
        "fold": fold,
        "n_features": n_features,
        "n_train": int(len(train_indices)),
        "n_valid": int(len(valid_indices)),
        "n_train_records": int(groups.iloc[train_indices].nunique()),
        "n_valid_records": int(groups.iloc[valid_indices].nunique()),
        **metrics,
    }


def make_result_row_from_predictions(
    group_values: dict[str, object],
    fold: int,
    n_features: int,
    fold_metadata: dict[int, dict[str, object]],
    metrics: dict[str, float | int],
) -> dict[str, object]:
    metadata = fold_metadata[fold]
    return {
        **group_values,
        "fold": fold,
        "n_features": n_features,
        "n_train": int(metadata["n_train"]),
        "n_valid": int(metadata["n_valid"]),
        "n_train_records": int(metadata["n_train_records"]),
        "n_valid_records": int(metadata["n_valid_records"]),
        **metrics,
    }


def make_prediction_rows(
    metadata: pd.DataFrame,
    regime: str,
    window_set: str,
    feature_set: str,
    model_name: str,
    config_name: str,
    postprocessing: str,
    fold: int,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    experiment = make_experiment_name(
        regime,
        window_set,
        feature_set,
        model_name,
        config_name,
        postprocessing,
    )
    rows = []

    for row_index, (_, row) in enumerate(metadata.iterrows()):
        rows.append(
            {
                "record_id": row["record_id"],
                "segment_id": row["segment_id"],
                "window_sec": int(row["window_sec"]),
                "start_sec": float(row["start_sec"]),
                "end_sec": float(row["end_sec"]),
                "label_binary": int(row["label_binary"]),
                "regime": regime,
                "window_set": window_set,
                "feature_set": feature_set,
                "experiment": experiment,
                "model": model_name,
                "config_name": config_name,
                "postprocessing": postprocessing,
                "fold": fold,
                "y_proba": float(y_proba[row_index]),
                "y_pred": int(y_pred[row_index]),
            }
        )

    return rows


def build_smoothed_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    smoothed = predictions.copy()
    smoothed = smoothed.sort_values(["record_id", "start_sec", "segment_id"]).copy()
    smoothed["y_proba"] = smoothed.groupby("record_id", sort=False)["y_proba"].transform(
        lambda values: values.rolling(window=3, center=True, min_periods=1).mean()
    )
    smoothed["y_pred"] = (smoothed["y_proba"] >= THRESHOLD).astype(int)
    smoothed["postprocessing"] = "smoothed"

    for columns, group in smoothed.groupby(
        ["regime", "window_set", "feature_set", "model", "config_name"],
        sort=False,
    ):
        regime, window_set, feature_set, model_name, config_name = columns
        experiment = make_experiment_name(
            str(regime),
            str(window_set),
            str(feature_set),
            str(model_name),
            str(config_name),
            "smoothed",
        )
        smoothed.loc[group.index, "experiment"] = experiment

    return smoothed.sort_index()


def run_single_experiment_cv(
    features: pd.DataFrame,
    regime: str,
    window_set: str,
    feature_set: str,
    feature_columns: list[str],
    model_name: str,
    config_name: str,
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    if not feature_columns:
        print(f"Skipping {regime}/{window_set}/{feature_set}/{model_name}/{config_name}: no features.")
        return result_rows, pd.DataFrame(columns=PREDICTION_COLUMNS)

    y = features["label_binary"].astype(int).reset_index(drop=True)
    groups = features["record_id"].astype(str).reset_index(drop=True)
    X = features[feature_columns].reset_index(drop=True)
    metadata_all = features[
        ["record_id", "segment_id", "window_sec", "start_sec", "end_sec", "label_binary"]
    ].reset_index(drop=True)

    if y.nunique() < 2:
        print(f"Skipping {regime}/{window_set}/{feature_set}/{model_name}/{config_name}: one class.")
        return result_rows, pd.DataFrame(columns=PREDICTION_COLUMNS)

    splits = make_cv_splits(X, y, groups)
    fold_metadata: dict[int, dict[str, object]] = {}

    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        X_train = X.iloc[train_indices]
        X_valid = X.iloc[valid_indices]
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]

        if y_train.nunique() < 2:
            print(f"Skipping fold {fold}: training split has one class.")
            continue

        model = make_model_pipeline(model_name, config_name, y_train)
        model.fit(X_train, y_train)

        y_proba = predict_positive_probability(model, X_valid)
        y_pred = (y_proba >= THRESHOLD).astype(int)
        metrics = calculate_metrics(y_valid, y_proba, y_pred)

        fold_metadata[fold] = {
            "n_train": int(len(train_indices)),
            "n_valid": int(len(valid_indices)),
            "n_train_records": int(groups.iloc[train_indices].nunique()),
            "n_valid_records": int(groups.iloc[valid_indices].nunique()),
        }

        result_rows.append(
            make_result_row(
                regime=regime,
                window_set=window_set,
                feature_set=feature_set,
                model_name=model_name,
                config_name=config_name,
                postprocessing="raw",
                fold=fold,
                n_features=len(feature_columns),
                train_indices=train_indices,
                valid_indices=valid_indices,
                groups=groups,
                metrics=metrics,
            )
        )

        metadata = metadata_all.iloc[valid_indices]
        prediction_rows.extend(
            make_prediction_rows(
                metadata=metadata,
                regime=regime,
                window_set=window_set,
                feature_set=feature_set,
                model_name=model_name,
                config_name=config_name,
                postprocessing="raw",
                fold=fold,
                y_proba=y_proba,
                y_pred=y_pred,
            )
        )

    if not prediction_rows:
        return result_rows, pd.DataFrame(columns=PREDICTION_COLUMNS)

    raw_predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    smoothed_predictions = build_smoothed_predictions(raw_predictions)
    group_values = {
        "regime": regime,
        "window_set": window_set,
        "feature_set": feature_set,
        "experiment": make_experiment_name(
            regime,
            window_set,
            feature_set,
            model_name,
            config_name,
            "smoothed",
        ),
        "model": model_name,
        "config_name": config_name,
        "postprocessing": "smoothed",
    }

    for fold, fold_group in smoothed_predictions.groupby("fold", sort=True):
        y_true = fold_group["label_binary"].astype(int)
        y_proba = fold_group["y_proba"].astype(float).to_numpy()
        y_pred = (y_proba >= THRESHOLD).astype(int)
        metrics = calculate_metrics(y_true, y_proba, y_pred)
        result_rows.append(
            make_result_row_from_predictions(
                group_values=group_values,
                fold=int(fold),
                n_features=len(feature_columns),
                fold_metadata=fold_metadata,
                metrics=metrics,
            )
        )

    predictions = pd.concat(
        [raw_predictions, smoothed_predictions],
        ignore_index=True,
        sort=False,
    )
    return result_rows, predictions


def summarize_results(results_by_fold: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for keys, group in results_by_fold.groupby(GROUP_COLUMNS, sort=True):
        row = dict(zip(GROUP_COLUMNS, keys))
        row["n_features"] = int(group["n_features"].iloc[0])

        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))

        for column in CONFUSION_COLUMNS:
            row[f"{column}_mean"] = float(group[column].mean())

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def summarize_confusion_matrices(results_by_fold: pd.DataFrame) -> pd.DataFrame:
    return (
        results_by_fold.groupby(GROUP_COLUMNS, as_index=False)[CONFUSION_COLUMNS]
        .sum()
        .sort_values(GROUP_COLUMNS)
    )


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


def build_best_thresholds(predictions: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if predictions.empty:
        return pd.DataFrame()

    for keys, group in predictions.groupby(GROUP_COLUMNS, sort=True):
        key_values = dict(zip(GROUP_COLUMNS, keys))
        y_true = group["label_binary"].astype(int)
        y_proba = group["y_proba"].astype(float)

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

        rows.append(best_threshold_row(key_values, "max_f1", by_f1))
        rows.append(best_threshold_row(key_values, "max_youden", by_youden))

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

    return pd.DataFrame(rows)


def print_dataset_overview(features: pd.DataFrame) -> None:
    print("Segment dataset")
    print(f"  rows: {len(features)}")
    print(f"  records: {features['record_id'].nunique()}")
    print(f"  positive_rate: {features['label_binary'].mean():.4f}")
    print(f"  60s rows: {int((features['window_sec'] == 60).sum())}")
    print(f"  10s rows: {int((features['window_sec'] == 10).sum())}")
    print("  Note: subject-level CV uses record_id as groups.")


def print_best_summary(
    results_summary: pd.DataFrame,
    best_thresholds: pd.DataFrame,
) -> None:
    ranked_auc = results_summary.dropna(subset=["roc_auc_mean"]).sort_values(
        "roc_auc_mean",
        ascending=False,
    )
    ranked_f1 = results_summary.dropna(subset=["f1_mean"]).sort_values(
        "f1_mean",
        ascending=False,
    )
    tuned = best_thresholds[best_thresholds["selection_rule"] == "max_f1"].copy()
    tuned = tuned.dropna(subset=["f1"]).sort_values("f1", ascending=False)
    sens = best_thresholds[
        best_thresholds["selection_rule"] == "sensitivity_ge_0.70_max_specificity"
    ].copy()
    sens = sens.dropna(subset=["specificity"]).sort_values(
        ["specificity", "f1"],
        ascending=[False, False],
    )

    print("\nSegment model summary")
    if not ranked_auc.empty:
        best_auc = ranked_auc.iloc[0]
        print(
            "  Best ROC-AUC: "
            f"{best_auc['experiment']} = {best_auc['roc_auc_mean']:.4f} "
            f"(delta vs 0.6725: {best_auc['roc_auc_mean'] - PREVIOUS_BEST_AUC:+.4f})"
        )
    if not ranked_f1.empty:
        best_f1 = ranked_f1.iloc[0]
        print(
            "  Best F1 @0.5: "
            f"{best_f1['experiment']} = {best_f1['f1_mean']:.4f} "
            f"(delta vs 0.4152: {best_f1['f1_mean'] - PREVIOUS_BEST_F1:+.4f})"
        )
    if not tuned.empty:
        best_tuned = tuned.iloc[0]
        print(
            "  Best tuned F1: "
            f"{best_tuned['experiment']} @ threshold={best_tuned['threshold']:.2f} "
            f"= {best_tuned['f1']:.4f} "
            f"(delta vs 0.4311: {best_tuned['f1'] - PREVIOUS_THRESHOLD_TUNED_F1:+.4f})"
        )
    if not sens.empty:
        best_sens = sens.iloc[0]
        print(
            "  Best sensitivity>=0.70 variant: "
            f"{best_sens['experiment']} @ threshold={best_sens['threshold']:.2f}, "
            f"sensitivity={best_sens['sensitivity']:.4f}, "
            f"specificity={best_sens['specificity']:.4f}"
        )


def main() -> None:
    if not FEATURES_PATH.exists():
        raise SystemExit(
            f"Segment feature table not found: {FEATURES_PATH}. "
            "Run: python scripts/experiments/03_build_segment_features.py"
        )

    import_xgboost()
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURES_PATH)
    if features.empty:
        raise SystemExit("Segment feature table is empty.")

    print_dataset_overview(features)

    result_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    predictions_header_written = False
    pd.DataFrame(columns=PREDICTION_COLUMNS).to_csv(CV_PREDICTIONS_PATH, index=False)
    predictions_header_written = True

    for regime in REGIMES:
        regime_features = filter_regime(features, regime)
        for window_set in WINDOW_SETS:
            data = filter_window(regime_features, window_set)
            if data.empty:
                print(f"Skipping {regime}/{window_set}: no rows.")
                continue

            feature_sets = build_feature_sets(data)
            for feature_set, feature_columns in feature_sets.items():
                for model_name, config_name in model_configs():
                    print(
                        f"\nTraining {regime} / {window_set} / {feature_set} / "
                        f"{model_name} / {config_name}"
                    )
                    rows, predictions = run_single_experiment_cv(
                        features=data,
                        regime=regime,
                        window_set=window_set,
                        feature_set=feature_set,
                        feature_columns=feature_columns,
                        model_name=model_name,
                        config_name=config_name,
                    )
                    result_rows.extend(rows)
                    if not predictions.empty:
                        predictions.to_csv(
                            CV_PREDICTIONS_PATH,
                            mode="a",
                            header=not predictions_header_written,
                            index=False,
                        )
                        predictions_header_written = True
                        threshold_table = build_best_thresholds(predictions)
                        threshold_rows.extend(threshold_table.to_dict(orient="records"))

    results_by_fold = pd.DataFrame(result_rows, columns=RESULT_BY_FOLD_COLUMNS)
    results_summary = summarize_results(results_by_fold)
    confusion_matrices = summarize_confusion_matrices(results_by_fold)
    best_thresholds = pd.DataFrame(threshold_rows)
    best_results = (
        results_summary.sort_values("roc_auc_mean", ascending=False)
        .head(30)
        .reset_index(drop=True)
    )

    results_by_fold.to_csv(RESULTS_BY_FOLD_PATH, index=False)
    results_summary.to_csv(RESULTS_SUMMARY_PATH, index=False)
    confusion_matrices.to_csv(CONFUSION_MATRICES_PATH, index=False)
    best_results.to_csv(BEST_RESULTS_PATH, index=False)
    best_thresholds.to_csv(BEST_THRESHOLDS_PATH, index=False)

    print_best_summary(results_summary, best_thresholds)
    print(f"  Results summary: {RESULTS_SUMMARY_PATH}")
    print(f"  Top-30 best results: {BEST_RESULTS_PATH}")


if __name__ == "__main__":
    main()
