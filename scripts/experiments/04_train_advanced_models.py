from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
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
MODEL_NAME = "XGBoost"

PREVIOUS_BEST_AUC = 0.6725
PREVIOUS_BEST_F1 = 0.4152
PREVIOUS_THRESHOLD_TUNED_F1 = 0.4311

FEATURES_PATH = DATA_PROCESSED_DIR / "features_advanced.csv"
RESULTS_BY_FOLD_PATH = REPORTS_TABLES_DIR / "advanced_model_results_by_fold.csv"
RESULTS_SUMMARY_PATH = REPORTS_TABLES_DIR / "advanced_model_results_summary.csv"
CONFUSION_MATRICES_PATH = REPORTS_TABLES_DIR / "advanced_model_confusion_matrices.csv"
CV_PREDICTIONS_PATH = REPORTS_TABLES_DIR / "advanced_cv_predictions.csv"
BEST_RESULTS_PATH = REPORTS_TABLES_DIR / "advanced_best_results.csv"
BEST_THRESHOLDS_PATH = REPORTS_TABLES_DIR / "advanced_best_thresholds.csv"

EXCLUDED_COLUMNS = {
    "record_id",
    "epoch_id",
    "start_sec",
    "end_sec",
    "sleep_stage",
    "label",
    "label_binary",
    "is_sleep_epoch",
    "n_events",
    "event_types",
    "event_start_sec",
    "event_duration_sec",
}

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

RESULT_BY_FOLD_COLUMNS = [
    "regime",
    "experiment",
    "model",
    "config_name",
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
    "epoch_id",
    "label_binary",
    "regime",
    "experiment",
    "model",
    "config_name",
    "fold",
    "y_proba",
    "y_pred",
]

BEST_RESULT_COLUMNS = [
    "regime",
    "experiment",
    "model",
    "config_name",
    "n_features",
    "roc_auc_mean",
    "roc_auc_std",
    "f1_mean",
    "sensitivity_mean",
    "specificity_mean",
    "average_precision_mean",
]

THRESHOLDS = np.round(np.arange(0.05, 1.00, 0.05), 2)

XGBOOST_CONFIGS = {
    "config_1": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 3,
        "min_child_weight": 1,
        "reg_lambda": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "config_2": {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 3,
        "min_child_weight": 1,
        "reg_lambda": 1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "config_3": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 2,
        "min_child_weight": 1,
        "reg_lambda": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "config_4": {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "max_depth": 2,
        "min_child_weight": 1,
        "reg_lambda": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
    },
    "config_5": {
        "n_estimators": 300,
        "learning_rate": 0.05,
        "max_depth": 4,
        "min_child_weight": 3,
        "reg_lambda": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    },
    "config_6": {
        "n_estimators": 500,
        "learning_rate": 0.02,
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
    seen = set()
    result = []

    for column in columns:
        if column not in seen:
            result.append(column)
            seen.add(column)

    return result


def is_numeric_feature(features: pd.DataFrame, column: str) -> bool:
    return column in features.columns and pd.api.types.is_numeric_dtype(features[column])


def columns_with_prefixes(features: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [
        column
        for column in features.columns
        if column.startswith(prefixes)
        and column not in EXCLUDED_COLUMNS
        and is_numeric_feature(features, column)
    ]


def filter_feature_columns(features: pd.DataFrame, columns: list[str]) -> list[str]:
    return [
        column
        for column in unique_preserve_order(columns)
        if column not in EXCLUDED_COLUMNS and is_numeric_feature(features, column)
    ]


def build_feature_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    advanced_spo2 = columns_with_prefixes(
        features,
        (
            "spo2_",
            "rn_spo2_",
            "prev1_spo2_",
            "next1_spo2_",
            "lag1_spo2_",
            "lag2_spo2_",
            "lead1_spo2_",
            "lead2_spo2_",
            "roll3_mean_spo2_",
            "roll3_min_spo2_",
            "roll3_max_spo2_",
            "roll5_mean_spo2_",
            "roll5_min_spo2_",
            "roll5_max_spo2_",
            "roll5_median_spo2_",
            "delta1_spo2_",
            "delta2_spo2_",
            "lead_delta1_spo2_",
            "lead_delta2_spo2_",
            "drop_from_roll5_median_spo2_",
        ),
    )

    advanced_respiratory = columns_with_prefixes(
        features,
        (
            "flow_",
            "rn_flow_",
            "ribcage_",
            "rn_ribcage_",
            "abdo_",
            "rn_abdo_",
            "effort_",
            "rn_effort_",
            "prev1_flow_",
            "next1_flow_",
            "lag1_flow_",
            "lag2_flow_",
            "lead1_flow_",
            "lead2_flow_",
            "roll3_mean_flow_",
            "roll3_min_flow_",
            "roll3_max_flow_",
            "roll5_mean_flow_",
            "roll5_min_flow_",
            "roll5_max_flow_",
            "prev1_ribcage_",
            "next1_ribcage_",
            "lag1_ribcage_",
            "lag2_ribcage_",
            "lead1_ribcage_",
            "lead2_ribcage_",
            "roll3_mean_ribcage_",
            "roll3_min_ribcage_",
            "roll3_max_ribcage_",
            "roll5_mean_ribcage_",
            "roll5_min_ribcage_",
            "roll5_max_ribcage_",
            "prev1_abdo_",
            "next1_abdo_",
            "lag1_abdo_",
            "lag2_abdo_",
            "lead1_abdo_",
            "lead2_abdo_",
            "roll3_mean_abdo_",
            "roll3_min_abdo_",
            "roll3_max_abdo_",
            "roll5_mean_abdo_",
            "roll5_min_abdo_",
            "roll5_max_abdo_",
            "prev1_effort_",
            "next1_effort_",
            "lag1_effort_",
            "lag2_effort_",
            "lead1_effort_",
            "lead2_effort_",
            "roll3_mean_effort_",
            "roll3_min_effort_",
            "roll3_max_effort_",
            "roll5_mean_effort_",
            "roll5_min_effort_",
            "roll5_max_effort_",
            "flow_energy_to_",
            "flow_abs_to_",
            "ribcage_to_abdo_energy_ratio",
            "flow_low_amp_x_",
        ),
    )

    ecg_features = columns_with_prefixes(features, ("ecg_", "rn_ecg_"))
    advanced_respiratory_spo2 = filter_feature_columns(
        features,
        [*advanced_respiratory, *advanced_spo2],
    )
    advanced_full = filter_feature_columns(
        features,
        [*advanced_respiratory_spo2, *ecg_features],
    )

    return {
        "advanced_spo2": advanced_spo2,
        "advanced_respiratory": advanced_respiratory,
        "advanced_respiratory_spo2": advanced_respiratory_spo2,
        "advanced_full": advanced_full,
        "advanced_late_fusion": advanced_full,
        "ecg_component": ecg_features,
    }


def make_cv_splits(
    features: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
) -> list[tuple[np.ndarray, np.ndarray]]:
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        return list(splitter.split(features, y, groups))
    except ImportError:
        splitter = GroupKFold(n_splits=N_SPLITS)
        return list(splitter.split(features, y, groups))


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())

    if n_positive == 0:
        return 1.0

    return n_negative / n_positive


def make_xgboost_pipeline(
    config: dict[str, int | float],
    scale_pos_weight: float,
) -> Pipeline:
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
                    scale_pos_weight=scale_pos_weight,
                ),
            ),
        ]
    )


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


def calculate_metrics(
    y_true: pd.Series,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

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


def make_result_row(
    regime: str,
    experiment: str,
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
        "experiment": experiment,
        "model": MODEL_NAME,
        "config_name": config_name,
        "fold": fold,
        "n_features": n_features,
        "n_train": int(len(train_indices)),
        "n_valid": int(len(valid_indices)),
        "n_train_records": int(groups.iloc[train_indices].nunique()),
        "n_valid_records": int(groups.iloc[valid_indices].nunique()),
        **metrics,
    }


def make_prediction_rows(
    metadata: pd.DataFrame,
    regime: str,
    experiment: str,
    config_name: str,
    fold: int,
    y_proba: np.ndarray,
    y_pred: np.ndarray,
) -> list[dict[str, object]]:
    rows = []

    for row_index, (_, row) in enumerate(metadata.iterrows()):
        rows.append(
            {
                "record_id": row["record_id"],
                "epoch_id": row["epoch_id"],
                "label_binary": int(row["label_binary"]),
                "regime": regime,
                "experiment": experiment,
                "model": MODEL_NAME,
                "config_name": config_name,
                "fold": fold,
                "y_proba": float(y_proba[row_index]),
                "y_pred": int(y_pred[row_index]),
            }
        )

    return rows


def predict_positive_probability(model: Pipeline, X_valid: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X_valid)[:, 1]


def run_single_experiment_cv(
    features: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    regime: str,
    experiment: str,
    feature_columns: list[str],
    config_name: str,
    config: dict[str, int | float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result_rows = []
    prediction_rows = []

    if not feature_columns:
        print(f"Skipping {regime}/{experiment}/{config_name}: no features.")
        return result_rows, prediction_rows

    X = features[feature_columns]

    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        X_train = X.iloc[train_indices]
        X_valid = X.iloc[valid_indices]
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]

        model = make_xgboost_pipeline(config, calculate_scale_pos_weight(y_train))
        model.fit(X_train, y_train)

        y_proba = predict_positive_probability(model, X_valid)
        y_pred = (y_proba >= THRESHOLD).astype(int)
        metrics = calculate_metrics(y_valid, y_proba, y_pred)

        result_rows.append(
            make_result_row(
                regime=regime,
                experiment=experiment,
                config_name=config_name,
                fold=fold,
                n_features=len(feature_columns),
                train_indices=train_indices,
                valid_indices=valid_indices,
                groups=groups,
                metrics=metrics,
            )
        )

        metadata = features.iloc[valid_indices][["record_id", "epoch_id", "label_binary"]]
        prediction_rows.extend(
            make_prediction_rows(
                metadata=metadata,
                regime=regime,
                experiment=experiment,
                config_name=config_name,
                fold=fold,
                y_proba=y_proba,
                y_pred=y_pred,
            )
        )

    return result_rows, prediction_rows


def run_late_fusion_cv(
    features: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    regime: str,
    feature_sets: dict[str, list[str]],
    config_name: str,
    config: dict[str, int | float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result_rows = []
    prediction_rows = []

    component_sets = [
        feature_sets["advanced_spo2"],
        feature_sets["advanced_respiratory"],
        feature_sets["ecg_component"],
    ]
    if any(not columns for columns in component_sets):
        print(f"Skipping {regime}/advanced_late_fusion/{config_name}: missing component.")
        return result_rows, prediction_rows

    n_features = len(unique_preserve_order([column for columns in component_sets for column in columns]))

    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]
        component_probabilities = []

        for feature_columns in component_sets:
            X_train = features.iloc[train_indices][feature_columns]
            X_valid = features.iloc[valid_indices][feature_columns]

            model = make_xgboost_pipeline(config, calculate_scale_pos_weight(y_train))
            model.fit(X_train, y_train)
            component_probabilities.append(predict_positive_probability(model, X_valid))

        y_proba = np.mean(np.vstack(component_probabilities), axis=0)
        y_pred = (y_proba >= THRESHOLD).astype(int)
        metrics = calculate_metrics(y_valid, y_proba, y_pred)

        result_rows.append(
            make_result_row(
                regime=regime,
                experiment="advanced_late_fusion",
                config_name=config_name,
                fold=fold,
                n_features=n_features,
                train_indices=train_indices,
                valid_indices=valid_indices,
                groups=groups,
                metrics=metrics,
            )
        )

        metadata = features.iloc[valid_indices][["record_id", "epoch_id", "label_binary"]]
        prediction_rows.extend(
            make_prediction_rows(
                metadata=metadata,
                regime=regime,
                experiment="advanced_late_fusion",
                config_name=config_name,
                fold=fold,
                y_proba=y_proba,
                y_pred=y_pred,
            )
        )

    return result_rows, prediction_rows


def summarize_results(results_by_fold: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    group_columns = ["regime", "experiment", "model", "config_name"]

    for keys, group in results_by_fold.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, keys))
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
        results_by_fold.groupby(
            ["regime", "experiment", "model", "config_name"],
            as_index=False,
        )[CONFUSION_COLUMNS]
        .sum()
        .sort_values(["regime", "experiment", "model", "config_name"])
    )


def build_best_results(results_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        results_summary[BEST_RESULT_COLUMNS]
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )


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
    group_columns = ["regime", "experiment", "model", "config_name"]

    for keys, group in predictions.groupby(group_columns, sort=True):
        key_values = dict(zip(group_columns, keys))
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


def print_regime_balance(features: pd.DataFrame, regime_name: str) -> None:
    y = features["label_binary"].astype(int)
    n_positive = int((y == 1).sum())
    n_negative = int((y == 0).sum())
    positive_rate = n_positive / len(y) if len(y) else 0.0

    print(f"{regime_name}")
    print(f"  rows: {len(features)}")
    print(f"  normal: {n_negative}")
    print(f"  apnea_hypopnea: {n_positive}")
    print(f"  positive_rate: {positive_rate:.4f}")


def print_feature_set_sizes(features: pd.DataFrame) -> None:
    feature_sets = build_feature_sets(features)
    print("Feature set sizes")
    for name in [
        "advanced_spo2",
        "advanced_respiratory",
        "advanced_respiratory_spo2",
        "advanced_full",
        "advanced_late_fusion",
    ]:
        print(f"  {name}: {len(feature_sets[name])}")


def run_regime(
    features: pd.DataFrame,
    regime: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    y = features["label_binary"].astype(int)
    groups = features["record_id"].astype(str)
    splits = make_cv_splits(features, y, groups)
    feature_sets = build_feature_sets(features)

    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for config_name, config in XGBOOST_CONFIGS.items():
        for experiment in [
            "advanced_spo2",
            "advanced_respiratory",
            "advanced_respiratory_spo2",
            "advanced_full",
        ]:
            print(f"\nTraining {regime} / {experiment} / {config_name}")
            rows, predictions = run_single_experiment_cv(
                features=features,
                y=y,
                groups=groups,
                splits=splits,
                regime=regime,
                experiment=experiment,
                feature_columns=feature_sets[experiment],
                config_name=config_name,
                config=config,
            )
            result_rows.extend(rows)
            prediction_rows.extend(predictions)

        print(f"\nTraining {regime} / advanced_late_fusion / {config_name}")
        rows, predictions = run_late_fusion_cv(
            features=features,
            y=y,
            groups=groups,
            splits=splits,
            regime=regime,
            feature_sets=feature_sets,
            config_name=config_name,
            config=config,
        )
        result_rows.extend(rows)
        prediction_rows.extend(predictions)

    return result_rows, prediction_rows


def print_best_summary(best_results: pd.DataFrame, best_thresholds: pd.DataFrame) -> None:
    best_auc = best_results.iloc[0]
    best_f1 = best_results.sort_values("f1_mean", ascending=False).iloc[0]
    tuned_rows = best_thresholds[best_thresholds["selection_rule"] == "max_f1"]
    best_tuned_f1 = tuned_rows.sort_values("f1", ascending=False).iloc[0]

    print("Best result by roc_auc_mean")
    print(
        f"  {best_auc['regime']} / {best_auc['experiment']} / {best_auc['config_name']}: "
        f"roc_auc_mean={best_auc['roc_auc_mean']:.4f}, "
        f"delta_vs_previous={best_auc['roc_auc_mean'] - PREVIOUS_BEST_AUC:+.4f}"
    )
    print("Best result by f1_mean at threshold=0.5")
    print(
        f"  {best_f1['regime']} / {best_f1['experiment']} / {best_f1['config_name']}: "
        f"f1_mean={best_f1['f1_mean']:.4f}, "
        f"delta_vs_previous={best_f1['f1_mean'] - PREVIOUS_BEST_F1:+.4f}"
    )
    print("Best result by max_f1 threshold sweep")
    print(
        f"  {best_tuned_f1['regime']} / {best_tuned_f1['experiment']} / "
        f"{best_tuned_f1['config_name']} @ threshold={best_tuned_f1['threshold']:.2f}: "
        f"f1={best_tuned_f1['f1']:.4f}, "
        f"delta_vs_previous={best_tuned_f1['f1'] - PREVIOUS_THRESHOLD_TUNED_F1:+.4f}"
    )


def main() -> None:
    if not FEATURES_PATH.exists():
        raise SystemExit(
            f"Advanced feature table not found: {FEATURES_PATH}. "
            "Run: python scripts/experiments/03_build_advanced_features.py"
        )

    import_xgboost()
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURES_PATH)
    all_epochs = features.copy()
    sleep_only = features[features["is_sleep_epoch"] == 1].copy()

    print("Data regimes")
    print_regime_balance(all_epochs, "all_epochs")
    print_regime_balance(sleep_only, "sleep_only")
    print_feature_set_sizes(all_epochs)
    print("Previous reference metrics")
    print(f"  previous best AUC: {PREVIOUS_BEST_AUC:.4f}")
    print(f"  previous best F1: {PREVIOUS_BEST_F1:.4f}")
    print(f"  previous threshold-tuned F1: {PREVIOUS_THRESHOLD_TUNED_F1:.4f}")

    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for regime, regime_features in [
        ("all_epochs", all_epochs),
        ("sleep_only", sleep_only),
    ]:
        rows, predictions = run_regime(regime_features, regime)
        result_rows.extend(rows)
        prediction_rows.extend(predictions)

    results_by_fold = pd.DataFrame(result_rows, columns=RESULT_BY_FOLD_COLUMNS)
    predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    results_summary = summarize_results(results_by_fold)
    confusion_matrices = summarize_confusion_matrices(results_by_fold)
    best_results = build_best_results(results_summary)
    best_thresholds = build_best_thresholds(predictions)

    results_by_fold.to_csv(RESULTS_BY_FOLD_PATH, index=False)
    results_summary.to_csv(RESULTS_SUMMARY_PATH, index=False)
    confusion_matrices.to_csv(CONFUSION_MATRICES_PATH, index=False)
    predictions.to_csv(CV_PREDICTIONS_PATH, index=False)
    best_results.to_csv(BEST_RESULTS_PATH, index=False)
    best_thresholds.to_csv(BEST_THRESHOLDS_PATH, index=False)

    print()
    print_best_summary(best_results, best_thresholds)
    print(f"Results summary: {RESULTS_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
