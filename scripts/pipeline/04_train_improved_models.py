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
from scripts.mlops.tracking import (  # noqa: E402
    log_artifacts,
    log_metrics,
    log_params,
    mlflow_run,
)


RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5

FEATURES_PATH = DATA_PROCESSED_DIR / "features_model_ready.csv"
RESULTS_BY_FOLD_PATH = REPORTS_TABLES_DIR / "improved_model_results_by_fold.csv"
RESULTS_SUMMARY_PATH = REPORTS_TABLES_DIR / "improved_model_results_summary.csv"
CONFUSION_MATRICES_PATH = REPORTS_TABLES_DIR / "improved_model_confusion_matrices.csv"
CV_PREDICTIONS_PATH = REPORTS_TABLES_DIR / "improved_cv_predictions.csv"
BEST_RESULTS_PATH = REPORTS_TABLES_DIR / "improved_best_results.csv"

MODEL_NAME = "XGBoost"

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
    "feature_variant",
    "experiment",
    "model",
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
    "feature_variant",
    "experiment",
    "model",
    "fold",
    "y_proba",
    "y_pred",
]

BEST_RESULT_COLUMNS = [
    "regime",
    "feature_variant",
    "experiment",
    "model",
    "n_features",
    "roc_auc_mean",
    "roc_auc_std",
    "f1_mean",
    "sensitivity_mean",
    "specificity_mean",
    "average_precision_mean",
]


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


def filter_valid_feature_columns(
    features: pd.DataFrame,
    columns: list[str],
) -> list[str]:
    return [
        column
        for column in unique_preserve_order(columns)
        if column not in EXCLUDED_COLUMNS and is_numeric_feature(features, column)
    ]


def columns_with_prefixes(features: pd.DataFrame, prefixes: tuple[str, ...]) -> list[str]:
    return [
        column
        for column in features.columns
        if column.startswith(prefixes) and column not in EXCLUDED_COLUMNS
    ]


def build_feature_groups(
    features: pd.DataFrame,
    feature_variant: str,
) -> dict[str, list[str]]:
    base_ecg = columns_with_prefixes(features, ("ecg_",))
    base_flow = columns_with_prefixes(features, ("flow_",))
    base_effort = columns_with_prefixes(features, ("ribcage_", "abdo_", "effort_"))
    base_spo2 = columns_with_prefixes(features, ("spo2_",))

    if feature_variant == "base":
        return {
            "ecg": filter_valid_feature_columns(features, base_ecg),
            "flow": filter_valid_feature_columns(features, base_flow),
            "effort": filter_valid_feature_columns(features, base_effort),
            "spo2": filter_valid_feature_columns(features, base_spo2),
        }

    if feature_variant != "enhanced":
        raise ValueError(f"Unsupported feature variant: {feature_variant}")

    flow_context_prefixes = (
        "prev1_flow_",
        "next1_flow_",
        "roll3_mean_flow_",
        "roll3_min_flow_",
        "roll3_max_flow_",
    )
    effort_context_prefixes = (
        "prev1_ribcage_",
        "next1_ribcage_",
        "roll3_mean_ribcage_",
        "roll3_min_ribcage_",
        "roll3_max_ribcage_",
        "prev1_abdo_",
        "next1_abdo_",
        "roll3_mean_abdo_",
        "roll3_min_abdo_",
        "roll3_max_abdo_",
        "prev1_effort_",
        "next1_effort_",
        "roll3_mean_effort_",
        "roll3_min_effort_",
        "roll3_max_effort_",
    )
    spo2_context_prefixes = (
        "prev1_spo2_",
        "next1_spo2_",
        "roll3_mean_spo2_",
        "roll3_min_spo2_",
        "roll3_max_spo2_",
    )

    enhanced_ecg = [
        *base_ecg,
        *columns_with_prefixes(features, ("rn_ecg_",)),
    ]
    enhanced_flow = [
        *base_flow,
        *columns_with_prefixes(features, ("rn_flow_",)),
        *columns_with_prefixes(features, flow_context_prefixes),
    ]
    enhanced_effort = [
        *base_effort,
        *columns_with_prefixes(features, ("rn_ribcage_", "rn_abdo_", "rn_effort_")),
        *columns_with_prefixes(features, effort_context_prefixes),
    ]
    enhanced_spo2 = [
        *base_spo2,
        *columns_with_prefixes(features, ("rn_spo2_",)),
        *columns_with_prefixes(features, spo2_context_prefixes),
    ]

    return {
        "ecg": filter_valid_feature_columns(features, enhanced_ecg),
        "flow": filter_valid_feature_columns(features, enhanced_flow),
        "effort": filter_valid_feature_columns(features, enhanced_effort),
        "spo2": filter_valid_feature_columns(features, enhanced_spo2),
    }


def build_experiments(feature_groups: dict[str, list[str]]) -> dict[str, list[str]]:
    flow = feature_groups["flow"]
    effort = feature_groups["effort"]
    spo2 = feature_groups["spo2"]
    ecg = feature_groups["ecg"]

    return {
        "flow_only": flow,
        "effort_only": effort,
        "spo2_only": spo2,
        "respiratory_spo2_fusion": unique_preserve_order([*flow, *effort, *spo2]),
        "full_core_fusion": unique_preserve_order([*ecg, *flow, *effort, *spo2]),
        "late_fusion": unique_preserve_order([*ecg, *flow, *effort, *spo2]),
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


def make_xgboost_pipeline(scale_pos_weight: float) -> Pipeline:
    XGBClassifier = import_xgboost()

    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=3,
                    subsample=0.8,
                    colsample_bytree=0.8,
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


def predict_positive_probability(model: Pipeline, X_valid: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X_valid)[:, 1]


def make_result_row(
    regime: str,
    feature_variant: str,
    experiment: str,
    fold: int,
    n_features: int,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    groups: pd.Series,
    metrics: dict[str, float | int],
) -> dict[str, object]:
    return {
        "regime": regime,
        "feature_variant": feature_variant,
        "experiment": experiment,
        "model": MODEL_NAME,
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
    feature_variant: str,
    experiment: str,
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
                "feature_variant": feature_variant,
                "experiment": experiment,
                "model": MODEL_NAME,
                "fold": fold,
                "y_proba": float(y_proba[row_index]),
                "y_pred": int(y_pred[row_index]),
            }
        )

    return rows


def run_xgboost_cv(
    features: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    regime: str,
    feature_variant: str,
    experiment: str,
    feature_columns: list[str],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result_rows = []
    prediction_rows = []

    if not feature_columns:
        print(f"Skipping {regime}/{feature_variant}/{experiment}: no features.")
        return result_rows, prediction_rows

    X = features[feature_columns]

    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        X_train = X.iloc[train_indices]
        X_valid = X.iloc[valid_indices]
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]

        model = make_xgboost_pipeline(calculate_scale_pos_weight(y_train))
        model.fit(X_train, y_train)

        y_proba = predict_positive_probability(model, X_valid)
        y_pred = (y_proba >= THRESHOLD).astype(int)
        metrics = calculate_metrics(y_valid, y_proba, y_pred)

        result_rows.append(
            make_result_row(
                regime=regime,
                feature_variant=feature_variant,
                experiment=experiment,
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
                feature_variant=feature_variant,
                experiment=experiment,
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
    feature_variant: str,
    feature_groups: dict[str, list[str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result_rows = []
    prediction_rows = []

    ecg_features = feature_groups["ecg"]
    respiratory_features = unique_preserve_order(
        [*feature_groups["flow"], *feature_groups["effort"]]
    )
    spo2_features = feature_groups["spo2"]
    component_features = [ecg_features, respiratory_features, spo2_features]

    if any(not columns for columns in component_features):
        print(f"Skipping {regime}/{feature_variant}/late_fusion: missing component.")
        return result_rows, prediction_rows

    n_features = len(
        unique_preserve_order([column for columns in component_features for column in columns])
    )

    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]
        component_probabilities = []

        for feature_columns in component_features:
            X_train = features.iloc[train_indices][feature_columns]
            X_valid = features.iloc[valid_indices][feature_columns]

            model = make_xgboost_pipeline(calculate_scale_pos_weight(y_train))
            model.fit(X_train, y_train)
            component_probabilities.append(predict_positive_probability(model, X_valid))

        y_proba = np.mean(np.vstack(component_probabilities), axis=0)
        y_pred = (y_proba >= THRESHOLD).astype(int)
        metrics = calculate_metrics(y_valid, y_proba, y_pred)

        result_rows.append(
            make_result_row(
                regime=regime,
                feature_variant=feature_variant,
                experiment="late_fusion",
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
                feature_variant=feature_variant,
                experiment="late_fusion",
                fold=fold,
                y_proba=y_proba,
                y_pred=y_pred,
            )
        )

    return result_rows, prediction_rows


def summarize_results(results_by_fold: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    group_columns = ["regime", "feature_variant", "experiment", "model"]
    for group_keys, group in results_by_fold.groupby(group_columns, sort=True):
        row = dict(zip(group_columns, group_keys))
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
            ["regime", "feature_variant", "experiment", "model"],
            as_index=False,
        )[CONFUSION_COLUMNS]
        .sum()
        .sort_values(["regime", "feature_variant", "experiment", "model"])
    )


def build_best_results(results_summary: pd.DataFrame) -> pd.DataFrame:
    return (
        results_summary[BEST_RESULT_COLUMNS]
        .sort_values("roc_auc_mean", ascending=False)
        .reset_index(drop=True)
    )


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


def print_experiment_sizes(
    all_features: pd.DataFrame,
    sleep_features: pd.DataFrame,
) -> None:
    print("Experiment feature counts")

    for feature_variant in ["base", "enhanced"]:
        groups = build_feature_groups(all_features, feature_variant)
        experiments = build_experiments(groups)
        print(f"  all_epochs / {feature_variant}")
        for experiment, feature_columns in experiments.items():
            print(f"    {experiment}: {len(feature_columns)}")

    for feature_variant in ["base", "enhanced"]:
        groups = build_feature_groups(sleep_features, feature_variant)
        experiments = build_experiments(groups)
        print(f"  sleep_only / {feature_variant}")
        for experiment, feature_columns in experiments.items():
            print(f"    {experiment}: {len(feature_columns)}")


def print_best_results(best_results: pd.DataFrame) -> None:
    if best_results.empty:
        print("No best results available.")
        return

    best_roc_auc = best_results.iloc[0]
    best_f1 = best_results.sort_values("f1_mean", ascending=False).iloc[0]

    print("Best result by roc_auc_mean")
    print(
        f"  {best_roc_auc['regime']} / {best_roc_auc['feature_variant']} / "
        f"{best_roc_auc['experiment']}: "
        f"roc_auc_mean={best_roc_auc['roc_auc_mean']:.4f}, "
        f"f1_mean={best_roc_auc['f1_mean']:.4f}"
    )
    print("Best result by f1_mean")
    print(
        f"  {best_f1['regime']} / {best_f1['feature_variant']} / "
        f"{best_f1['experiment']}: "
        f"f1_mean={best_f1['f1_mean']:.4f}, "
        f"roc_auc_mean={best_f1['roc_auc_mean']:.4f}"
    )


def run_regime_variant(
    features: pd.DataFrame,
    regime: str,
    feature_variant: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    y = features["label_binary"].astype(int)
    groups = features["record_id"].astype(str)
    splits = make_cv_splits(features, y, groups)
    feature_groups = build_feature_groups(features, feature_variant)
    experiments = build_experiments(feature_groups)

    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for experiment in [
        "flow_only",
        "effort_only",
        "spo2_only",
        "respiratory_spo2_fusion",
        "full_core_fusion",
    ]:
        print(f"\nTraining {regime} / {feature_variant} / {experiment}")
        rows, predictions = run_xgboost_cv(
            features=features,
            y=y,
            groups=groups,
            splits=splits,
            regime=regime,
            feature_variant=feature_variant,
            experiment=experiment,
            feature_columns=experiments[experiment],
        )
        result_rows.extend(rows)
        prediction_rows.extend(predictions)

    print(f"\nTraining {regime} / {feature_variant} / late_fusion")
    rows, predictions = run_late_fusion_cv(
        features=features,
        y=y,
        groups=groups,
        splits=splits,
        regime=regime,
        feature_variant=feature_variant,
        feature_groups=feature_groups,
    )
    result_rows.extend(rows)
    prediction_rows.extend(predictions)

    return result_rows, prediction_rows


def log_mlflow_summary(
    features: pd.DataFrame,
    all_epochs: pd.DataFrame,
    sleep_only: pd.DataFrame,
    results_summary: pd.DataFrame,
    best_results: pd.DataFrame,
) -> None:
    if best_results.empty:
        return

    best_auc = best_results.sort_values("roc_auc_mean", ascending=False).iloc[0]
    best_f1 = best_results.sort_values("f1_mean", ascending=False).iloc[0]

    with mlflow_run(
        "train_improved_models",
        tags={
            "stage": "training",
            "pipeline": "improved_models",
            "validation": "subject_level_cv",
        },
    ) as mlflow:
        if mlflow is None:
            return

        log_params(
            mlflow,
            {
                "model": MODEL_NAME,
                "random_state": RANDOM_STATE,
                "n_splits": N_SPLITS,
                "threshold": THRESHOLD,
                "n_rows_total": len(features),
                "n_rows_all_epochs": len(all_epochs),
                "n_rows_sleep_only": len(sleep_only),
                "n_records": features["record_id"].nunique(),
                "n_summary_rows": len(results_summary),
                "best_auc_regime": best_auc["regime"],
                "best_auc_feature_variant": best_auc["feature_variant"],
                "best_auc_experiment": best_auc["experiment"],
                "best_f1_regime": best_f1["regime"],
                "best_f1_feature_variant": best_f1["feature_variant"],
                "best_f1_experiment": best_f1["experiment"],
            },
        )
        log_metrics(
            mlflow,
            {
                "best_roc_auc_mean": best_auc["roc_auc_mean"],
                "best_roc_auc_std": best_auc["roc_auc_std"],
                "best_f1_mean": best_f1["f1_mean"],
                "best_sensitivity_mean": best_f1["sensitivity_mean"],
                "best_specificity_mean": best_f1["specificity_mean"],
                "best_average_precision_mean": best_auc["average_precision_mean"],
            },
        )
        log_artifacts(
            mlflow,
            [
                RESULTS_BY_FOLD_PATH,
                RESULTS_SUMMARY_PATH,
                CONFUSION_MATRICES_PATH,
                CV_PREDICTIONS_PATH,
                BEST_RESULTS_PATH,
            ],
        )


def main() -> None:
    if not FEATURES_PATH.exists():
        raise SystemExit(
            f"Model-ready feature table not found: {FEATURES_PATH}. "
            "Run: python scripts/pipeline/03_build_model_ready_features.py"
        )

    # Fail early with a clear message before the CV loop starts.
    import_xgboost()

    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURES_PATH)
    all_epochs = features.copy()
    sleep_only = features[features["is_sleep_epoch"] == 1].copy()

    print("Data regimes")
    print_regime_balance(all_epochs, "all_epochs")
    print_regime_balance(sleep_only, "sleep_only")
    print_experiment_sizes(all_epochs, sleep_only)

    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    for regime, regime_features in [
        ("all_epochs", all_epochs),
        ("sleep_only", sleep_only),
    ]:
        for feature_variant in ["base", "enhanced"]:
            rows, predictions = run_regime_variant(
                features=regime_features,
                regime=regime,
                feature_variant=feature_variant,
            )
            result_rows.extend(rows)
            prediction_rows.extend(predictions)

    results_by_fold = pd.DataFrame(result_rows, columns=RESULT_BY_FOLD_COLUMNS)
    cv_predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    results_summary = summarize_results(results_by_fold)
    confusion_matrices = summarize_confusion_matrices(results_by_fold)
    best_results = build_best_results(results_summary)

    results_by_fold.to_csv(RESULTS_BY_FOLD_PATH, index=False)
    results_summary.to_csv(RESULTS_SUMMARY_PATH, index=False)
    confusion_matrices.to_csv(CONFUSION_MATRICES_PATH, index=False)
    cv_predictions.to_csv(CV_PREDICTIONS_PATH, index=False)
    best_results.to_csv(BEST_RESULTS_PATH, index=False)

    log_mlflow_summary(features, all_epochs, sleep_only, results_summary, best_results)

    print()
    print_best_results(best_results)
    print(f"Results summary: {RESULTS_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
