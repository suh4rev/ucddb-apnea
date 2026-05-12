from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, REPORTS_TABLES_DIR  # noqa: E402


RANDOM_STATE = 42
N_SPLITS = 5
THRESHOLD = 0.5

FEATURES_PATH = DATA_PROCESSED_DIR / "features_all.csv"
RESULTS_BY_FOLD_PATH = REPORTS_TABLES_DIR / "model_results_by_fold.csv"
RESULTS_SUMMARY_PATH = REPORTS_TABLES_DIR / "model_results_summary.csv"
CONFUSION_MATRICES_PATH = REPORTS_TABLES_DIR / "model_confusion_matrices.csv"
CV_PREDICTIONS_PATH = REPORTS_TABLES_DIR / "cv_predictions.csv"
ABLATION_SUMMARY_PATH = REPORTS_TABLES_DIR / "ablation_results_summary.csv"
LATE_FUSION_SUMMARY_PATH = REPORTS_TABLES_DIR / "late_fusion_results_summary.csv"

LEAKAGE_COLUMNS = {
    "record_id",
    "epoch_id",
    "start_sec",
    "end_sec",
    "sleep_stage",
    "label",
    "label_binary",
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
    "experiment",
    "model",
    "fold",
    "y_proba",
    "y_pred",
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


def get_feature_groups(features: pd.DataFrame) -> dict[str, list[str]]:
    return {
        "ecg_features": [
            column for column in features.columns if column.startswith("ecg_")
        ],
        "flow_features": [
            column for column in features.columns if column.startswith("flow_")
        ],
        "ribcage_features": [
            column for column in features.columns if column.startswith("ribcage_")
        ],
        "abdo_features": [
            column for column in features.columns if column.startswith("abdo_")
        ],
        "effort_features": [
            column for column in features.columns if column.startswith("effort_")
        ],
        "spo2_features": [
            column for column in features.columns if column.startswith("spo2_")
        ],
    }


def build_experiments(feature_groups: dict[str, list[str]]) -> dict[str, list[str]]:
    ecg = feature_groups["ecg_features"]
    flow = feature_groups["flow_features"]
    ribcage = feature_groups["ribcage_features"]
    abdo = feature_groups["abdo_features"]
    effort = feature_groups["effort_features"]
    spo2 = feature_groups["spo2_features"]

    effort_all = unique_preserve_order([*ribcage, *abdo, *effort])
    respiratory_fusion = unique_preserve_order([*flow, *effort_all])
    respiratory_spo2_fusion = unique_preserve_order([*respiratory_fusion, *spo2])
    full_core_fusion = unique_preserve_order([*ecg, *respiratory_fusion, *spo2])

    return {
        "ecg_only": ecg,
        "flow_only": flow,
        "effort_only": effort_all,
        "spo2_only": spo2,
        "respiratory_fusion": respiratory_fusion,
        "respiratory_spo2_fusion": respiratory_spo2_fusion,
        "full_core_fusion": full_core_fusion,
        "full_minus_ecg": respiratory_spo2_fusion,
        "full_minus_flow": unique_preserve_order([*ecg, *effort_all, *spo2]),
        "full_minus_effort": unique_preserve_order([*ecg, *flow, *spo2]),
        "full_minus_spo2": unique_preserve_order([*ecg, *respiratory_fusion]),
    }


def make_cv_splits(
    X: pd.DataFrame,
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
        return list(splitter.split(X, y, groups))
    except ImportError:
        splitter = GroupKFold(n_splits=N_SPLITS)
        return list(splitter.split(X, y, groups))


def make_logistic_regression_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
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
                    n_estimators=300,
                    max_depth=None,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )


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


def make_model_pipeline(model_name: str, y_train: pd.Series) -> Pipeline:
    if model_name == "LogisticRegression":
        return make_logistic_regression_pipeline()

    if model_name == "RandomForest":
        return make_random_forest_pipeline()

    if model_name == "XGBoost":
        return make_xgboost_pipeline(calculate_scale_pos_weight(y_train))

    raise ValueError(f"Unsupported model: {model_name}")


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
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_valid)[:, 1]

    scores = model.decision_function(X_valid)
    return 1.0 / (1.0 + np.exp(-scores))


def make_result_row(
    experiment: str,
    model_name: str,
    fold: int,
    n_features: int,
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    groups: pd.Series,
    metrics: dict[str, float | int],
) -> dict[str, object]:
    return {
        "experiment": experiment,
        "model": model_name,
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
    experiment: str,
    model_name: str,
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
                "experiment": experiment,
                "model": model_name,
                "fold": fold,
                "y_proba": float(y_proba[row_index]),
                "y_pred": int(y_pred[row_index]),
            }
        )

    return rows


def run_single_model_cv(
    features: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    splits: list[tuple[np.ndarray, np.ndarray]],
    experiment: str,
    feature_columns: list[str],
    model_name: str,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    result_rows = []
    prediction_rows = []

    if not feature_columns:
        print(f"Skipping {experiment} / {model_name}: no feature columns.")
        return result_rows, prediction_rows

    X = features[feature_columns]

    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        X_train = X.iloc[train_indices]
        X_valid = X.iloc[valid_indices]
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]

        model = make_model_pipeline(model_name, y_train)
        model.fit(X_train, y_train)

        y_proba = predict_positive_probability(model, X_valid)
        y_pred = (y_proba >= THRESHOLD).astype(int)
        metrics = calculate_metrics(y_valid, y_proba, y_pred)

        result_rows.append(
            make_result_row(
                experiment=experiment,
                model_name=model_name,
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
                experiment=experiment,
                model_name=model_name,
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
    experiments: dict[str, list[str]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    component_experiments = ["ecg_only", "respiratory_fusion", "spo2_only"]
    result_rows = []
    prediction_rows = []

    for component in component_experiments:
        if not experiments[component]:
            print(f"Skipping late_fusion: no features for {component}.")
            return result_rows, prediction_rows

    n_features = len(
        unique_preserve_order(
            [
                column
                for component in component_experiments
                for column in experiments[component]
            ]
        )
    )

    for fold, (train_indices, valid_indices) in enumerate(splits, start=1):
        y_train = y.iloc[train_indices]
        y_valid = y.iloc[valid_indices]
        component_probabilities = []

        for component in component_experiments:
            feature_columns = experiments[component]
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
                experiment="late_fusion",
                model_name="XGBoost",
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
                experiment="late_fusion",
                model_name="XGBoost",
                fold=fold,
                y_proba=y_proba,
                y_pred=y_pred,
            )
        )

    return result_rows, prediction_rows


def summarize_results(results_by_fold: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []

    for (experiment, model_name), group in results_by_fold.groupby(
        ["experiment", "model"],
        sort=True,
    ):
        row: dict[str, object] = {
            "experiment": experiment,
            "model": model_name,
            "n_features": int(group["n_features"].iloc[0]),
        }

        for metric in METRIC_COLUMNS:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))

        for column in CONFUSION_COLUMNS:
            row[f"{column}_mean"] = float(group[column].mean())

        summary_rows.append(row)

    return pd.DataFrame(summary_rows)


def summarize_confusion_matrices(results_by_fold: pd.DataFrame) -> pd.DataFrame:
    return (
        results_by_fold.groupby(["experiment", "model"], as_index=False)[
            CONFUSION_COLUMNS
        ]
        .sum()
        .sort_values(["experiment", "model"])
    )


def build_ablation_summary(results_summary: pd.DataFrame) -> pd.DataFrame:
    ablation_experiments = [
        "full_core_fusion",
        "full_minus_ecg",
        "full_minus_flow",
        "full_minus_effort",
        "full_minus_spo2",
    ]

    ablation = results_summary[
        (results_summary["experiment"].isin(ablation_experiments))
        & (results_summary["model"] == "XGBoost")
    ].copy()

    full_rows = ablation[ablation["experiment"] == "full_core_fusion"]
    if full_rows.empty:
        ablation["delta_roc_auc_vs_full"] = np.nan
        ablation["delta_f1_vs_full"] = np.nan
        ablation["delta_sensitivity_vs_full"] = np.nan
        return ablation

    full_row = full_rows.iloc[0]
    ablation["delta_roc_auc_vs_full"] = (
        ablation["roc_auc_mean"] - full_row["roc_auc_mean"]
    )
    ablation["delta_f1_vs_full"] = ablation["f1_mean"] - full_row["f1_mean"]
    ablation["delta_sensitivity_vs_full"] = (
        ablation["sensitivity_mean"] - full_row["sensitivity_mean"]
    )

    order = {experiment: index for index, experiment in enumerate(ablation_experiments)}
    ablation["experiment_order"] = ablation["experiment"].map(order)
    ablation = ablation.sort_values("experiment_order").drop(
        columns=["experiment_order"]
    )

    return ablation


def print_data_overview(
    features: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    experiments: dict[str, list[str]],
) -> None:
    n_positive = int((y == 1).sum())
    n_normal = int((y == 0).sum())
    positive_rate = n_positive / len(y) if len(y) else 0.0

    print("Data overview")
    print(f"  Rows: {len(features)}")
    print(f"  Columns: {features.shape[1]}")
    print(f"  Records: {groups.nunique()}")
    print("Class balance")
    print(f"  normal: {n_normal}")
    print(f"  apnea_hypopnea: {n_positive}")
    print(f"  positive_rate: {positive_rate:.4f}")
    print("Experiments")

    for experiment, feature_columns in experiments.items():
        print(f"  {experiment}: {len(feature_columns)} features")

    late_features = unique_preserve_order(
        [
            *experiments["ecg_only"],
            *experiments["respiratory_fusion"],
            *experiments["spo2_only"],
        ]
    )
    print(f"  late_fusion: {len(late_features)} features")


def print_best_result(results_summary: pd.DataFrame) -> None:
    if results_summary.empty or "roc_auc_mean" not in results_summary.columns:
        print("Best result by mean roc_auc: not available.")
        return

    ranked = results_summary.dropna(subset=["roc_auc_mean"])
    if ranked.empty:
        print("Best result by mean roc_auc: not available.")
        return

    best = ranked.sort_values("roc_auc_mean", ascending=False).iloc[0]
    print("Best result by mean roc_auc")
    print(
        f"  {best['experiment']} / {best['model']}: "
        f"roc_auc_mean={best['roc_auc_mean']:.4f}, "
        f"f1_mean={best['f1_mean']:.4f}, "
        f"sensitivity_mean={best['sensitivity_mean']:.4f}"
    )


def main() -> None:
    if not FEATURES_PATH.exists():
        raise SystemExit(
            f"Feature table not found: {FEATURES_PATH}. "
            "Run: python scripts/pipeline/03_extract_features.py"
        )

    # Fail early with a clear message before the long CV loop starts.
    import_xgboost()

    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(FEATURES_PATH)
    y = features["label_binary"].astype(int)
    groups = features["record_id"].astype(str)

    feature_groups = get_feature_groups(features)
    experiments = build_experiments(feature_groups)
    print_data_overview(features, y, groups, experiments)

    splits = make_cv_splits(features, y, groups)

    result_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []

    xgboost_experiments = [
        "ecg_only",
        "flow_only",
        "effort_only",
        "spo2_only",
        "respiratory_fusion",
        "respiratory_spo2_fusion",
        "full_core_fusion",
        "full_minus_ecg",
        "full_minus_flow",
        "full_minus_effort",
        "full_minus_spo2",
    ]

    for experiment in xgboost_experiments:
        print(f"\nTraining {experiment} / XGBoost")
        rows, predictions = run_single_model_cv(
            features=features,
            y=y,
            groups=groups,
            splits=splits,
            experiment=experiment,
            feature_columns=experiments[experiment],
            model_name="XGBoost",
        )
        result_rows.extend(rows)
        prediction_rows.extend(predictions)

    for model_name in ["LogisticRegression", "RandomForest"]:
        print(f"\nTraining full_core_fusion / {model_name}")
        rows, predictions = run_single_model_cv(
            features=features,
            y=y,
            groups=groups,
            splits=splits,
            experiment="full_core_fusion",
            feature_columns=experiments["full_core_fusion"],
            model_name=model_name,
        )
        result_rows.extend(rows)
        prediction_rows.extend(predictions)

    print("\nTraining late_fusion / XGBoost")
    rows, predictions = run_late_fusion_cv(
        features=features,
        y=y,
        groups=groups,
        splits=splits,
        experiments=experiments,
    )
    result_rows.extend(rows)
    prediction_rows.extend(predictions)

    results_by_fold = pd.DataFrame(result_rows, columns=RESULT_BY_FOLD_COLUMNS)
    cv_predictions = pd.DataFrame(prediction_rows, columns=PREDICTION_COLUMNS)
    results_summary = summarize_results(results_by_fold)
    confusion_matrices = summarize_confusion_matrices(results_by_fold)
    ablation_summary = build_ablation_summary(results_summary)
    late_fusion_summary = results_summary[
        results_summary["experiment"] == "late_fusion"
    ].copy()

    results_by_fold.to_csv(RESULTS_BY_FOLD_PATH, index=False)
    results_summary.to_csv(RESULTS_SUMMARY_PATH, index=False)
    confusion_matrices.to_csv(CONFUSION_MATRICES_PATH, index=False)
    cv_predictions.to_csv(CV_PREDICTIONS_PATH, index=False)
    ablation_summary.to_csv(ABLATION_SUMMARY_PATH, index=False)
    late_fusion_summary.to_csv(LATE_FUSION_SUMMARY_PATH, index=False)

    print()
    print_best_result(results_summary)
    print(f"Results summary: {RESULTS_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
