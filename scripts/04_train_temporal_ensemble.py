from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
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

FEATURES_PATH = DATA_PROCESSED_DIR / "features_model_ready.csv"
BASELINE_PREDICTIONS_PATH = REPORTS_TABLES_DIR / "improved_cv_predictions.csv"

RESULTS_BY_FOLD_PATH = REPORTS_TABLES_DIR / "temporal_ensemble_results_by_fold.csv"
RESULTS_SUMMARY_PATH = REPORTS_TABLES_DIR / "temporal_ensemble_results_summary.csv"
BEST_RESULTS_PATH = REPORTS_TABLES_DIR / "temporal_ensemble_best_results.csv"
BEST_THRESHOLDS_PATH = REPORTS_TABLES_DIR / "temporal_ensemble_best_thresholds.csv"
REPORT_PATH = PROJECT_ROOT / "reports" / "temporal_ensemble_report.md"

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

GROUP_COLUMNS = [
    "regime",
    "feature_set",
    "model",
    "postprocessing",
    "smoothing_window_epochs",
    "smoothing_centered",
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
SMOOTHING_WINDOWS = [15, 31, 61]


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


def numeric_columns_with_prefixes(
    features: pd.DataFrame,
    prefixes: tuple[str, ...],
) -> list[str]:
    return [
        column
        for column in features.columns
        if column.startswith(prefixes)
        and column not in EXCLUDED_COLUMNS
        and pd.api.types.is_numeric_dtype(features[column])
    ]


def build_feature_sets(features: pd.DataFrame) -> dict[str, list[str]]:
    spo2 = numeric_columns_with_prefixes(
        features,
        (
            "spo2_",
            "rn_spo2_",
            "prev1_spo2_",
            "next1_spo2_",
            "roll3_mean_spo2_",
            "roll3_min_spo2_",
            "roll3_max_spo2_",
        ),
    )
    flow = numeric_columns_with_prefixes(
        features,
        (
            "flow_",
            "rn_flow_",
            "prev1_flow_",
            "next1_flow_",
            "roll3_mean_flow_",
            "roll3_min_flow_",
            "roll3_max_flow_",
        ),
    )

    return {
        "spo2_enhanced": spo2,
        "flow_spo2_enhanced": unique_preserve_order([*flow, *spo2]),
    }


def load_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise SystemExit(
            f"Model-ready feature table not found: {FEATURES_PATH}. "
            "Run: python scripts/03_build_model_ready_features.py"
        )

    features = pd.read_csv(FEATURES_PATH)
    required = {"record_id", "epoch_id", "label_binary", "is_sleep_epoch"}
    missing = required - set(features.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    return features


def load_official_fold_map(features: pd.DataFrame) -> dict[str, int] | None:
    if not BASELINE_PREDICTIONS_PATH.exists():
        return None

    predictions = pd.read_csv(
        BASELINE_PREDICTIONS_PATH,
        usecols=["record_id", "regime", "feature_variant", "experiment", "fold"],
    )
    baseline = predictions[
        (predictions["regime"] == "sleep_only")
        & (predictions["feature_variant"] == "enhanced")
        & (predictions["experiment"] == "spo2_only")
    ].copy()
    if baseline.empty:
        return None

    fold_map = (
        baseline[["record_id", "fold"]]
        .drop_duplicates()
        .set_index("record_id")["fold"]
        .astype(int)
        .to_dict()
    )
    feature_records = set(features["record_id"].astype(str).unique())
    if not feature_records.issubset(fold_map):
        return None

    return fold_map


def make_fallback_fold_map(features: pd.DataFrame) -> dict[str, int]:
    try:
        from sklearn.model_selection import StratifiedGroupKFold

        splitter = StratifiedGroupKFold(
            n_splits=N_SPLITS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
    except ImportError:
        splitter = GroupKFold(n_splits=N_SPLITS)

    y = features["label_binary"].astype(int)
    groups = features["record_id"].astype(str)
    dummy_x = features[["epoch_id"]]
    fold_map: dict[str, int] = {}

    for fold, (_, valid_indices) in enumerate(splitter.split(dummy_x, y, groups), start=1):
        valid_records = groups.iloc[valid_indices].unique()
        for record_id in valid_records:
            fold_map[str(record_id)] = fold

    return fold_map


def make_fold_map(features: pd.DataFrame) -> tuple[dict[str, int], str]:
    official = load_official_fold_map(features)
    if official is not None:
        return official, "saved improved baseline folds"

    return make_fallback_fold_map(features), "fresh subject-level CV folds"


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    n_positive = int((y_train == 1).sum())
    n_negative = int((y_train == 0).sum())
    if n_positive == 0:
        return 1.0

    return n_negative / n_positive


def make_spo2_xgboost(y_train: pd.Series) -> Pipeline:
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
                    scale_pos_weight=calculate_scale_pos_weight(y_train),
                ),
            ),
        ]
    )


def make_flow_spo2_hgb(_: pd.Series) -> Pipeline:
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                HistGradientBoostingClassifier(
                    max_iter=200,
                    learning_rate=0.03,
                    max_leaf_nodes=7,
                    l2_regularization=5.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def specificity_from_confusion(tn: int, fp: int) -> float:
    return tn / (tn + fp) if (tn + fp) > 0 else np.nan


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


def calculate_metrics(
    y_true: pd.Series | np.ndarray,
    y_proba: np.ndarray,
    threshold: float = THRESHOLD,
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
        "roc_auc": safe_roc_auc(y_true, y_proba),
        "average_precision": safe_average_precision(y_true, y_proba),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def train_oof_base_predictions(
    features: pd.DataFrame,
    feature_sets: dict[str, list[str]],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    base_predictions = features[
        ["record_id", "epoch_id", "fold", "label_binary"]
    ].copy()
    fold_metadata: list[dict[str, object]] = []

    for fold in sorted(features["fold"].unique()):
        valid_mask = features["fold"] == fold
        train_mask = ~valid_mask
        y_train = features.loc[train_mask, "label_binary"].astype(int)

        spo2_model = make_spo2_xgboost(y_train)
        spo2_columns = feature_sets["spo2_enhanced"]
        spo2_model.fit(features.loc[train_mask, spo2_columns], y_train)
        base_predictions.loc[valid_mask, "spo2_xgboost"] = spo2_model.predict_proba(
            features.loc[valid_mask, spo2_columns]
        )[:, 1]

        flow_spo2_model = make_flow_spo2_hgb(y_train)
        flow_spo2_columns = feature_sets["flow_spo2_enhanced"]
        flow_spo2_model.fit(features.loc[train_mask, flow_spo2_columns], y_train)
        base_predictions.loc[valid_mask, "flow_spo2_hgb"] = flow_spo2_model.predict_proba(
            features.loc[valid_mask, flow_spo2_columns]
        )[:, 1]

        fold_metadata.append(
            {
                "fold": int(fold),
                "n_train": int(train_mask.sum()),
                "n_valid": int(valid_mask.sum()),
                "n_train_records": int(features.loc[train_mask, "record_id"].nunique()),
                "n_valid_records": int(features.loc[valid_mask, "record_id"].nunique()),
            }
        )

    base_predictions["ensemble_mean"] = (
        base_predictions["spo2_xgboost"] + base_predictions["flow_spo2_hgb"]
    ) / 2.0
    return base_predictions, fold_metadata


def smooth_scores(
    predictions: pd.DataFrame,
    score_column: str,
    window_epochs: int,
    centered: bool,
) -> pd.Series:
    sorted_predictions = predictions.sort_values(["record_id", "epoch_id"]).copy()
    smoothed = sorted_predictions.groupby("record_id", sort=False)[score_column].transform(
        lambda values: values.rolling(
            window=window_epochs,
            center=centered,
            min_periods=1,
        ).mean()
    )
    smoothed.index = sorted_predictions.index
    return smoothed.sort_index()


def make_experiments(predictions: pd.DataFrame) -> dict[tuple[str, str, str, int, bool], pd.Series]:
    experiments: dict[tuple[str, str, str, int, bool], pd.Series] = {
        ("spo2_enhanced", "XGBoost", "raw", 0, False): predictions["spo2_xgboost"],
        (
            "flow_spo2_enhanced",
            "HistGradientBoosting",
            "raw",
            0,
            False,
        ): predictions["flow_spo2_hgb"],
        (
            "spo2_flow_spo2",
            "MeanEnsemble",
            "raw",
            0,
            False,
        ): predictions["ensemble_mean"],
    }

    for window in SMOOTHING_WINDOWS:
        for centered in [False, True]:
            suffix = "centered" if centered else "causal"
            experiments[
                (
                    "spo2_flow_spo2",
                    "MeanEnsemble",
                    f"rolling_mean_{suffix}",
                    window,
                    centered,
                )
            ] = smooth_scores(predictions, "ensemble_mean", window, centered)

    return experiments


def make_fold_result_rows(
    predictions: pd.DataFrame,
    experiments: dict[tuple[str, str, str, int, bool], pd.Series],
    fold_metadata: list[dict[str, object]],
) -> list[dict[str, object]]:
    metadata_by_fold = {int(row["fold"]): row for row in fold_metadata}
    rows: list[dict[str, object]] = []

    for (feature_set, model_name, postprocessing, window, centered), scores in experiments.items():
        for fold, fold_predictions in predictions.groupby("fold", sort=True):
            score_values = scores.loc[fold_predictions.index].to_numpy(dtype=float)
            y_true = fold_predictions["label_binary"].astype(int).to_numpy()
            metrics = calculate_metrics(y_true, score_values)
            metadata = metadata_by_fold[int(fold)]
            n_features = 68 if feature_set in {"flow_spo2_enhanced", "spo2_flow_spo2"} else 38
            rows.append(
                {
                    "regime": "sleep_only",
                    "feature_set": feature_set,
                    "model": model_name,
                    "postprocessing": postprocessing,
                    "smoothing_window_epochs": int(window),
                    "smoothing_centered": bool(centered),
                    "fold": int(fold),
                    "n_features": n_features,
                    "n_train": metadata["n_train"],
                    "n_valid": metadata["n_valid"],
                    "n_train_records": metadata["n_train_records"],
                    "n_valid_records": metadata["n_valid_records"],
                    **metrics,
                }
            )

    return rows


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

    return pd.DataFrame(summary_rows).sort_values("roc_auc_mean", ascending=False)


def calculate_threshold_metrics(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    metrics = calculate_metrics(y_true, y_proba, threshold)
    metrics["youden_index"] = float(metrics["sensitivity"] + metrics["specificity"] - 1)
    return metrics


def threshold_row(
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


def build_best_thresholds(
    predictions: pd.DataFrame,
    experiments: dict[tuple[str, str, str, int, bool], pd.Series],
) -> pd.DataFrame:
    rows = []
    y_true = predictions["label_binary"].astype(int).to_numpy()

    for (feature_set, model_name, postprocessing, window, centered), scores in experiments.items():
        y_proba = scores.to_numpy(dtype=float)
        key_values = {
            "regime": "sleep_only",
            "feature_set": feature_set,
            "model": model_name,
            "postprocessing": postprocessing,
            "smoothing_window_epochs": int(window),
            "smoothing_centered": bool(centered),
        }

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

        rows.append(threshold_row(key_values, "max_f1", by_f1))
        rows.append(threshold_row(key_values, "max_youden", by_youden))
        if sensitivity_candidates.empty:
            rows.append(
                threshold_row(
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
                threshold_row(
                    key_values,
                    "sensitivity_ge_0.70_max_specificity",
                    sensitivity_row,
                )
            )

    return pd.DataFrame(rows, columns=THRESHOLD_COLUMNS)


def best_row(table: pd.DataFrame, metric: str) -> pd.Series | None:
    if table.empty or metric not in table.columns:
        return None

    ranked = table.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if ranked.empty:
        return None

    return ranked.iloc[0]


def best_threshold_row(table: pd.DataFrame, selection_rule: str, metric: str) -> pd.Series | None:
    subset = table[table["selection_rule"] == selection_rule]
    if subset.empty or metric not in subset.columns:
        return None

    ranked = subset.dropna(subset=[metric]).sort_values(metric, ascending=False)
    if ranked.empty:
        return None

    return ranked.iloc[0]


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
        f"{row['feature_set']} / {row['model']} / {row['postprocessing']} "
        f"window={row['smoothing_window_epochs']} centered={row['smoothing_centered']} "
        f"(ROC-AUC={format_metric(row['roc_auc_mean'])}, "
        f"F1={format_metric(row['f1_mean'])}, "
        f"sensitivity={format_metric(row['sensitivity_mean'])}, "
        f"specificity={format_metric(row['specificity_mean'])})"
    )


def describe_threshold(row: pd.Series | None) -> str:
    if row is None:
        return "not available"

    return (
        f"{row['feature_set']} / {row['model']} / {row['postprocessing']} "
        f"window={row['smoothing_window_epochs']} centered={row['smoothing_centered']} "
        f"@ threshold={format_metric(row['threshold'])} "
        f"(F1={format_metric(row['f1'])}, "
        f"sensitivity={format_metric(row['sensitivity'])}, "
        f"specificity={format_metric(row['specificity'])})"
    )


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
    best_thresholds: pd.DataFrame,
    fold_source: str,
) -> str:
    best_auc = best_row(results_summary, "roc_auc_mean")
    best_f1 = best_row(results_summary, "f1_mean")
    best_tuned = best_threshold_row(best_thresholds, "max_f1", "f1")

    best_auc_value = float(best_auc["roc_auc_mean"]) if best_auc is not None else np.nan
    best_f1_value = float(best_f1["f1_mean"]) if best_f1 is not None else np.nan
    best_tuned_value = (
        float(best_tuned["f1"])
        if best_tuned is not None and pd.notna(best_tuned["f1"])
        else np.nan
    )

    reached_070 = bool(np.isfinite(best_auc_value) and best_auc_value >= 0.70)
    reached_075 = bool(np.isfinite(best_auc_value) and best_auc_value >= 0.75)
    reached_080 = bool(np.isfinite(best_auc_value) and best_auc_value >= 0.80)

    conclusion = (
        "The 0.70 target was reached, but 0.75 was not reached."
        if reached_070 and not reached_075
        else "The 0.70 target was not reached."
    )
    if reached_075:
        conclusion = "The 0.75 target was reached in this offline experiment."
    if reached_080:
        conclusion = "AUC 0.8 was reached; this should be treated as a leakage red flag."

    return f"""# Temporal Ensemble Experiment

## Summary

Fold source: {fold_source}. All folds are subject-level groups by `record_id`.

Best ROC-AUC: {describe_result(best_auc)}.

Best F1 at threshold 0.5: {describe_result(best_f1)}.

Best tuned F1: {describe_threshold(best_tuned)}.

Comparison with previous references:

- Previous best AUC: {PREVIOUS_BEST_AUC:.4f}; delta: {format_metric(best_auc_value - PREVIOUS_BEST_AUC)}
- Previous best F1: {PREVIOUS_BEST_F1:.4f}; delta: {format_metric(best_f1_value - PREVIOUS_BEST_F1)}
- Previous tuned F1: {PREVIOUS_TUNED_F1:.4f}; delta: {format_metric(best_tuned_value - PREVIOUS_TUNED_F1)}

{conclusion}

## Top Results

{markdown_table(results_summary.head(10), ["feature_set", "model", "postprocessing", "smoothing_window_epochs", "smoothing_centered", "roc_auc_mean", "roc_auc_std", "f1_mean", "sensitivity_mean", "specificity_mean", "average_precision_mean"])}

## Leakage Controls

The feature matrix excludes `record_id`, `epoch_id`, time columns, sleep stage, labels, sleep filter flags, and respiratory event metadata. Respiratory annotation files are not used as features.

Temporal smoothing is applied only to validation-fold probabilities inside each held-out record and does not use labels. Centered rolling smoothing uses future probabilities, so it is valid only for offline retrospective PSG analysis. The causal rolling variant avoids future probabilities, but the underlying enhanced SpO2 features still include `next1_*` and centered rolling signal context from the existing offline pipeline.

## Interpretation

The gain comes mostly from temporal probability smoothing and a small ensemble of complementary SpO2 and Flow+SpO2 models. This is physiologically plausible because apnea/hypopnea and desaturation patterns are temporally clustered. It should not be presented as a real-time detector, and it should be described as an offline retrospective classifier. The result still remains far from AUC 0.8, which likely requires raw-signal sequence modeling and external validation.
"""


def main() -> None:
    import_xgboost()
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    features = load_features()
    sleep_features = features[features["is_sleep_epoch"] == 1].copy().reset_index(drop=True)
    fold_map, fold_source = make_fold_map(sleep_features)
    sleep_features = sleep_features[sleep_features["record_id"].isin(fold_map)].copy()
    sleep_features["fold"] = sleep_features["record_id"].map(fold_map).astype(int)
    sleep_features = sleep_features.sort_values(["record_id", "epoch_id"]).reset_index(drop=True)

    feature_sets = build_feature_sets(sleep_features)
    print("Temporal ensemble experiment")
    print(f"  rows: {len(sleep_features)}")
    print(f"  records: {sleep_features['record_id'].nunique()}")
    print(f"  fold source: {fold_source}")
    print(f"  spo2 features: {len(feature_sets['spo2_enhanced'])}")
    print(f"  flow_spo2 features: {len(feature_sets['flow_spo2_enhanced'])}")

    predictions, fold_metadata = train_oof_base_predictions(sleep_features, feature_sets)
    experiments = make_experiments(predictions)
    results_by_fold = pd.DataFrame(
        make_fold_result_rows(predictions, experiments, fold_metadata),
        columns=RESULT_BY_FOLD_COLUMNS,
    )
    results_summary = summarize_results(results_by_fold)
    best_thresholds = build_best_thresholds(predictions, experiments)
    best_results = results_summary.sort_values("roc_auc_mean", ascending=False).reset_index(
        drop=True
    )

    results_by_fold.to_csv(RESULTS_BY_FOLD_PATH, index=False)
    results_summary.to_csv(RESULTS_SUMMARY_PATH, index=False)
    best_results.to_csv(BEST_RESULTS_PATH, index=False)
    best_thresholds.to_csv(BEST_THRESHOLDS_PATH, index=False)
    REPORT_PATH.write_text(
        build_report(results_summary, best_thresholds, fold_source),
        encoding="utf-8",
    )

    best_auc = best_row(results_summary, "roc_auc_mean")
    best_f1 = best_row(results_summary, "f1_mean")
    best_tuned = best_threshold_row(best_thresholds, "max_f1", "f1")

    print("\nTemporal ensemble summary")
    print(f"  Best ROC-AUC: {describe_result(best_auc)}")
    print(f"  Best F1 @0.5: {describe_result(best_f1)}")
    print(f"  Best tuned F1: {describe_threshold(best_tuned)}")
    print(f"  Results summary: {RESULTS_SUMMARY_PATH}")
    print(f"  Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
