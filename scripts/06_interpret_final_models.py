from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import average_precision_score, make_scorer
from sklearn.pipeline import Pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, REPORTS_FIGURES_DIR, REPORTS_TABLES_DIR  # noqa: E402


RANDOM_STATE = 42
MAX_PERMUTATION_ROWS = 3000

FEATURES_PATH = DATA_PROCESSED_DIR / "features_model_ready.csv"

SPO2_IMPORTANCE_PATH = REPORTS_TABLES_DIR / "final_feature_importance_spo2_xgboost.csv"
FLOW_SPO2_IMPORTANCE_PATH = (
    REPORTS_TABLES_DIR / "final_feature_importance_flow_spo2_hgb.csv"
)
SPO2_FIGURE_PATH = REPORTS_FIGURES_DIR / "final_feature_importance_spo2_xgboost.png"
FLOW_SPO2_FIGURE_PATH = (
    REPORTS_FIGURES_DIR / "final_feature_importance_flow_spo2_hgb.png"
)
SHAP_FIGURE_PATH = REPORTS_FIGURES_DIR / "final_shap_summary_spo2_xgboost.png"
REPORT_PATH = PROJECT_ROOT / "reports" / "interpretability_report.md"

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


def load_sleep_features() -> pd.DataFrame:
    if not FEATURES_PATH.exists():
        raise SystemExit(
            f"Feature table not found: {FEATURES_PATH}. "
            "Run: python scripts/03_build_model_ready_features.py"
        )

    features = pd.read_csv(FEATURES_PATH)
    required = {"label_binary", "is_sleep_epoch"}
    missing = required - set(features.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {sorted(missing)}")

    return features[features["is_sleep_epoch"] == 1].copy()


def calculate_scale_pos_weight(y: pd.Series) -> float:
    n_positive = int((y == 1).sum())
    n_negative = int((y == 0).sum())

    if n_positive == 0:
        return 1.0

    return n_negative / n_positive


def make_spo2_xgboost(y: pd.Series) -> Pipeline:
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
                    scale_pos_weight=calculate_scale_pos_weight(y),
                ),
            ),
        ]
    )


def make_flow_spo2_hgb() -> Pipeline:
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


def xgboost_gain_importance(
    model: Pipeline,
    feature_columns: list[str],
) -> pd.DataFrame:
    booster = model.named_steps["model"].get_booster()
    score = booster.get_score(importance_type="gain")
    rows = []

    for index, feature in enumerate(feature_columns):
        rows.append(
            {
                "feature": feature,
                "importance_gain": float(score.get(f"f{index}", 0.0)),
            }
        )

    importance = pd.DataFrame(rows)
    total = float(importance["importance_gain"].sum())
    importance["importance_gain_normalized"] = (
        importance["importance_gain"] / total if total > 0 else 0.0
    )
    return importance.sort_values("importance_gain", ascending=False).reset_index(drop=True)


def sample_for_permutation(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    if len(X) <= MAX_PERMUTATION_ROWS:
        return X, y

    sample = (
        pd.DataFrame({"label_binary": y})
        .groupby("label_binary", group_keys=False)
        .sample(
            n=min(
                MAX_PERMUTATION_ROWS // max(1, y.nunique()),
                int(y.value_counts().min()),
            ),
            random_state=RANDOM_STATE,
        )
    )
    if len(sample) < min(MAX_PERMUTATION_ROWS, len(X)):
        remaining = X.index.difference(sample.index)
        n_remaining = min(MAX_PERMUTATION_ROWS - len(sample), len(remaining))
        if n_remaining > 0:
            extra = pd.Series(remaining).sample(n=n_remaining, random_state=RANDOM_STATE)
            sample = pd.concat([sample, pd.DataFrame(index=extra.to_numpy())], axis=0)

    sample_indices = sample.index[:MAX_PERMUTATION_ROWS]
    return X.loc[sample_indices], y.loc[sample_indices]


def hgb_permutation_importance(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
) -> pd.DataFrame:
    X_sample, y_sample = sample_for_permutation(X, y)
    scorer = make_scorer(average_precision_score, response_method="predict_proba")
    result = permutation_importance(
        model,
        X_sample,
        y_sample,
        scoring=scorer,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "permutation_importance_mean": result.importances_mean,
            "permutation_importance_std": result.importances_std,
        }
    )
    return importance.sort_values(
        "permutation_importance_mean",
        ascending=False,
    ).reset_index(drop=True)


def plot_importance(
    importance: pd.DataFrame,
    value_column: str,
    title: str,
    output_path: Path,
    top_n: int = 20,
) -> None:
    top = importance.head(top_n).iloc[::-1].copy()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top[value_column])
    ax.set_xlabel(value_column)
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def try_shap_summary(
    model: Pipeline,
    X: pd.DataFrame,
) -> str:
    try:
        import shap  # type: ignore[import-not-found]
    except Exception as exc:  # noqa: BLE001
        return f"SHAP skipped: shap is not installed ({type(exc).__name__})."

    try:
        sample = X.sample(n=min(1000, len(X)), random_state=RANDOM_STATE)
        transformed = model.named_steps["imputer"].transform(sample)
        explainer = shap.TreeExplainer(model.named_steps["model"])
        shap_values = explainer.shap_values(transformed)
        plt.figure()
        shap.summary_plot(
            shap_values,
            transformed,
            feature_names=sample.columns,
            plot_type="bar",
            show=False,
            max_display=20,
        )
        plt.tight_layout()
        plt.savefig(SHAP_FIGURE_PATH, dpi=200, bbox_inches="tight")
        plt.close()
        return f"SHAP summary bar plot saved to `{relative(SHAP_FIGURE_PATH)}`."
    except Exception as exc:  # noqa: BLE001
        plt.close("all")
        return f"SHAP skipped because it failed: {type(exc).__name__}: {exc}"


def relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def markdown_top_features(
    importance: pd.DataFrame,
    value_column: str,
    top_n: int = 10,
) -> str:
    rows = []
    for rank, (_, row) in enumerate(importance.head(top_n).iterrows(), start=1):
        rows.append(
            f"| {rank} | {row['feature']} | {float(row[value_column]):.6f} |"
        )

    return "\n".join(
        [
            "| rank | feature | importance |",
            "| --- | --- | --- |",
            *rows,
        ]
    )


def build_report(
    spo2_importance: pd.DataFrame,
    hgb_importance: pd.DataFrame,
    shap_status: str,
    n_sleep_rows: int,
    n_spo2_features: int,
    n_flow_spo2_features: int,
) -> str:
    return f"""# Interpretability Report

## Scope

Final component models were trained once on the sleep-only subset of `features_model_ready.csv` only for interpretability. No new validation results were produced. Rows used: {n_sleep_rows}; SpO2 features: {n_spo2_features}; Flow+SpO2 features: {n_flow_spo2_features}.

## Top SpO2 XGBoost Features

{markdown_top_features(spo2_importance, "importance_gain")}

## Top Flow+SpO2 HistGradientBoosting Features

{markdown_top_features(hgb_importance, "permutation_importance_mean")}

## Physiological Interpretation

SpO2 minimum, desaturation/drop features, and ratios below 90/92 percent are directly related to oxygen desaturation during apnea and hypopnea episodes. Flow low-amplitude features reflect reduced or absent airflow, which is a core respiratory manifestation of apnea/hypopnea. Temporal and contextual features, including previous, next, and rolling-window summaries, reflect the delayed and clustered nature of desaturation relative to respiratory flow changes.

## Limitations

Feature importance does not prove causality. Correlated features can share or mask importance, and permutation importance depends on the sampled data. The models were trained on the small UCDDB dataset and are not a clinical validation. Some contextual features use future neighboring signal information and therefore belong to offline retrospective PSG analysis rather than real-time detection.

## SHAP

{shap_status}

## Files

- `{relative(SPO2_IMPORTANCE_PATH)}`
- `{relative(FLOW_SPO2_IMPORTANCE_PATH)}`
- `{relative(SPO2_FIGURE_PATH)}`
- `{relative(FLOW_SPO2_FIGURE_PATH)}`
"""


def main() -> None:
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    sleep_features = load_sleep_features()
    feature_sets = build_feature_sets(sleep_features)
    y = sleep_features["label_binary"].astype(int)

    spo2_columns = feature_sets["spo2_enhanced"]
    flow_spo2_columns = feature_sets["flow_spo2_enhanced"]

    if not spo2_columns or not flow_spo2_columns:
        raise SystemExit("Could not build required feature sets.")

    print("Training final interpretability models")
    print(f"  sleep_only rows: {len(sleep_features)}")
    print(f"  spo2_enhanced features: {len(spo2_columns)}")
    print(f"  flow_spo2_enhanced features: {len(flow_spo2_columns)}")

    spo2_model = make_spo2_xgboost(y)
    spo2_model.fit(sleep_features[spo2_columns], y)
    spo2_importance = xgboost_gain_importance(spo2_model, spo2_columns)

    hgb_model = make_flow_spo2_hgb()
    X_flow_spo2 = sleep_features[flow_spo2_columns]
    hgb_model.fit(X_flow_spo2, y)
    hgb_importance = hgb_permutation_importance(hgb_model, X_flow_spo2, y)

    spo2_importance.to_csv(SPO2_IMPORTANCE_PATH, index=False)
    hgb_importance.to_csv(FLOW_SPO2_IMPORTANCE_PATH, index=False)

    plot_importance(
        spo2_importance,
        "importance_gain",
        "SpO2 XGBoost feature importance (gain)",
        SPO2_FIGURE_PATH,
    )
    plot_importance(
        hgb_importance,
        "permutation_importance_mean",
        "Flow+SpO2 HistGradientBoosting permutation importance",
        FLOW_SPO2_FIGURE_PATH,
    )

    shap_status = try_shap_summary(spo2_model, sleep_features[spo2_columns])
    REPORT_PATH.write_text(
        build_report(
            spo2_importance=spo2_importance,
            hgb_importance=hgb_importance,
            shap_status=shap_status,
            n_sleep_rows=len(sleep_features),
            n_spo2_features=len(spo2_columns),
            n_flow_spo2_features=len(flow_spo2_columns),
        ),
        encoding="utf-8",
    )

    print("Interpretability artifacts created")
    print(f"  SpO2 importance: {SPO2_IMPORTANCE_PATH}")
    print(f"  Flow+SpO2 importance: {FLOW_SPO2_IMPORTANCE_PATH}")
    print(f"  Report: {REPORT_PATH}")
    print(f"  {shap_status}")


if __name__ == "__main__":
    main()
