from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, REPORTS_TABLES_DIR  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"

REQUIRED_FILES = [
    DATA_PROCESSED_DIR / "epochs.csv",
    DATA_PROCESSED_DIR / "features_all.csv",
    DATA_PROCESSED_DIR / "features_model_ready.csv",
    DATA_PROCESSED_DIR / "features_advanced.csv",
    REPORTS_TABLES_DIR / "improved_model_results_summary.csv",
    REPORTS_TABLES_DIR / "advanced_best_results.csv",
]

TRAINING_SCRIPTS = [
    PROJECT_ROOT / "scripts" / "04_train_models.py",
    PROJECT_ROOT / "scripts" / "04_train_improved_models.py",
    PROJECT_ROOT / "scripts" / "04_train_advanced_models.py",
]

FORBIDDEN_X_COLUMNS = {
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

LABEL_VALUES = {"normal", "apnea_hypopnea"}
LABEL_BINARY_VALUES = {0, 1}

CHECKS_PATH = REPORTS_TABLES_DIR / "pipeline_audit_checks.csv"
MISSINGNESS_PATH = REPORTS_TABLES_DIR / "pipeline_audit_feature_missingness.csv"
REPORT_PATH = REPORTS_DIR / "pipeline_audit_report.md"


def add_check(
    checks: list[dict[str, str]],
    check_name: str,
    status: str,
    details: str,
) -> None:
    checks.append(
        {
            "check_name": check_name,
            "status": status,
            "details": details,
        }
    )


def read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None

    return pd.read_csv(path)


def relative(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not import {path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def unique_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)

    return result


def get_training_feature_columns(
    script_path: Path,
    features_by_path: dict[Path, pd.DataFrame],
) -> list[str]:
    module_name = f"audit_{script_path.stem}"
    module = load_module(script_path, module_name)
    feature_path = Path(getattr(module, "FEATURES_PATH"))
    features = features_by_path.get(feature_path)

    if features is None:
        features = pd.read_csv(feature_path)
        features_by_path[feature_path] = features

    columns: list[str] = []

    if script_path.name == "04_train_models.py":
        groups = module.get_feature_groups(features)
        experiments = module.build_experiments(groups)
        for experiment_columns in experiments.values():
            columns.extend(experiment_columns)
        late_columns = module.unique_preserve_order(
            [
                *experiments.get("ecg_only", []),
                *experiments.get("respiratory_fusion", []),
                *experiments.get("spo2_only", []),
            ]
        )
        columns.extend(late_columns)

    elif script_path.name == "04_train_improved_models.py":
        for feature_variant in ["base", "enhanced"]:
            groups = module.build_feature_groups(features, feature_variant)
            experiments = module.build_experiments(groups)
            for experiment_columns in experiments.values():
                columns.extend(experiment_columns)

    elif script_path.name == "04_train_advanced_models.py":
        feature_sets = module.build_feature_sets(features)
        for experiment_columns in feature_sets.values():
            columns.extend(experiment_columns)

    return unique_preserve_order([str(column) for column in columns])


def check_required_files(checks: list[dict[str, str]]) -> None:
    missing = [relative(path) for path in REQUIRED_FILES if not path.exists()]

    if missing:
        add_check(
            checks,
            "required_files_exist",
            "FAIL",
            f"Missing files: {', '.join(missing)}",
        )
    else:
        add_check(
            checks,
            "required_files_exist",
            "PASS",
            f"All {len(REQUIRED_FILES)} required files exist.",
        )

    optional = REPORTS_TABLES_DIR / "improved_best_results.csv"
    status = "PASS" if optional.exists() else "WARNING"
    details = (
        f"Optional comparison file exists: {relative(optional)}"
        if optional.exists()
        else f"Optional comparison file is missing: {relative(optional)}"
    )
    add_check(checks, "optional_improved_best_results", status, details)


def check_table_integrity(
    checks: list[dict[str, str]],
    epochs: pd.DataFrame | None,
    features_all: pd.DataFrame | None,
    model_ready: pd.DataFrame | None,
    advanced: pd.DataFrame | None,
) -> None:
    if epochs is None or features_all is None:
        add_check(
            checks,
            "epochs_features_all_row_count",
            "FAIL",
            "Could not compare row counts because epochs.csv or features_all.csv is missing.",
        )
    elif len(epochs) == len(features_all):
        add_check(
            checks,
            "epochs_features_all_row_count",
            "PASS",
            f"Both tables have {len(epochs)} rows.",
        )
    else:
        add_check(
            checks,
            "epochs_features_all_row_count",
            "FAIL",
            f"epochs.csv rows={len(epochs)}, features_all.csv rows={len(features_all)}.",
        )

    for name, table in [
        ("features_model_ready_nonempty", model_ready),
        ("features_advanced_nonempty", advanced),
    ]:
        if table is None:
            add_check(checks, name, "FAIL", "Table is missing.")
        elif table.empty:
            add_check(checks, name, "FAIL", "Table exists but is empty.")
        else:
            add_check(checks, name, "PASS", f"Table has {len(table)} rows.")

    for table_name, table in [
        ("epochs", epochs),
        ("features_all", features_all),
        ("features_model_ready", model_ready),
        ("features_advanced", advanced),
    ]:
        if table is None:
            continue

        required_id_columns = {"record_id", "epoch_id"}
        missing_id_columns = required_id_columns - set(table.columns)
        if missing_id_columns:
            add_check(
                checks,
                f"{table_name}_record_epoch_unique",
                "FAIL",
                f"Missing ID columns: {sorted(missing_id_columns)}.",
            )
            continue

        duplicated = int(table.duplicated(["record_id", "epoch_id"]).sum())
        status = "PASS" if duplicated == 0 else "FAIL"
        add_check(
            checks,
            f"{table_name}_record_epoch_unique",
            status,
            f"Duplicated record_id+epoch_id rows: {duplicated}.",
        )

    for table_name, table in [
        ("epochs", epochs),
        ("features_model_ready", model_ready),
        ("features_advanced", advanced),
    ]:
        if table is None:
            continue

        missing_label_columns = {"label", "label_binary"} - set(table.columns)
        if missing_label_columns:
            add_check(
                checks,
                f"{table_name}_labels_valid",
                "FAIL",
                f"Missing label columns: {sorted(missing_label_columns)}.",
            )
            continue

        labels = set(table["label"].dropna().astype(str).unique())
        label_binary = set(table["label_binary"].dropna().astype(int).unique())
        invalid_labels = labels - LABEL_VALUES
        invalid_binary = label_binary - LABEL_BINARY_VALUES

        mapping = (
            table[["label", "label_binary"]]
            .drop_duplicates()
            .sort_values(["label", "label_binary"])
        )
        bad_normal = mapping[
            (mapping["label"] == "normal") & (mapping["label_binary"].astype(int) != 0)
        ]
        bad_positive = mapping[
            (mapping["label"] == "apnea_hypopnea")
            & (mapping["label_binary"].astype(int) != 1)
        ]

        if invalid_labels or invalid_binary or not bad_normal.empty or not bad_positive.empty:
            add_check(
                checks,
                f"{table_name}_labels_valid",
                "FAIL",
                "Invalid labels or label mapping. "
                f"labels={sorted(labels)}, label_binary={sorted(label_binary)}, "
                f"mapping={mapping.to_dict(orient='records')}",
            )
        else:
            add_check(
                checks,
                f"{table_name}_labels_valid",
                "PASS",
                "Labels are limited to normal/apnea_hypopnea and map to 0/1.",
            )


def check_training_scripts(
    checks: list[dict[str, str]],
    features_by_path: dict[Path, pd.DataFrame],
) -> dict[str, list[str]]:
    script_features: dict[str, list[str]] = {}

    for script_path in TRAINING_SCRIPTS:
        if not script_path.exists():
            add_check(
                checks,
                f"{script_path.name}_exists",
                "FAIL",
                f"Missing training script: {relative(script_path)}",
            )
            continue

        text = script_path.read_text(encoding="utf-8")
        if "train_test_split" in text:
            add_check(
                checks,
                f"{script_path.name}_no_epoch_random_split",
                "FAIL",
                "Found train_test_split in training script.",
            )
        else:
            add_check(
                checks,
                f"{script_path.name}_no_epoch_random_split",
                "PASS",
                "No train_test_split usage found.",
            )

        has_group_split = "StratifiedGroupKFold" in text or "GroupKFold" in text
        has_record_groups = bool(
            re.search(r"groups\s*=\s*features\s*\[\s*[\"']record_id[\"']\s*\]", text)
        )
        if has_group_split and has_record_groups:
            add_check(
                checks,
                f"{script_path.name}_subject_level_cv",
                "PASS",
                "Uses GroupKFold/StratifiedGroupKFold with groups=record_id.",
            )
        else:
            add_check(
                checks,
                f"{script_path.name}_subject_level_cv",
                "FAIL",
                "Could not confirm GroupKFold/StratifiedGroupKFold with groups=record_id.",
            )

        try:
            feature_columns = get_training_feature_columns(script_path, features_by_path)
        except Exception as exc:  # noqa: BLE001
            add_check(
                checks,
                f"{script_path.name}_x_columns_reconstructable",
                "FAIL",
                f"Could not reconstruct X columns: {type(exc).__name__}: {exc}",
            )
            continue

        script_features[script_path.name] = feature_columns
        forbidden_used = sorted(FORBIDDEN_X_COLUMNS & set(feature_columns))
        event_or_label_columns = sorted(
            column
            for column in feature_columns
            if "event" in column.lower() or "label" in column.lower()
        )

        if forbidden_used or event_or_label_columns:
            add_check(
                checks,
                f"{script_path.name}_no_forbidden_x_columns",
                "FAIL",
                "Forbidden or event/label-derived columns in X: "
                f"{sorted(set(forbidden_used + event_or_label_columns))}",
            )
        else:
            add_check(
                checks,
                f"{script_path.name}_no_forbidden_x_columns",
                "PASS",
                f"Reconstructed {len(feature_columns)} X columns; none are forbidden.",
            )

        feature_path = Path(getattr(load_module(script_path, f"audit_path_{script_path.stem}"), "FEATURES_PATH"))
        table = features_by_path.get(feature_path)
        if table is None:
            table = pd.read_csv(feature_path)
            features_by_path[feature_path] = table

        string_columns = [
            column
            for column in feature_columns
            if column in table.columns and not pd.api.types.is_numeric_dtype(table[column])
        ]
        if string_columns:
            add_check(
                checks,
                f"{script_path.name}_x_columns_numeric",
                "FAIL",
                f"String/non-numeric X columns: {string_columns}",
            )
        else:
            add_check(
                checks,
                f"{script_path.name}_x_columns_numeric",
                "PASS",
                "All reconstructed X columns are numeric.",
            )

    return script_features


def build_missingness_table(
    tables: dict[str, pd.DataFrame | None],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for table_name, table in tables.items():
        if table is None or table.empty:
            continue

        feature_candidates = [
            column
            for column in table.columns
            if column not in FORBIDDEN_X_COLUMNS
            and "event" not in column.lower()
            and "label" not in column.lower()
            and pd.api.types.is_numeric_dtype(table[column])
        ]

        for column in feature_candidates:
            nan_ratio = float(table[column].isna().mean())
            rows.append(
                {
                    "source_table": table_name,
                    "feature": column,
                    "nan_ratio": nan_ratio,
                    "missing_over_20_percent": bool(nan_ratio > 0.20),
                }
            )

    return pd.DataFrame(rows)


def check_missingness(
    checks: list[dict[str, str]],
    missingness: pd.DataFrame,
) -> None:
    if missingness.empty:
        add_check(
            checks,
            "feature_missingness_available",
            "WARNING",
            "No numeric feature columns were available for missingness audit.",
        )
        return

    max_nan_ratio = float(missingness["nan_ratio"].max())
    n_over_20 = int(missingness["missing_over_20_percent"].sum())
    status = "PASS" if n_over_20 == 0 else "WARNING"
    add_check(
        checks,
        "feature_missingness",
        status,
        f"max_nan_ratio={max_nan_ratio:.4f}; features_with_nan_over_20_percent={n_over_20}.",
    )


def load_best_results() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    improved_best_path = REPORTS_TABLES_DIR / "improved_best_results.csv"
    improved_summary_path = REPORTS_TABLES_DIR / "improved_model_results_summary.csv"
    advanced_best_path = REPORTS_TABLES_DIR / "advanced_best_results.csv"

    improved = read_csv_if_exists(improved_best_path)
    if improved is None:
        improved = read_csv_if_exists(improved_summary_path)

    advanced = read_csv_if_exists(advanced_best_path)
    return improved, advanced


def best_by_auc(results: pd.DataFrame | None) -> pd.Series | None:
    if results is None or results.empty or "roc_auc_mean" not in results.columns:
        return None

    ranked = results.dropna(subset=["roc_auc_mean"]).sort_values(
        "roc_auc_mean",
        ascending=False,
    )
    if ranked.empty:
        return None

    return ranked.iloc[0]


def row_descriptor(row: pd.Series | None) -> str:
    if row is None:
        return "not available"

    pieces = []
    for column in ["regime", "feature_variant", "experiment", "model", "config_name"]:
        if column in row.index and pd.notna(row[column]):
            pieces.append(f"{column}={row[column]}")

    metric = (
        f"roc_auc_mean={float(row['roc_auc_mean']):.4f}"
        if "roc_auc_mean" in row.index and pd.notna(row["roc_auc_mean"])
        else "roc_auc_mean=NA"
    )
    f1 = (
        f"f1_mean={float(row['f1_mean']):.4f}"
        if "f1_mean" in row.index and pd.notna(row["f1_mean"])
        else "f1_mean=NA"
    )
    return f"{' / '.join(pieces)} ({metric}, {f1})"


def check_results(
    checks: list[dict[str, str]],
) -> dict[str, object]:
    improved, advanced = load_best_results()
    improved_best = best_by_auc(improved)
    advanced_best = best_by_auc(advanced)

    if improved_best is None or advanced_best is None:
        add_check(
            checks,
            "improved_vs_advanced_results",
            "WARNING",
            "Could not compare improved and advanced best results.",
        )
        return {
            "improved": improved,
            "advanced": advanced,
            "improved_best": improved_best,
            "advanced_best": advanced_best,
            "advanced_delta_auc": np.nan,
        }

    delta = float(advanced_best["roc_auc_mean"] - improved_best["roc_auc_mean"])
    status = "PASS" if delta <= 0 else "WARNING"
    add_check(
        checks,
        "improved_vs_advanced_results",
        status,
        "Advanced features did not improve the best ROC-AUC."
        if delta <= 0
        else "Advanced features improved ROC-AUC; inspect possible overfitting/leakage.",
    )

    return {
        "improved": improved,
        "advanced": advanced,
        "improved_best": improved_best,
        "advanced_best": advanced_best,
        "advanced_delta_auc": delta,
    }


def combine_results_for_reporting(
    improved: pd.DataFrame | None,
    advanced: pd.DataFrame | None,
) -> pd.DataFrame:
    frames = []

    if improved is not None and not improved.empty:
        result = improved.copy()
        result["result_source"] = "improved"
        frames.append(result)

    if advanced is not None and not advanced.empty:
        result = advanced.copy()
        result["result_source"] = "advanced"
        frames.append(result)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True, sort=False)


def best_filtered(results: pd.DataFrame, mask: pd.Series) -> pd.Series | None:
    subset = results[mask].copy()
    if subset.empty or "roc_auc_mean" not in subset.columns:
        return None

    subset = subset.dropna(subset=["roc_auc_mean"]).sort_values(
        "roc_auc_mean",
        ascending=False,
    )
    if subset.empty:
        return None

    return subset.iloc[0]


def format_metric(value: object) -> str:
    if pd.isna(value):
        return "NA"

    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def make_markdown_table(rows: list[dict[str, object]], columns: list[str]) -> str:
    if not rows:
        return "_No rows available._"

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")

    return "\n".join([header, separator, *body])


def build_report(
    checks: list[dict[str, str]],
    missingness: pd.DataFrame,
    result_context: dict[str, object],
    script_features: dict[str, list[str]],
) -> str:
    n_fail = sum(check["status"] == "FAIL" for check in checks)
    n_warning = sum(check["status"] == "WARNING" for check in checks)
    n_pass = sum(check["status"] == "PASS" for check in checks)

    failed_checks = [check for check in checks if check["status"] == "FAIL"]
    warning_checks = [check for check in checks if check["status"] == "WARNING"]

    improved_best = result_context.get("improved_best")
    advanced_best = result_context.get("advanced_best")
    delta_auc = result_context.get("advanced_delta_auc", np.nan)
    combined_results = combine_results_for_reporting(
        result_context.get("improved"),  # type: ignore[arg-type]
        result_context.get("advanced"),  # type: ignore[arg-type]
    )

    best_all = None
    best_sleep = None
    best_multimodal = None
    if not combined_results.empty:
        if "regime" in combined_results.columns:
            best_all = best_filtered(combined_results, combined_results["regime"] == "all_epochs")
            best_sleep = best_filtered(combined_results, combined_results["regime"] == "sleep_only")

        experiment_text = combined_results.get("experiment", pd.Series("", index=combined_results.index)).astype(str)
        multimodal_mask = experiment_text.str.contains(
            "fusion|respiratory_spo2|full|late",
            case=False,
            regex=True,
            na=False,
        )
        best_multimodal = best_filtered(combined_results, multimodal_mask)

    missing_over_20 = (
        missingness[missingness["missing_over_20_percent"]]
        .sort_values("nan_ratio", ascending=False)
        .head(20)
        if not missingness.empty
        else pd.DataFrame()
    )

    missing_rows = [
        {
            "table": row["source_table"],
            "feature": row["feature"],
            "nan_ratio": format_metric(row["nan_ratio"]),
        }
        for _, row in missing_over_20.iterrows()
    ]

    risk_rows = [
        {"check": check["check_name"], "status": check["status"], "details": check["details"]}
        for check in [*failed_checks, *warning_checks]
    ]

    script_rows = [
        {
            "script": script_name,
            "n_reconstructed_x_columns": len(columns),
        }
        for script_name, columns in script_features.items()
    ]

    can_continue = n_fail == 0
    if can_continue:
        correctness_summary = (
            "The current pipeline passes the hard audit checks. The existing model "
            "scripts use subject-level validation by record_id, and reconstructed X "
            "columns do not include labels, event metadata, epoch IDs, sleep stage, "
            "or record IDs."
        )
    else:
        correctness_summary = (
            "The current pipeline has hard audit failures. New modeling experiments "
            "should not be run until these failures are fixed."
        )

    advanced_sentence = "Advanced result comparison is unavailable."
    if improved_best is not None and advanced_best is not None:
        advanced_sentence = (
            "Advanced features did not improve the best improved baseline "
            f"(delta ROC-AUC={float(delta_auc):+.4f})."
            if float(delta_auc) <= 0
            else "Advanced features improved the best improved baseline; inspect this carefully."
        )

    report = f"""# Pipeline Audit Report

## Summary

{correctness_summary}

Checks: PASS={n_pass}, WARNING={n_warning}, FAIL={n_fail}.

## Hard Failures

{make_markdown_table(risk_rows if failed_checks else [], ["check", "status", "details"])}

## Warnings And Risks

{make_markdown_table(risk_rows if warning_checks else [], ["check", "status", "details"])}

## Leakage And Split Review

Forbidden X columns audited: {", ".join(sorted(FORBIDDEN_X_COLUMNS))}.

{make_markdown_table(script_rows, ["script", "n_reconstructed_x_columns"])}

No reconstructed X feature set used respiratory event metadata or labels. Existing contextual lead/next features use future signal context; this is acceptable only for offline retrospective PSG analysis and should not be described as real-time detection.

## Feature Missingness

The detailed missingness table is saved to `{relative(MISSINGNESS_PATH)}`.

Top features with NaN ratio over 20 percent:

{make_markdown_table(missing_rows, ["table", "feature", "nan_ratio"])}

## Result Review

Best improved baseline: {row_descriptor(improved_best)}.

Best advanced result: {row_descriptor(advanced_best)}.

{advanced_sentence}

Best all_epochs result: {row_descriptor(best_all)}.

Best sleep_only result: {row_descriptor(best_sleep)}.

Best multimodal result: {row_descriptor(best_multimodal)}.

## Why Current ROC-AUC May Be Limited

The UCDDB subject count is small, and subject-level validation is intentionally harder than random epoch-level splits. Thirty-second aggregate tabular features lose waveform morphology and temporal event structure. Labels are event-overlap labels, so borderline windows around event onset and recovery are intrinsically noisy. SpO2 desaturation can lag airflow reduction, which makes single-window classification difficult without temporal modeling.

## Recommendation

Continue only if FAIL=0. The next honest experiment should move closer to event-detection logic: use raw Flow and SpO2-derived segment features, evaluate 60-second and 10-second windows, keep record_id grouped CV, exclude all event metadata from X, and clearly mark any future signal context as offline retrospective analysis.
"""

    return report


def print_summary(checks: list[dict[str, str]], result_context: dict[str, object]) -> None:
    n_fail = sum(check["status"] == "FAIL" for check in checks)
    n_warning = sum(check["status"] == "WARNING" for check in checks)
    n_pass = sum(check["status"] == "PASS" for check in checks)

    print("Pipeline audit summary")
    print(f"  PASS: {n_pass}")
    print(f"  WARNING: {n_warning}")
    print(f"  FAIL: {n_fail}")

    improved_best = result_context.get("improved_best")
    advanced_best = result_context.get("advanced_best")
    print(f"  Best improved: {row_descriptor(improved_best)}")
    print(f"  Best advanced: {row_descriptor(advanced_best)}")
    print(f"  Audit checks: {CHECKS_PATH}")
    print(f"  Audit report: {REPORT_PATH}")

    if n_fail:
        print("  FAIL checks:")
        for check in checks:
            if check["status"] == "FAIL":
                print(f"    - {check['check_name']}: {check['details']}")


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, str]] = []
    check_required_files(checks)

    epochs = read_csv_if_exists(DATA_PROCESSED_DIR / "epochs.csv")
    features_all = read_csv_if_exists(DATA_PROCESSED_DIR / "features_all.csv")
    model_ready = read_csv_if_exists(DATA_PROCESSED_DIR / "features_model_ready.csv")
    advanced = read_csv_if_exists(DATA_PROCESSED_DIR / "features_advanced.csv")

    check_table_integrity(checks, epochs, features_all, model_ready, advanced)

    features_by_path: dict[Path, pd.DataFrame] = {}
    for path, table in [
        (DATA_PROCESSED_DIR / "features_all.csv", features_all),
        (DATA_PROCESSED_DIR / "features_model_ready.csv", model_ready),
        (DATA_PROCESSED_DIR / "features_advanced.csv", advanced),
    ]:
        if table is not None:
            features_by_path[path] = table

    script_features = check_training_scripts(checks, features_by_path)
    missingness = build_missingness_table(
        {
            "features_model_ready": model_ready,
            "features_advanced": advanced,
        }
    )
    check_missingness(checks, missingness)
    result_context = check_results(checks)

    checks_table = pd.DataFrame(checks, columns=["check_name", "status", "details"])
    checks_table.to_csv(CHECKS_PATH, index=False)
    missingness.to_csv(MISSINGNESS_PATH, index=False)

    report = build_report(checks, missingness, result_context, script_features)
    REPORT_PATH.write_text(report, encoding="utf-8")

    print_summary(checks, result_context)

    if (checks_table["status"] == "FAIL").any():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
