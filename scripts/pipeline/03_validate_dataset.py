from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, REPORTS_TABLES_DIR  # noqa: E402


REPORTS_DIR = PROJECT_ROOT / "reports"

INPUT_FILES = {
    "ucddb_audit": REPORTS_TABLES_DIR / "ucddb_audit.csv",
    "epochs": DATA_PROCESSED_DIR / "epochs.csv",
    "features_all": DATA_PROCESSED_DIR / "features_all.csv",
    "epochs_summary_overall": REPORTS_TABLES_DIR / "epochs_summary_overall.csv",
    "epochs_summary_by_record": REPORTS_TABLES_DIR
    / "epochs_summary_by_record.csv",
    "feature_extraction_summary": REPORTS_TABLES_DIR
    / "feature_extraction_summary.csv",
}

VALIDATION_CHECKS_PATH = REPORTS_TABLES_DIR / "validation_checks.csv"
FEATURE_MISSINGNESS_PATH = REPORTS_TABLES_DIR / "validation_feature_missingness.csv"
FEATURE_RANGES_PATH = REPORTS_TABLES_DIR / "validation_feature_ranges.csv"
LABEL_DISTRIBUTION_PATH = (
    REPORTS_TABLES_DIR / "validation_label_distribution_by_record.csv"
)
VALIDATION_REPORT_PATH = REPORTS_DIR / "validation_report.md"

EXPECTED_RECORD_COUNT = 25
EXPECTED_LABELS = {"normal", "apnea_hypopnea"}
EXPECTED_BINARY_LABELS = {0, 1}
POSITIVE_RATE_MIN = 0.01
POSITIVE_RATE_MAX = 0.90

LEAKAGE_COLUMNS = [
    "record_id",
    "epoch_id",
    "start_sec",
    "end_sec",
    "sleep_stage",
    "label",
    "label_binary",
]

FEATURE_GROUP_PREFIXES = {
    "ecg": "ecg_",
    "flow": "flow_",
    "ribcage": "ribcage_",
    "abdo": "abdo_",
    "effort": "effort_",
    "spo2": "spo2_",
}


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


def read_table(
    name: str,
    path: Path,
    checks: list[dict[str, str]],
) -> pd.DataFrame | None:
    if not path.exists():
        return None

    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        add_check(
            checks,
            f"read_{name}",
            "FAIL",
            f"Could not read {path}: {type(exc).__name__}: {exc}",
        )
        return None


def check_input_files(checks: list[dict[str, str]]) -> None:
    for name, path in INPUT_FILES.items():
        if path.exists():
            add_check(checks, f"file_exists_{name}", "PASS", str(path))
        else:
            add_check(checks, f"file_exists_{name}", "FAIL", f"Missing file: {path}")


def get_feature_columns(features: pd.DataFrame | None) -> list[str]:
    if features is None:
        return []

    prefixes = tuple(FEATURE_GROUP_PREFIXES.values())
    return [column for column in features.columns if column.startswith(prefixes)]


def check_table_sizes(
    epochs: pd.DataFrame | None,
    features: pd.DataFrame | None,
    checks: list[dict[str, str]],
) -> None:
    if epochs is None:
        add_check(checks, "epochs_not_empty", "FAIL", "epochs.csv is unavailable.")
    elif epochs.empty:
        add_check(checks, "epochs_not_empty", "FAIL", "epochs.csv is empty.")
    else:
        add_check(checks, "epochs_not_empty", "PASS", f"Rows: {len(epochs)}")

    if features is None:
        add_check(
            checks,
            "features_not_empty",
            "FAIL",
            "features_all.csv is unavailable.",
        )
    elif features.empty:
        add_check(checks, "features_not_empty", "FAIL", "features_all.csv is empty.")
    else:
        add_check(checks, "features_not_empty", "PASS", f"Rows: {len(features)}")

    if epochs is not None and features is not None:
        if len(epochs) == len(features):
            add_check(
                checks,
                "epochs_features_row_count_match",
                "PASS",
                f"Both tables have {len(epochs)} rows.",
            )
        else:
            add_check(
                checks,
                "epochs_features_row_count_match",
                "FAIL",
                f"epochs rows={len(epochs)}, features rows={len(features)}.",
            )

        add_check(
            checks,
            "row_count_context",
            "PASS",
            (
                f"Observed {len(epochs)} epochs. For the current full UCDDB "
                "preparation this is expected to be around 20793, but this "
                "is not enforced as a hard validation rule."
            ),
        )


def check_record_ids(
    epochs: pd.DataFrame | None,
    features: pd.DataFrame | None,
    checks: list[dict[str, str]],
) -> None:
    if epochs is None or "record_id" not in epochs.columns:
        add_check(checks, "epochs_record_count", "FAIL", "record_id unavailable.")
        epochs_ids = set()
    else:
        epochs_ids = set(epochs["record_id"].dropna().astype(str))
        status = "PASS" if len(epochs_ids) == EXPECTED_RECORD_COUNT else "FAIL"
        add_check(
            checks,
            "epochs_record_count",
            status,
            f"Unique record_id count: {len(epochs_ids)}",
        )

    if features is None or "record_id" not in features.columns:
        add_check(checks, "features_record_count", "FAIL", "record_id unavailable.")
        feature_ids = set()
    else:
        feature_ids = set(features["record_id"].dropna().astype(str))
        status = "PASS" if len(feature_ids) == EXPECTED_RECORD_COUNT else "FAIL"
        add_check(
            checks,
            "features_record_count",
            status,
            f"Unique record_id count: {len(feature_ids)}",
        )

    if epochs is not None and features is not None:
        if epochs_ids == feature_ids:
            add_check(
                checks,
                "record_id_sets_match",
                "PASS",
                "record_id sets match between epochs and features.",
            )
        else:
            missing_in_features = sorted(epochs_ids - feature_ids)
            missing_in_epochs = sorted(feature_ids - epochs_ids)
            add_check(
                checks,
                "record_id_sets_match",
                "FAIL",
                (
                    f"Missing in features: {missing_in_features}; "
                    f"missing in epochs: {missing_in_epochs}"
                ),
            )


def check_unique_keys(
    table: pd.DataFrame | None,
    table_name: str,
    checks: list[dict[str, str]],
) -> None:
    check_name = f"{table_name}_record_epoch_unique"

    if table is None or not {"record_id", "epoch_id"}.issubset(table.columns):
        add_check(checks, check_name, "FAIL", "record_id/epoch_id unavailable.")
        return

    duplicate_count = int(table.duplicated(["record_id", "epoch_id"]).sum())
    status = "PASS" if duplicate_count == 0 else "FAIL"
    add_check(checks, check_name, status, f"Duplicate key rows: {duplicate_count}")


def check_labels(epochs: pd.DataFrame | None, checks: list[dict[str, str]]) -> None:
    if epochs is None:
        add_check(checks, "labels_available", "FAIL", "epochs.csv is unavailable.")
        return

    required_columns = {"label", "label_binary"}
    if not required_columns.issubset(epochs.columns):
        add_check(
            checks,
            "labels_available",
            "FAIL",
            f"Missing label columns: {sorted(required_columns - set(epochs.columns))}",
        )
        return

    labels = set(epochs["label"].dropna().astype(str))
    status = "PASS" if labels.issubset(EXPECTED_LABELS) else "FAIL"
    add_check(checks, "label_values", status, f"Observed labels: {sorted(labels)}")

    binary_values = set(pd.to_numeric(epochs["label_binary"], errors="coerce").dropna())
    binary_values_as_int = {int(value) for value in binary_values}
    status = "PASS" if binary_values_as_int.issubset(EXPECTED_BINARY_LABELS) else "FAIL"
    add_check(
        checks,
        "label_binary_values",
        status,
        f"Observed label_binary values: {sorted(binary_values_as_int)}",
    )

    normal_mismatch = int(
        (
            (epochs["label"] == "normal")
            & (pd.to_numeric(epochs["label_binary"], errors="coerce") != 0)
        ).sum()
    )
    apnea_mismatch = int(
        (
            (epochs["label"] == "apnea_hypopnea")
            & (pd.to_numeric(epochs["label_binary"], errors="coerce") != 1)
        ).sum()
    )
    mismatch_count = normal_mismatch + apnea_mismatch
    status = "PASS" if mismatch_count == 0 else "FAIL"
    add_check(
        checks,
        "label_binary_consistency",
        status,
        f"normal mismatches={normal_mismatch}, apnea_hypopnea mismatches={apnea_mismatch}",
    )

    positive_rate = float(pd.to_numeric(epochs["label_binary"], errors="coerce").mean())
    if POSITIVE_RATE_MIN <= positive_rate <= POSITIVE_RATE_MAX:
        status = "PASS"
    else:
        status = "FAIL"
    add_check(
        checks,
        "positive_rate_reasonable",
        status,
        f"Positive rate: {positive_rate:.4f}",
    )


def check_time_intervals(
    epochs: pd.DataFrame | None,
    checks: list[dict[str, str]],
) -> None:
    if epochs is None:
        add_check(checks, "time_columns_available", "FAIL", "epochs.csv unavailable.")
        return

    required_columns = {"record_id", "epoch_id", "start_sec", "end_sec"}
    if not required_columns.issubset(epochs.columns):
        add_check(
            checks,
            "time_columns_available",
            "FAIL",
            f"Missing columns: {sorted(required_columns - set(epochs.columns))}",
        )
        return

    start_sec = pd.to_numeric(epochs["start_sec"], errors="coerce")
    end_sec = pd.to_numeric(epochs["end_sec"], errors="coerce")
    duration = end_sec - start_sec

    duration_ok = bool(np.isclose(duration, 30).all())
    add_check(
        checks,
        "epoch_duration_30_seconds",
        "PASS" if duration_ok else "FAIL",
        f"Invalid duration rows: {int((~np.isclose(duration, 30)).sum())}",
    )

    start_ok = bool((start_sec >= 0).all())
    add_check(
        checks,
        "epoch_start_non_negative",
        "PASS" if start_ok else "FAIL",
        f"Rows with start_sec < 0: {int((start_sec < 0).sum())}",
    )

    end_after_start_ok = bool((end_sec > start_sec).all())
    add_check(
        checks,
        "epoch_end_after_start",
        "PASS" if end_after_start_ok else "FAIL",
        f"Rows with end_sec <= start_sec: {int((end_sec <= start_sec).sum())}",
    )

    bad_records = []
    for record_id, group in epochs.groupby("record_id", sort=True):
        epoch_ids = sorted(pd.to_numeric(group["epoch_id"], errors="coerce").dropna())
        expected_epoch_ids = list(range(len(epoch_ids)))
        if epoch_ids != expected_epoch_ids:
            bad_records.append(str(record_id))

    status = "PASS" if not bad_records else "FAIL"
    add_check(
        checks,
        "epoch_id_sequential_by_record",
        status,
        (
            "All records have sequential epoch_id from 0."
            if not bad_records
            else f"Records with gaps or non-sequential epoch_id: {bad_records}"
        ),
    )


def check_feature_groups(
    features: pd.DataFrame | None,
    checks: list[dict[str, str]],
) -> None:
    if features is None:
        add_check(checks, "feature_groups_available", "FAIL", "features unavailable.")
        return

    for group_name, prefix in FEATURE_GROUP_PREFIXES.items():
        matching_columns = [column for column in features.columns if column.startswith(prefix)]
        status = "PASS" if matching_columns else "FAIL"
        add_check(
            checks,
            f"feature_group_{group_name}",
            status,
            f"Columns found: {len(matching_columns)}",
        )


def build_missingness_table(
    features: pd.DataFrame | None,
    checks: list[dict[str, str]],
) -> pd.DataFrame:
    feature_columns = get_feature_columns(features)
    rows = []

    if features is None or not feature_columns:
        add_check(
            checks,
            "feature_missingness_available",
            "FAIL",
            "No feature columns available.",
        )
        return pd.DataFrame(columns=["feature", "n_missing", "n_rows", "nan_ratio", "status"])

    n_rows = len(features)
    for column in feature_columns:
        n_missing = int(features[column].isna().sum())
        nan_ratio = n_missing / n_rows if n_rows else 0.0

        if nan_ratio == 1.0:
            status = "FAIL"
        elif nan_ratio > 0.20:
            status = "WARNING"
        else:
            status = "PASS"

        rows.append(
            {
                "feature": column,
                "n_missing": n_missing,
                "n_rows": n_rows,
                "nan_ratio": nan_ratio,
                "status": status,
            }
        )

    missingness = pd.DataFrame(rows)
    n_fail = int((missingness["status"] == "FAIL").sum())
    n_warning = int((missingness["status"] == "WARNING").sum())

    if n_fail > 0:
        status = "FAIL"
    elif n_warning > 0:
        status = "WARNING"
    else:
        status = "PASS"

    add_check(
        checks,
        "feature_missingness",
        status,
        f"Features with FAIL={n_fail}, WARNING={n_warning}",
    )

    return missingness


def check_infinite_values(
    features: pd.DataFrame | None,
    checks: list[dict[str, str]],
) -> None:
    feature_columns = get_feature_columns(features)
    if features is None or not feature_columns:
        add_check(checks, "feature_infinite_values", "FAIL", "No features available.")
        return

    numeric_features = features[feature_columns].apply(pd.to_numeric, errors="coerce")
    infinite_count = int(np.isinf(numeric_features.to_numpy()).sum())
    status = "PASS" if infinite_count == 0 else "FAIL"
    add_check(
        checks,
        "feature_infinite_values",
        status,
        f"Infinite values in numeric features: {infinite_count}",
    )


def make_range_row(
    features: pd.DataFrame,
    feature: str,
    rule: str,
    valid_mask: pd.Series,
) -> dict[str, object]:
    values = pd.to_numeric(features[feature], errors="coerce")
    non_missing = values.dropna()
    violations = non_missing[~valid_mask.loc[non_missing.index]]

    if feature not in features.columns:
        return {
            "feature": feature,
            "rule": rule,
            "status": "FAIL",
            "n_checked": 0,
            "n_violations": 0,
            "min_value": np.nan,
            "max_value": np.nan,
            "details": "Feature column missing.",
        }

    if non_missing.empty:
        status = "WARNING"
        details = "No non-missing values to check."
    elif len(violations) > 0:
        status = "FAIL"
        details = "Out-of-range values found."
    else:
        status = "PASS"
        details = "All non-missing values are in range."

    return {
        "feature": feature,
        "rule": rule,
        "status": status,
        "n_checked": int(len(non_missing)),
        "n_violations": int(len(violations)),
        "min_value": float(non_missing.min()) if not non_missing.empty else np.nan,
        "max_value": float(non_missing.max()) if not non_missing.empty else np.nan,
        "details": details,
    }


def add_range_check(
    rows: list[dict[str, object]],
    features: pd.DataFrame,
    feature: str,
    rule: str,
    valid_mask_builder,
) -> None:
    if feature not in features.columns:
        rows.append(
            {
                "feature": feature,
                "rule": rule,
                "status": "FAIL",
                "n_checked": 0,
                "n_violations": 0,
                "min_value": np.nan,
                "max_value": np.nan,
                "details": "Feature column missing.",
            }
        )
        return

    values = pd.to_numeric(features[feature], errors="coerce")
    valid_mask = valid_mask_builder(values)
    rows.append(make_range_row(features, feature, rule, valid_mask))


def build_feature_ranges_table(
    features: pd.DataFrame | None,
    checks: list[dict[str, str]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    columns = [
        "feature",
        "rule",
        "status",
        "n_checked",
        "n_violations",
        "min_value",
        "max_value",
        "details",
    ]

    if features is None:
        add_check(checks, "feature_ranges", "FAIL", "features_all.csv unavailable.")
        return pd.DataFrame(columns=columns)

    for feature in ["spo2_mean", "spo2_min", "spo2_max"]:
        add_range_check(
            rows,
            features,
            feature,
            "50 <= value <= 100",
            lambda values: (values >= 50) & (values <= 100),
        )

    for feature in ["spo2_below_90_ratio", "spo2_below_92_ratio"]:
        add_range_check(
            rows,
            features,
            feature,
            "0 <= value <= 1",
            lambda values: (values >= 0) & (values <= 1),
        )

    add_range_check(
        rows,
        features,
        "flow_low_amplitude_ratio",
        "0 <= value <= 1",
        lambda values: (values >= 0) & (values <= 1),
    )

    for feature in [column for column in features.columns if "zero_crossing_rate" in column]:
        add_range_check(
            rows,
            features,
            feature,
            "0 <= value <= 1",
            lambda values: (values >= 0) & (values <= 1),
        )

    add_range_check(
        rows,
        features,
        "effort_corr",
        "-1 <= value <= 1",
        lambda values: (values >= -1) & (values <= 1),
    )
    add_range_check(
        rows,
        features,
        "ecg_hr_mean",
        "30 <= value <= 220",
        lambda values: (values >= 30) & (values <= 220),
    )
    add_range_check(
        rows,
        features,
        "ecg_rr_mean",
        "value > 0",
        lambda values: values > 0,
    )

    ranges = pd.DataFrame(rows, columns=columns)
    n_fail = int((ranges["status"] == "FAIL").sum())
    n_warning = int((ranges["status"] == "WARNING").sum())

    if n_fail > 0:
        status = "FAIL"
    elif n_warning > 0:
        status = "WARNING"
    else:
        status = "PASS"

    add_check(
        checks,
        "feature_ranges",
        status,
        f"Range checks with FAIL={n_fail}, WARNING={n_warning}",
    )

    return ranges


def build_label_distribution(
    epochs: pd.DataFrame | None,
    checks: list[dict[str, str]],
) -> pd.DataFrame:
    columns = [
        "record_id",
        "n_epochs",
        "n_normal",
        "n_apnea_hypopnea",
        "positive_rate",
        "has_positive_epochs",
        "has_normal_epochs",
    ]

    if epochs is None or not {"record_id", "label_binary"}.issubset(epochs.columns):
        add_check(
            checks,
            "label_distribution_by_record",
            "FAIL",
            "record_id/label_binary unavailable.",
        )
        return pd.DataFrame(columns=columns)

    rows = []
    for record_id, group in epochs.groupby("record_id", sort=True):
        label_binary = pd.to_numeric(group["label_binary"], errors="coerce")
        n_epochs = len(group)
        n_apnea_hypopnea = int((label_binary == 1).sum())
        n_normal = int((label_binary == 0).sum())
        rows.append(
            {
                "record_id": record_id,
                "n_epochs": n_epochs,
                "n_normal": n_normal,
                "n_apnea_hypopnea": n_apnea_hypopnea,
                "positive_rate": n_apnea_hypopnea / n_epochs if n_epochs else 0.0,
                "has_positive_epochs": n_apnea_hypopnea > 0,
                "has_normal_epochs": n_normal > 0,
            }
        )

    distribution = pd.DataFrame(rows, columns=columns)
    n_positive_records = int(distribution["has_positive_epochs"].sum())
    n_normal_records = int(distribution["has_normal_epochs"].sum())

    add_check(
        checks,
        "subject_split_positive_records",
        "PASS" if n_positive_records >= 5 else "FAIL",
        f"Records with positive epochs: {n_positive_records}",
    )
    add_check(
        checks,
        "subject_split_normal_records",
        "PASS" if n_normal_records >= 5 else "FAIL",
        f"Records with normal epochs: {n_normal_records}",
    )

    return distribution


def write_validation_report(
    checks: pd.DataFrame,
    epochs: pd.DataFrame | None,
    features: pd.DataFrame | None,
    report_path: Path,
) -> None:
    status_counts = checks["status"].value_counts().to_dict()
    n_epochs = len(epochs) if epochs is not None else 0
    n_features = len(features) if features is not None else 0
    n_records = (
        epochs["record_id"].nunique()
        if epochs is not None and "record_id" in epochs.columns
        else 0
    )

    failed_checks = checks[checks["status"] == "FAIL"]
    warning_checks = checks[checks["status"] == "WARNING"]

    lines = [
        "# Dataset Validation Report",
        "",
        "This report validates prepared UCDDB tables before model training.",
        "",
        "## Summary",
        "",
        f"- PASS: {status_counts.get('PASS', 0)}",
        f"- WARNING: {status_counts.get('WARNING', 0)}",
        f"- FAIL: {status_counts.get('FAIL', 0)}",
        f"- Epoch rows: {n_epochs}",
        f"- Feature rows: {n_features}",
        f"- Records: {n_records}",
        "",
        "## Leakage Guard",
        "",
        "The following columns must not be used as model features `X`:",
        "",
        ", ".join(f"`{column}`" for column in LEAKAGE_COLUMNS),
        "",
        "`sleep_stage` should be kept for analysis only and not used for training.",
        "",
        "## Recommended Split",
        "",
        (
            "Use `StratifiedGroupKFold` when available, or `GroupKFold` with "
            "`record_id` as the grouping variable. Do not randomly split "
            "individual epochs without grouping by record."
        ),
        "",
        "## Output Tables",
        "",
        f"- `{VALIDATION_CHECKS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{FEATURE_MISSINGNESS_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{FEATURE_RANGES_PATH.relative_to(PROJECT_ROOT)}`",
        f"- `{LABEL_DISTRIBUTION_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        "## Failed Checks",
        "",
    ]

    if failed_checks.empty:
        lines.append("No failed checks.")
    else:
        for _, row in failed_checks.iterrows():
            lines.append(f"- `{row['check_name']}`: {row['details']}")

    lines.extend(["", "## Warnings", ""])

    if warning_checks.empty:
        lines.append("No warnings.")
    else:
        for _, row in warning_checks.iterrows():
            lines.append(f"- `{row['check_name']}`: {row['details']}")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    checks: list[dict[str, str]] = []
    check_input_files(checks)

    tables = {
        name: read_table(name, path, checks)
        for name, path in INPUT_FILES.items()
    }

    epochs = tables["epochs"]
    features = tables["features_all"]

    check_table_sizes(epochs, features, checks)
    check_record_ids(epochs, features, checks)
    check_unique_keys(epochs, "epochs", checks)
    check_unique_keys(features, "features", checks)
    check_labels(epochs, checks)
    check_time_intervals(epochs, checks)
    check_feature_groups(features, checks)

    missingness = build_missingness_table(features, checks)
    check_infinite_values(features, checks)
    ranges = build_feature_ranges_table(features, checks)
    distribution = build_label_distribution(epochs, checks)

    add_check(
        checks,
        "leakage_columns_documented",
        "PASS",
        "Leakage columns are documented in validation_report.md.",
    )
    add_check(
        checks,
        "split_recommendation_documented",
        "PASS",
        "Subject-level split recommendation is documented in validation_report.md.",
    )

    checks_table = pd.DataFrame(checks, columns=["check_name", "status", "details"])

    checks_table.to_csv(VALIDATION_CHECKS_PATH, index=False)
    missingness.to_csv(FEATURE_MISSINGNESS_PATH, index=False)
    ranges.to_csv(FEATURE_RANGES_PATH, index=False)
    distribution.to_csv(LABEL_DISTRIBUTION_PATH, index=False)
    write_validation_report(checks_table, epochs, features, VALIDATION_REPORT_PATH)

    status_counts = checks_table["status"].value_counts().to_dict()
    n_pass = status_counts.get("PASS", 0)
    n_warning = status_counts.get("WARNING", 0)
    n_fail = status_counts.get("FAIL", 0)

    print("\nValidation summary")
    print(f"  PASS: {n_pass}")
    print(f"  WARNING: {n_warning}")
    print(f"  FAIL: {n_fail}")
    print(f"  Report: {VALIDATION_REPORT_PATH}")

    if n_fail > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
