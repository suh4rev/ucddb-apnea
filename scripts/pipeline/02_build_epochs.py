from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_PROCESSED_DIR, DATA_RAW_DIR, REPORTS_TABLES_DIR  # noqa: E402


EPOCH_SECONDS = 30
POSITIVE_LABEL = "apnea_hypopnea"
NEGATIVE_LABEL = "normal"

TIME_PATTERN = re.compile(r"^\d{2}:\d{2}:\d{2}$")
RECORD_ID_PATTERN = re.compile(r"ucddb\d+", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"^[+-]?(?:\d+(?:\.\d*)?|\.\d+)$")
RESPIRATORY_EVENT_PREFIXES = ("APNEA", "HYP")

EPOCH_COLUMNS = [
    "record_id",
    "epoch_id",
    "start_sec",
    "end_sec",
    "sleep_stage",
    "label",
    "label_binary",
    "n_events",
    "event_types",
    "event_start_sec",
    "event_duration_sec",
]

SUMMARY_BY_RECORD_COLUMNS = [
    "record_id",
    "duration_seconds",
    "n_epochs",
    "n_normal",
    "n_apnea_hypopnea",
    "positive_rate",
    "n_parsed_events",
    "n_parse_errors",
    "has_stage_file",
    "n_stage_rows",
]

SUMMARY_OVERALL_COLUMNS = [
    "n_records",
    "n_epochs",
    "n_normal",
    "n_apnea_hypopnea",
    "positive_rate",
    "n_parsed_events",
    "n_parse_errors",
]

PARSE_ERROR_COLUMNS = [
    "record_id",
    "file",
    "line_number",
    "line",
    "error",
]


def import_pyedflib():
    try:
        import pyedflib
    except ImportError as exc:
        raise SystemExit(
            "pyedflib is not installed. Install dependencies with: "
            "pip install -r requirements.txt"
        ) from exc

    return pyedflib


def get_record_id(path: Path) -> str | None:
    match = RECORD_ID_PATTERN.search(path.name)
    if match:
        return match.group(0).lower()

    return None


def find_signal_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []

    return sorted(
        path
        for path in raw_dir.glob("ucddb*.rec")
        if path.is_file() and "lifecard" not in path.name.lower()
    )


def time_to_seconds(value: str) -> int:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + int(seconds)


def is_number(value: str) -> bool:
    return bool(NUMBER_PATTERN.match(value))


def parse_duration(tokens: list[str]) -> float | None:
    for token in tokens:
        cleaned_token = token.strip().rstrip(",;")
        if is_number(cleaned_token):
            return float(cleaned_token)

    return None


def is_respiratory_event(event_type: str) -> bool:
    normalized_event_type = event_type.upper()
    return normalized_event_type.startswith(RESPIRATORY_EVENT_PREFIXES)


def parse_resp_event_file(
    record_id: str,
    event_path: Path | None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if event_path is None:
        return [], []

    events: list[dict[str, object]] = []
    parse_errors: list[dict[str, object]] = []

    with event_path.open("r", encoding="utf-8", errors="ignore") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            tokens = line.split()
            if not tokens or not TIME_PATTERN.match(tokens[0]):
                continue

            if len(tokens) < 3:
                parse_errors.append(
                    make_parse_error(
                        record_id,
                        event_path,
                        line_number,
                        line,
                        "Expected time, event type, and duration.",
                    )
                )
                continue

            start_sec = time_to_seconds(tokens[0])
            event_type = tokens[1]
            duration_sec = parse_duration(tokens[2:])

            if duration_sec is None:
                parse_errors.append(
                    make_parse_error(
                        record_id,
                        event_path,
                        line_number,
                        line,
                        "No numeric duration after event type.",
                    )
                )
                continue

            if not is_respiratory_event(event_type):
                continue

            events.append(
                {
                    "start_sec": float(start_sec),
                    "end_sec": float(start_sec) + duration_sec,
                    "event_type": event_type,
                    "duration_sec": duration_sec,
                }
            )

    return events, parse_errors


def make_parse_error(
    record_id: str,
    path: Path,
    line_number: int,
    line: str,
    error: str,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "file": str(path.relative_to(PROJECT_ROOT)),
        "line_number": line_number,
        "line": line,
        "error": error,
    }


def parse_stage_value(line: str) -> int | float | None:
    token = line.strip().split()[0] if line.strip() else ""
    if not token:
        return None

    value = float(token)
    if value.is_integer():
        return int(value)

    return value


def read_stage_file(stage_path: Path | None) -> list[int | float | None]:
    if stage_path is None:
        return []

    stages: list[int | float | None] = []
    with stage_path.open("r", encoding="utf-8", errors="ignore") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue

            try:
                stages.append(parse_stage_value(line))
            except (ValueError, IndexError):
                stages.append(None)

    return stages


def read_record_duration_seconds(signal_path: Path) -> float:
    pyedflib = import_pyedflib()

    with pyedflib.EdfReader(str(signal_path)) as reader:
        return float(reader.file_duration)


def find_optional_file(record_id: str, suffix: str) -> Path | None:
    matches = sorted(DATA_RAW_DIR.glob(f"{record_id}{suffix}"))
    if matches:
        return matches[0]

    return None


def format_number_list(values: list[float]) -> str:
    return "; ".join(f"{value:g}" for value in values)


def format_text_list(values: list[str]) -> str:
    return "; ".join(values)


def find_epoch_events(
    events: list[dict[str, object]],
    epoch_start_sec: int,
    epoch_end_sec: int,
) -> list[dict[str, object]]:
    return [
        event
        for event in events
        if float(event["start_sec"]) < epoch_end_sec
        and float(event["end_sec"]) > epoch_start_sec
    ]


def build_epoch_rows(
    record_id: str,
    duration_seconds: float,
    events: list[dict[str, object]],
    stages: list[int | float | None],
) -> list[dict[str, object]]:
    n_epochs = int(duration_seconds // EPOCH_SECONDS)
    rows = []

    for epoch_id in range(n_epochs):
        start_sec = epoch_id * EPOCH_SECONDS
        end_sec = start_sec + EPOCH_SECONDS
        epoch_events = find_epoch_events(events, start_sec, end_sec)

        label_binary = int(bool(epoch_events))
        label = POSITIVE_LABEL if label_binary else NEGATIVE_LABEL
        sleep_stage = stages[epoch_id] if epoch_id < len(stages) else None

        rows.append(
            {
                "record_id": record_id,
                "epoch_id": epoch_id,
                "start_sec": start_sec,
                "end_sec": end_sec,
                "sleep_stage": sleep_stage,
                "label": label,
                "label_binary": label_binary,
                "n_events": len(epoch_events),
                "event_types": format_text_list(
                    [str(event["event_type"]) for event in epoch_events]
                ),
                "event_start_sec": format_number_list(
                    [float(event["start_sec"]) for event in epoch_events]
                ),
                "event_duration_sec": format_number_list(
                    [float(event["duration_sec"]) for event in epoch_events]
                ),
            }
        )

    return rows


def make_summary_row(
    record_id: str,
    duration_seconds: float | None,
    epoch_rows: list[dict[str, object]],
    events: list[dict[str, object]],
    parse_errors: list[dict[str, object]],
    stage_path: Path | None,
    stages: list[int | float | None],
) -> dict[str, object]:
    n_epochs = len(epoch_rows)
    n_apnea_hypopnea = sum(int(row["label_binary"]) for row in epoch_rows)
    n_normal = n_epochs - n_apnea_hypopnea
    positive_rate = n_apnea_hypopnea / n_epochs if n_epochs else 0.0

    return {
        "record_id": record_id,
        "duration_seconds": duration_seconds,
        "n_epochs": n_epochs,
        "n_normal": n_normal,
        "n_apnea_hypopnea": n_apnea_hypopnea,
        "positive_rate": positive_rate,
        "n_parsed_events": len(events),
        "n_parse_errors": len(parse_errors),
        "has_stage_file": stage_path is not None,
        "n_stage_rows": len(stages),
    }


def build_overall_summary(
    summary_rows: list[dict[str, object]],
) -> dict[str, object]:
    n_records = len(summary_rows)
    n_epochs = sum(int(row["n_epochs"]) for row in summary_rows)
    n_normal = sum(int(row["n_normal"]) for row in summary_rows)
    n_apnea_hypopnea = sum(int(row["n_apnea_hypopnea"]) for row in summary_rows)
    positive_rate = n_apnea_hypopnea / n_epochs if n_epochs else 0.0
    n_parsed_events = sum(int(row["n_parsed_events"]) for row in summary_rows)
    n_parse_errors = sum(int(row["n_parse_errors"]) for row in summary_rows)

    return {
        "n_records": n_records,
        "n_epochs": n_epochs,
        "n_normal": n_normal,
        "n_apnea_hypopnea": n_apnea_hypopnea,
        "positive_rate": positive_rate,
        "n_parsed_events": n_parsed_events,
        "n_parse_errors": n_parse_errors,
    }


def save_table(rows: list[dict[str, object]], columns: list[str], output_path: Path) -> None:
    table = pd.DataFrame(rows, columns=columns)
    table.to_csv(output_path, index=False)


def main() -> None:
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    signal_files = find_signal_files(DATA_RAW_DIR)

    all_epoch_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    all_parse_errors: list[dict[str, object]] = []

    print(f"Raw data directory: {DATA_RAW_DIR}")
    print(f"Found UCDDB .rec files: {len(signal_files)}")

    for signal_path in tqdm(signal_files, desc="Building epochs"):
        record_id = get_record_id(signal_path)
        if record_id is None:
            continue

        event_path = find_optional_file(record_id, "_respevt.txt")
        stage_path = find_optional_file(record_id, "_stage.txt")

        events, parse_errors = parse_resp_event_file(record_id, event_path)
        stages = read_stage_file(stage_path)

        try:
            duration_seconds = read_record_duration_seconds(signal_path)
            epoch_rows = build_epoch_rows(record_id, duration_seconds, events, stages)
        except Exception as exc:  # noqa: BLE001
            print(f"Could not build epochs for {record_id}: {type(exc).__name__}: {exc}")
            duration_seconds = None
            epoch_rows = []

        all_epoch_rows.extend(epoch_rows)
        all_parse_errors.extend(parse_errors)
        summary_rows.append(
            make_summary_row(
                record_id=record_id,
                duration_seconds=duration_seconds,
                epoch_rows=epoch_rows,
                events=events,
                parse_errors=parse_errors,
                stage_path=stage_path,
                stages=stages,
            )
        )

    overall_summary = build_overall_summary(summary_rows)

    epochs_path = DATA_PROCESSED_DIR / "epochs.csv"
    summary_by_record_path = REPORTS_TABLES_DIR / "epochs_summary_by_record.csv"
    summary_overall_path = REPORTS_TABLES_DIR / "epochs_summary_overall.csv"
    parse_errors_path = REPORTS_TABLES_DIR / "resp_event_parse_errors.csv"

    save_table(all_epoch_rows, EPOCH_COLUMNS, epochs_path)
    save_table(summary_rows, SUMMARY_BY_RECORD_COLUMNS, summary_by_record_path)
    save_table([overall_summary], SUMMARY_OVERALL_COLUMNS, summary_overall_path)
    save_table(all_parse_errors, PARSE_ERROR_COLUMNS, parse_errors_path)

    print("\nEpoch build summary")
    print(f"  Processed records: {overall_summary['n_records']}")
    print(f"  Created epochs: {overall_summary['n_epochs']}")
    print(f"  Positive epochs: {overall_summary['n_apnea_hypopnea']}")
    print(f"  Normal epochs: {overall_summary['n_normal']}")
    print(f"  Positive rate: {overall_summary['positive_rate']:.4f}")
    print(f"  Saved epochs: {epochs_path}")

    if not signal_files:
        print("No .rec files found. Run: python scripts/pipeline/00_download_ucddb.py")


if __name__ == "__main__":
    main()
