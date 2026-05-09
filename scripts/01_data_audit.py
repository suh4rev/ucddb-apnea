from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_RAW_DIR, REPORTS_TABLES_DIR  # noqa: E402


SIGNAL_EXTENSIONS = {".edf", ".bdf", ".rec"}
ANNOTATION_EXTENSIONS = {".txt", ".csv", ".tsv", ".xml", ".ann"}
RESPIRATORY_EVENT_KEYWORDS = (
    "respevt",
    "resp_event",
    "respiratory",
    "apnea",
    "apnoea",
    "hypopnea",
    "hypopnoea",
)

AUDIT_COLUMNS = [
    "record_id",
    "signal_file",
    "signal_read_ok",
    "read_error",
    "duration_seconds",
    "n_channels",
    "channel_names",
    "sampling_frequencies",
    "annotation_files",
    "has_respiratory_event_annotations",
    "respiratory_event_files",
    "respiratory_event_lines",
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


def find_ucddb_files(raw_dir: Path) -> list[Path]:
    if not raw_dir.exists():
        return []

    return sorted(path for path in raw_dir.rglob("*") if path.is_file())


def get_record_id(path: Path) -> str | None:
    match = re.search(r"ucddb\d+", path.name.lower())
    if match:
        return match.group(0)

    return None


def group_files_by_record(files: list[Path]) -> dict[str, list[Path]]:
    records: dict[str, list[Path]] = {}

    for path in files:
        record_id = get_record_id(path)
        if record_id is None:
            continue

        records.setdefault(record_id, []).append(path)

    return records


def is_signal_file(path: Path) -> bool:
    return path.suffix.lower() in SIGNAL_EXTENSIONS


def is_annotation_file(path: Path) -> bool:
    return path.suffix.lower() in ANNOTATION_EXTENSIONS and not is_signal_file(path)


def is_respiratory_event_file(path: Path) -> bool:
    name = path.name.lower()
    return is_annotation_file(path) and any(
        keyword in name for keyword in RESPIRATORY_EVENT_KEYWORDS
    )


def choose_signal_file(paths: list[Path]) -> Path | None:
    signal_files = sorted(path for path in paths if is_signal_file(path))
    if not signal_files:
        return None

    rec_files = [
        path
        for path in signal_files
        if path.suffix.lower() == ".rec" and "lifecard" not in path.name.lower()
    ]
    if rec_files:
        return rec_files[0]

    non_lifecard_files = [
        path for path in signal_files if "lifecard" not in path.name.lower()
    ]
    if non_lifecard_files:
        return non_lifecard_files[0]

    return signal_files[0]


def count_non_empty_lines(path: Path) -> int | None:
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as file:
            return sum(1 for line in file if line.strip())
    except OSError:
        return None


def read_signal_metadata(signal_path: Path) -> dict[str, object]:
    pyedflib = import_pyedflib()

    with pyedflib.EdfReader(str(signal_path)) as reader:
        n_channels = reader.signals_in_file
        channel_names = reader.getSignalLabels()
        sampling_frequencies = reader.getSampleFrequencies()
        duration_seconds = reader.file_duration

        if n_channels > 0:
            sample_count = reader.getNSamples()[0]
            n_samples_to_read = min(10, sample_count)
            if n_samples_to_read > 0:
                reader.readSignal(0, start=0, n=n_samples_to_read)

    return {
        "duration_seconds": duration_seconds,
        "n_channels": n_channels,
        "channel_names": channel_names,
        "sampling_frequencies": sampling_frequencies,
    }


def audit_record(record_id: str, paths: list[Path]) -> dict[str, object]:
    signal_path = choose_signal_file(paths)
    annotation_files = sorted(path for path in paths if is_annotation_file(path))
    respiratory_event_files = sorted(
        path for path in annotation_files if is_respiratory_event_file(path)
    )

    row: dict[str, object] = {
        "record_id": record_id,
        "signal_file": str(signal_path.relative_to(PROJECT_ROOT)) if signal_path else "",
        "signal_read_ok": False,
        "read_error": "",
        "duration_seconds": None,
        "n_channels": None,
        "channel_names": "",
        "sampling_frequencies": "",
        "annotation_files": "; ".join(
            str(path.relative_to(PROJECT_ROOT)) for path in annotation_files
        ),
        "has_respiratory_event_annotations": bool(respiratory_event_files),
        "respiratory_event_files": "; ".join(
            str(path.relative_to(PROJECT_ROOT)) for path in respiratory_event_files
        ),
        "respiratory_event_lines": None,
    }

    if respiratory_event_files:
        row["respiratory_event_lines"] = count_non_empty_lines(respiratory_event_files[0])

    if signal_path is None:
        row["read_error"] = "No EDF/BDF signal file found for this record."
        return row

    try:
        metadata = read_signal_metadata(signal_path)
    except Exception as exc:  # noqa: BLE001
        row["read_error"] = f"{type(exc).__name__}: {exc}"
        return row

    row["signal_read_ok"] = True
    row["duration_seconds"] = metadata["duration_seconds"]
    row["n_channels"] = metadata["n_channels"]
    row["channel_names"] = "; ".join(metadata["channel_names"])
    row["sampling_frequencies"] = "; ".join(
        str(freq) for freq in metadata["sampling_frequencies"]
    )

    return row


def print_record_summary(row: dict[str, object]) -> None:
    print(f"\nRecord: {row['record_id']}")
    print(f"  Signal file: {row['signal_file'] or 'not found'}")
    print(f"  Signal readable: {row['signal_read_ok']}")

    if row["read_error"]:
        print(f"  Read error: {row['read_error']}")

    print(f"  Channels: {row['channel_names'] or 'not available'}")
    print(f"  Sampling frequencies: {row['sampling_frequencies'] or 'not available'}")
    print(
        "  Respiratory event annotations: "
        f"{'yes' if row['has_respiratory_event_annotations'] else 'no'}"
    )

    if row["respiratory_event_files"]:
        print(f"  Respiratory event files: {row['respiratory_event_files']}")


def main() -> None:
    REPORTS_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    files = find_ucddb_files(DATA_RAW_DIR)
    records = group_files_by_record(files)

    print(f"Raw data directory: {DATA_RAW_DIR}")
    print(f"Found files: {len(files)}")
    print(f"Found UCDDB records: {len(records)}")

    record_items = sorted(records.items())
    if record_items:
        record_items = tqdm(record_items, desc="Auditing records")

    rows = []
    for record_id, paths in record_items:
        row = audit_record(record_id, paths)
        rows.append(row)
        print_record_summary(row)

    audit_table = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    output_path = REPORTS_TABLES_DIR / "ucddb_audit.csv"
    audit_table.to_csv(output_path, index=False)

    print(f"\nSaved audit table: {output_path}")

    if audit_table.empty:
        print(
            "No UCDDB records were found. Put files like "
            "ucddb002.rec and ucddb002_respevt.txt into data/raw/."
        )


if __name__ == "__main__":
    main()
