from __future__ import annotations

import fnmatch
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import DATA_RAW_DIR  # noqa: E402


BASE_URL = "https://physionet.org/files/ucddb/1.0.0/"
INDEX_TIMEOUT_SECONDS = 30
CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 180
MAX_DOWNLOAD_ATTEMPTS = 3
CHUNK_SIZE = 1024 * 1024

REQUIRED_FILES = {
    "RECORDS",
    "SubjectDetails.xls",
    "SHA256SUMS.txt",
}

REQUIRED_PATTERNS = (
    "ucddb*.rec",
    "ucddb*_respevt.txt",
    "ucddb*_stage.txt",
)

EXCLUDED_PATTERNS = (
    "*_lifecard.edf",
)


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return

        for key, value in attrs:
            if key == "href" and value:
                self.hrefs.append(value)


def get_filename_from_href(href: str) -> str | None:
    parsed_path = urlparse(href).path
    filename = Path(unquote(parsed_path)).name

    if not filename or filename == "..":
        return None

    return filename


def should_download(filename: str) -> bool:
    if any(fnmatch.fnmatch(filename, pattern) for pattern in EXCLUDED_PATTERNS):
        return False

    if filename in REQUIRED_FILES:
        return True

    return any(fnmatch.fnmatch(filename, pattern) for pattern in REQUIRED_PATTERNS)


def fetch_index(session: requests.Session) -> list[str]:
    response = session.get(BASE_URL, timeout=INDEX_TIMEOUT_SECONDS)
    response.raise_for_status()

    parser = LinkParser()
    parser.feed(response.text)

    filenames = []
    for href in parser.hrefs:
        filename = get_filename_from_href(href)
        if filename and should_download(filename):
            filenames.append(filename)

    return sorted(set(filenames))


def get_remote_size(session: requests.Session, filename: str) -> int | None:
    url = urljoin(BASE_URL, filename)

    response = session.head(
        url,
        allow_redirects=True,
        timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
    )
    response.raise_for_status()

    content_length = response.headers.get("content-length")
    if content_length is None:
        return None

    return int(content_length)


def remove_file(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError as exc:
        print(f"Could not remove {path}: {exc}")


def should_skip_existing_file(output_path: Path, remote_size: int | None) -> bool:
    if not output_path.exists() or not output_path.is_file():
        return False

    local_size = output_path.stat().st_size

    if remote_size is not None and local_size == remote_size:
        return True

    if remote_size is None:
        print(
            f"Existing file size cannot be verified, downloading again: "
            f"{output_path.name}"
        )
    elif local_size < remote_size:
        print(
            f"Existing file is incomplete, downloading again: "
            f"{output_path.name} ({format_size(local_size)} / {format_size(remote_size)})"
        )
    else:
        print(
            f"Existing file size differs from remote, downloading again: "
            f"{output_path.name} ({format_size(local_size)} / {format_size(remote_size)})"
        )

    remove_file(output_path)
    return False


def download_file_once(
    session: requests.Session,
    filename: str,
    output_dir: Path,
    remote_size: int | None,
) -> bool:
    url = urljoin(BASE_URL, filename)
    output_path = output_dir / filename
    temporary_path = output_path.with_name(f"{output_path.name}.part")

    remove_file(temporary_path)

    with session.get(
        url,
        stream=True,
        timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
    ) as response:
        response.raise_for_status()
        response_size = response.headers.get("content-length")
        total_size = int(response_size) if response_size else remote_size

        with temporary_path.open("wb") as file:
            with tqdm(
                total=total_size if total_size and total_size > 0 else None,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=filename,
            ) as progress:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    file.write(chunk)
                    progress.update(len(chunk))

    downloaded_size = temporary_path.stat().st_size
    expected_size = remote_size or total_size
    if expected_size is not None and downloaded_size != expected_size:
        raise OSError(
            f"incomplete download: got {downloaded_size} bytes, "
            f"expected {expected_size} bytes"
        )

    temporary_path.replace(output_path)
    return True


def download_file(
    session: requests.Session,
    filename: str,
    output_dir: Path,
    remote_size: int | None,
) -> bool:
    output_path = output_dir / filename
    temporary_path = output_path.with_name(f"{output_path.name}.part")

    for attempt in range(1, MAX_DOWNLOAD_ATTEMPTS + 1):
        try:
            if download_file_once(session, filename, output_dir, remote_size):
                return True
        except requests.RequestException as exc:
            print(f"Download error for {filename} (attempt {attempt}): {exc}")
        except OSError as exc:
            print(f"File error for {filename} (attempt {attempt}): {exc}")

        remove_file(temporary_path)

        if attempt < MAX_DOWNLOAD_ATTEMPTS:
            print(f"Retrying {filename}...")

    return False


def calculate_directory_size(path: Path) -> int:
    if not path.exists():
        return 0

    return sum(file.stat().st_size for file in path.rglob("*") if file.is_file())


def format_size(size_bytes: int) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(size_bytes)

    for unit in units:
        if size < 1024 or unit == units[-1]:
            return f"{size:.2f} {unit}"
        size /= 1024

    return f"{size_bytes} B"


def main() -> None:
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)

    with requests.Session() as session:
        try:
            filenames = fetch_index(session)
        except requests.RequestException as exc:
            print(f"Could not read PhysioNet directory index: {exc}")
            return

        if not filenames:
            print("No matching UCDDB files were found in the PhysioNet directory index.")
            return

        print(f"Found matching files: {len(filenames)}")
        print(f"Output directory: {DATA_RAW_DIR}")

        downloaded_count = 0
        skipped_count = 0
        failed_files: list[str] = []

        for filename in filenames:
            output_path = DATA_RAW_DIR / filename

            try:
                remote_size = get_remote_size(session, filename)
            except requests.RequestException as exc:
                print(f"Could not get remote size for {filename}: {exc}")
                failed_files.append(filename)
                continue
            except ValueError as exc:
                print(f"Invalid Content-Length for {filename}: {exc}")
                failed_files.append(filename)
                continue

            if should_skip_existing_file(output_path, remote_size):
                skipped_count += 1
                print(
                    f"Skipping complete existing file: "
                    f"{filename} ({format_size(remote_size or output_path.stat().st_size)})"
                )
                continue

            if download_file(session, filename, DATA_RAW_DIR, remote_size):
                downloaded_count += 1
            else:
                failed_files.append(filename)

    total_size = calculate_directory_size(DATA_RAW_DIR)

    print("\nDownload summary")
    print(f"  Downloaded files: {downloaded_count}")
    print(f"  Skipped existing files: {skipped_count}")
    print(f"  Failed files: {len(failed_files)}")
    print(f"  Total data/raw size: {format_size(total_size)}")

    if failed_files:
        print("\nFailed file list")
        for filename in failed_files:
            print(f"  - {filename}")


if __name__ == "__main__":
    main()
