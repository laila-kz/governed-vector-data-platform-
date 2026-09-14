"""Download the BEIR SciFact corpus and relevance judgments."""

from __future__ import annotations

import argparse
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Final

import requests

DEFAULT_URL: Final = (
    "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
)
DEFAULT_OUTPUT_DIR: Final = Path("data/raw/scifact")
REQUIRED_FILES: Final = ("corpus.jsonl", "queries.jsonl", "qrels/test.tsv")


def _find_member(archive: zipfile.ZipFile, required_name: str) -> str:
    """Find a required file regardless of the archive's top-level directory."""
    matches = [
        name
        for name in archive.namelist()
        if Path(name.replace("\\", "/")).as_posix().endswith(required_name)
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one archive member ending in {required_name!r}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _write_member(
    archive: zipfile.ZipFile, member_name: str, destination: Path
) -> None:
    """Copy one ZIP member to a file without extracting arbitrary archive paths."""
    with archive.open(member_name) as source, destination.open("wb") as target:
        shutil.copyfileobj(source, target)


def download_scifact(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    url: str = DEFAULT_URL,
    timeout: tuple[float, float] = (10.0, 120.0),
) -> dict[str, int]:
    """Download SciFact and return the byte size of each saved file."""
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="scifact-download-") as temp_dir:
        temp_path = Path(temp_dir)
        archive_path = temp_path / "scifact.zip"

        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with archive_path.open("wb") as archive_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        archive_file.write(chunk)

        staging_dir = temp_path / "staged"
        staging_dir.mkdir()
        with zipfile.ZipFile(archive_path) as archive:
            if archive.testzip() is not None:
                raise RuntimeError("The downloaded SciFact archive failed its CRC check.")

            for required_name in REQUIRED_FILES:
                member_name = _find_member(archive, required_name)
                _write_member(archive, member_name, staging_dir / Path(required_name).name)

        output_dir.mkdir(parents=True, exist_ok=True)
        sizes: dict[str, int] = {}
        for required_name in REQUIRED_FILES:
            staged_file = staging_dir / Path(required_name).name
            destination = output_dir / required_name
            destination.parent.mkdir(parents=True, exist_ok=True)
            staged_file.replace(destination)
            sizes[required_name] = destination.stat().st_size

    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where corpus.jsonl, queries.jsonl, and qrels/test.tsv are saved.",
    )
    parser.add_argument("--url", default=DEFAULT_URL, help="SciFact ZIP URL.")
    args = parser.parse_args()

    sizes = download_scifact(output_dir=args.output_dir, url=args.url)
    for filename, size in sizes.items():
        print(f"saved {args.output_dir / filename} ({size:,} bytes)")


if __name__ == "__main__":
    main()
