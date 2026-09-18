"""Validate, sanitize, chunk, and persist the SciFact document corpus."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from chunking.chunker import DEFAULT_DATASET_PATH, chunk_documents, write_chunks
from contracts.contract_validator import validate_document
from ingestion.sanitization import sanitize_document

DEFAULT_CORPUS_PATH = Path("data/raw/scifact/corpus.jsonl")
DEFAULT_QUARANTINE_PATH = Path("data/quarantine/scifact_load_errors.json")


def _canonical_scifact_record(raw_record: dict[str, Any]) -> dict[str, Any]:
    """Map the BEIR corpus shape to the governed document contract."""
    if not {"_id", "title", "text"}.issubset(raw_record):
        raise ValueError("SciFact record must contain _id, title, and text")
    document_id = str(raw_record["_id"])
    return {
        "doc_id": document_id,
        "title": raw_record["title"],
        "text": raw_record["text"],
        "metadata": {
            "source_uri": f"beir://scifact/{document_id}",
            "author": "BEIR/SciFact",
            "classification_level": "public",
            "language": "en",
        },
    }


def _iter_jsonl(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if line.strip():
                yield line_number, json.loads(line)


def load_documents(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    quarantine_path: Path = DEFAULT_QUARANTINE_PATH,
) -> dict[str, Any]:
    """Load SciFact into Lance after contract validation and sanitization."""
    valid_documents, invalid_records = load_valid_documents(corpus_path)

    chunks = chunk_documents(valid_documents)
    if chunks:
        write_chunks(chunks, dataset_path)

    report: dict[str, Any] = {
        "source_file": str(corpus_path),
        "valid_document_count": len(valid_documents),
        "invalid_document_count": len(invalid_records),
        "chunk_count": len(chunks),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    if invalid_records:
        quarantine_path = Path(quarantine_path)
        quarantine_path.parent.mkdir(parents=True, exist_ok=True)
        report["invalid_records"] = invalid_records
        report["quarantine_report"] = str(quarantine_path)
        with quarantine_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, ensure_ascii=True)
            report_file.write("\n")
    return report


def load_valid_documents(
    corpus_path: Path = DEFAULT_CORPUS_PATH,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return governed documents and rejected records from a SciFact JSONL file."""
    valid_documents: list[dict[str, Any]] = []
    invalid_records: list[dict[str, Any]] = []
    for line_number, raw_record in _iter_jsonl(Path(corpus_path)):
        try:
            canonical = _canonical_scifact_record(raw_record)
            validated = validate_document(canonical)
            valid_documents.append(sanitize_document(validated.model_dump()))
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            invalid_records.append(
                {
                    "record_number": line_number,
                    "record": raw_record,
                    "error": str(error),
                }
            )
    return valid_documents, invalid_records


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE_PATH)
    args = parser.parse_args()
    report = load_documents(args.corpus, args.dataset, args.quarantine)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if report["invalid_document_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()