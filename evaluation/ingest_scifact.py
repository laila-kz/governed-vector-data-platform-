"""Build the canonical SciFact relevance catalog used by IR evaluation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable

DEFAULT_QRELS_PATH = Path("data/raw/scifact/qrels/test.tsv")
DEFAULT_QUERIES_PATH = Path("data/raw/scifact/queries.jsonl")
DEFAULT_OUTPUT_PATH = Path("evaluation/beir_scifact_qrels.json")


def _read_qrels(path: Path) -> dict[str, list[str]]:
    with path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        expected_columns = {"query-id", "corpus-id", "score"}
        if set(reader.fieldnames or ()) != expected_columns:
            raise ValueError(
                f"SciFact qrels must have columns {sorted(expected_columns)}"
            )

        relevance: dict[str, list[str]] = {}
        for row_number, row in enumerate(reader, start=2):
            query_id = str(row["query-id"]).strip()
            corpus_id = str(row["corpus-id"]).strip()
            if not query_id or not corpus_id:
                raise ValueError(f"qrels row {row_number} has an empty identifier")
            try:
                score = float(row["score"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"qrels row {row_number} has an invalid score") from error
            if score <= 0:
                continue
            relevance.setdefault(query_id, []).append(corpus_id)

    return relevance


def _read_queries(path: Path) -> dict[str, str]:
    queries: dict[str, str] = {}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"queries line {line_number} is not valid JSON") from error
            if not isinstance(record, dict) or "_id" not in record or "text" not in record:
                raise ValueError(f"queries line {line_number} must contain _id and text")
            query_id = str(record["_id"]).strip()
            text = str(record["text"])
            if not query_id or not text.strip():
                raise ValueError(f"queries line {line_number} has an empty identifier or text")
            if query_id in queries:
                raise ValueError(f"queries contains duplicate query id {query_id!r}")
            queries[query_id] = text

    return queries


def build_scifact_catalog(
    qrels_path: Path | str = DEFAULT_QRELS_PATH,
    queries_path: Path | str = DEFAULT_QUERIES_PATH,
) -> list[dict[str, Any]]:
    """Join qrels with query text into the canonical evaluation records."""
    relevance = _read_qrels(Path(qrels_path))
    queries = _read_queries(Path(queries_path))
    missing_queries = sorted(set(relevance) - set(queries))
    if missing_queries:
        raise ValueError(
            f"qrels reference query ids absent from queries.jsonl: {missing_queries[:5]}"
        )

    def sort_key(query_id: str) -> tuple[int, str]:
        try:
            return (0, f"{int(query_id):020d}")
        except ValueError:
            return (1, query_id)

    return [
        {
            "query_id": query_id,
            "text": queries[query_id],
            "relevant_doc_ids": relevance[query_id],
        }
        for query_id in sorted(relevance, key=sort_key)
    ]


def write_scifact_catalog(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    qrels_path: Path | str = DEFAULT_QRELS_PATH,
    queries_path: Path | str = DEFAULT_QUERIES_PATH,
) -> int:
    """Write the canonical catalog and return its query count."""
    records = build_scifact_catalog(qrels_path, queries_path)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return len(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qrels", type=Path, default=DEFAULT_QRELS_PATH)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    count = write_scifact_catalog(args.output, args.qrels, args.queries)
    print(f"wrote {count:,} SciFact evaluation queries to {args.output}")


if __name__ == "__main__":
    main()