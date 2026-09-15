import json
from pathlib import Path

import pytest

from evaluation.ingest_scifact import build_scifact_catalog, write_scifact_catalog


def test_build_scifact_catalog_joins_qrels_and_query_text(tmp_path: Path) -> None:
    qrels = tmp_path / "test.tsv"
    qrels.write_text(
        "query-id\tcorpus-id\tscore\n"
        "2\tdoc-b\t1\n"
        "1\tdoc-a\t1\n"
        "2\tdoc-ignored\t0\n",
        encoding="utf-8",
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        '{"_id":"1","text":"first","metadata":{}}\n'
        '{"_id":"2","text":"second","metadata":{}}\n',
        encoding="utf-8",
    )

    assert build_scifact_catalog(qrels, queries) == [
        {"query_id": "1", "text": "first", "relevant_doc_ids": ["doc-a"]},
        {"query_id": "2", "text": "second", "relevant_doc_ids": ["doc-b"]},
    ]


def test_write_scifact_catalog_creates_json_artifact(tmp_path: Path) -> None:
    qrels = tmp_path / "test.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\n1\tdoc-1\t1\n", encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"_id":"1","text":"claim"}\n', encoding="utf-8")
    output = tmp_path / "evaluation" / "catalog.json"

    assert write_scifact_catalog(output, qrels, queries) == 1
    assert json.loads(output.read_text(encoding="utf-8"))[0]["query_id"] == "1"


def test_build_scifact_catalog_rejects_qrels_for_missing_query(tmp_path: Path) -> None:
    qrels = tmp_path / "test.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\n9\tdoc-1\t1\n", encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"_id":"1","text":"claim"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="absent"):
        build_scifact_catalog(qrels, queries)