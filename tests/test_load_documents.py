import json
from pathlib import Path
from typing import Any, Sequence

from ingestion.load_documents import load_documents
from chunking.chunker import read_chunks
from ingestion.bootstrap import bootstrap_baseline


def test_loader_sanitizes_before_chunking_and_persists_lance(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps(
            {
                "_id": "doc-1",
                "title": "Research contact analyst@example.com",
                "text": "Reach analyst@example.com at 192.168.1.2.",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = load_documents(
        corpus_path,
        dataset_path=tmp_path / "chunks.lance",
        quarantine_path=tmp_path / "quarantine.json",
    )
    chunks = read_chunks(tmp_path / "chunks.lance")

    assert report["valid_document_count"] == 1
    assert report["invalid_document_count"] == 0
    assert report["chunk_count"] == len(chunks)
    assert chunks[0]["pii_masked_flag"] is True
    assert "analyst@example.com" not in chunks[0]["chunk_text"]
    assert "192.168.1.2" not in chunks[0]["chunk_text"]


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.upserts: list[dict[str, Any]] = []
        self.aliases: list[Any] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, **kwargs: Any) -> None:
        self.collections.add(kwargs["collection_name"])

    def upsert(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)

    def get_aliases(self) -> Any:
        return type("Aliases", (), {"aliases": self.aliases})()

    def update_collection_aliases(self, **kwargs: Any) -> None:
        self.aliases.append(kwargs)


def test_bootstrap_materializes_catalog_and_baseline_vectors(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    corpus_path.write_text(
        json.dumps({"_id": "doc-1", "title": "Title", "text": "A short paper."}) + "\n",
        encoding="utf-8",
    )
    client = FakeQdrant()

    def fake_embed(texts: Sequence[str]) -> list[list[float]]:
        return [[0.1] * 384 for _ in texts]

    report = bootstrap_baseline(
        corpus_path=corpus_path,
        dataset_path=tmp_path / "chunks.lance",
        quarantine_path=tmp_path / "quarantine.json",
        database_path=tmp_path / "catalog.duckdb",
        qdrant_client=client,
        embed=fake_embed,
    )

    assert report["catalog_document_count"] == 1
    assert report["catalog_chunk_count"] == 1
    assert report["vector_count"] == 1
    assert client.collections == {"scifact_v1"}
    assert len(client.upserts) == 1
    assert client.upserts[0]["points"][0].payload["vector_id"] == "vec_0001"

    from catalog.db import CatalogDB

    with CatalogDB(tmp_path / "catalog.duckdb") as catalog:
        assert catalog.query("SELECT COUNT(*) FROM documents") == [(1,)]
        assert catalog.query("SELECT COUNT(*) FROM chunks") == [(1,)]
        assert catalog.query("SELECT COUNT(*) FROM vectors") == [(1,)]