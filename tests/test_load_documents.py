import json
from pathlib import Path

from ingestion.load_documents import load_documents
from chunking.chunker import read_chunks


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