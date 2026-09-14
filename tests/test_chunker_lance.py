import hashlib
from pathlib import Path

from chunking.chunker import chunk_documents, read_chunks, write_chunks


def test_chunk_hash_is_deterministic_and_strategy_versioned() -> None:
    document = {"doc_id": "doc-1", "text": "A short research abstract."}

    first = chunk_documents([document])
    second = chunk_documents([document])

    assert first == second
    expected_hash = hashlib.sha256(
        f"{first[0]['chunk_text']}1.0.0".encode("utf-8")
    ).hexdigest()
    assert first[0]["chunk_hash"] == expected_hash


def test_chunks_round_trip_through_lance(tmp_path: Path) -> None:
    document = {
        "doc_id": "doc-1",
        "text": "This is a document that must survive a Lance round trip.",
        "pii_masked_flag": True,
    }
    chunks = chunk_documents([document])
    dataset_path = tmp_path / "chunks.lance"

    assert write_chunks(chunks, dataset_path) == dataset_path
    saved_chunks = read_chunks(dataset_path)

    assert len(saved_chunks) == len(chunks)
    assert saved_chunks[0]["chunk_hash"] == chunks[0]["chunk_hash"]
    assert saved_chunks[0]["pii_masked_flag"] is True