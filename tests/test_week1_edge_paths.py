import io
import json
import sys
import types
import zipfile
from pathlib import Path

import pytest

from chunking.chunker import ChunkingStrategy, chunk_documents, write_chunks
from contracts.contract_validator import load_contract, validate_file
from embedding.embedder import fastembedder, ingest_chunks
from embedding.model_registry import get_model
from ingestion.download_scifact import _find_member, download_scifact
from ingestion.load_documents import load_documents
from qdrant_client import QdrantClient


def test_strategy_and_chunk_validation_errors(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"
    missing_path.write_text("strategy_name: fixed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing fields"):
        ChunkingStrategy.from_yaml(missing_path)

    with pytest.raises(ValueError, match="doc_id and text"):
        chunk_documents([{"doc_id": "doc-1"}])
    with pytest.raises(TypeError, match="must be a string"):
        chunk_documents([{"doc_id": "doc-1", "text": 1}])
    with pytest.raises(ValueError, match="without chunk"):
        write_chunks([], tmp_path / "chunks.lance")


def test_contract_file_formats_and_invalid_json(tmp_path: Path) -> None:
    assert load_contract()["contract_name"] == "governed_document"
    valid = {
        "doc_id": "doc-1",
        "title": "Title",
        "text": "Text",
        "metadata": {
            "source_uri": "https://example.test",
            "author": "Author",
            "classification_level": "public",
            "language": "en",
        },
    }
    json_path = tmp_path / "documents.json"
    json_path.write_text(json.dumps(valid), encoding="utf-8")
    assert validate_file(json_path)["valid_count"] == 1

    bad_path = tmp_path / "documents.txt"
    bad_path.write_text("text", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\.json or \.jsonl"):
        validate_file(bad_path)


def test_loader_quarantines_malformed_records(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"_id": "valid", "title": "Title", "text": "Text"})
        + "\n"
        + json.dumps({"_id": "invalid", "title": "Missing text"})
        + "\n",
        encoding="utf-8",
    )
    quarantine = tmp_path / "quarantine.json"
    report = load_documents(
        corpus,
        dataset_path=tmp_path / "chunks.lance",
        quarantine_path=quarantine,
    )
    assert report["valid_document_count"] == 1
    assert report["invalid_document_count"] == 1
    assert quarantine.is_file()


def test_qdrant_alias_replacement_and_dimension_guard() -> None:
    client = QdrantClient(":memory:")
    chunk = {
        "chunk_id": "doc:0",
        "chunk_text": "text",
        "doc_id": "doc",
        "strategy_name": "fixed_size_v1",
        "strategy_version": "1.0.0",
    }
    embed = lambda texts: [[0.1] * 384 for _ in texts]
    ingest_chunks([chunk], client, embed=embed, collection_name="first")
    ingest_chunks([chunk], client, embed=embed, collection_name="second")
    assert client.get_aliases().aliases[0].collection_name == "second"

    with pytest.raises(ValueError, match="dimension"):
        ingest_chunks([chunk], client, embed=lambda texts: [[0.1] * 3 for _ in texts])


def test_fastembed_adapter_and_existing_collection_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            assert model_name == "BAAI/bge-small-en-v1.5"

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.2] * 384 for _ in texts]

    monkeypatch.setitem(sys.modules, "fastembed", types.SimpleNamespace(TextEmbedding=FakeTextEmbedding))
    assert len(list(fastembedder()(["text"]))) == 1

    client = QdrantClient(":memory:")
    client.create_collection("existing", vectors_config={"size": 3, "distance": "Cosine"})
    with pytest.raises(ValueError, match="does not match"):
        ingest_chunks(
            [{"chunk_id": "c", "chunk_text": "t", "doc_id": "d"}],
            client,
            embed=lambda texts: [[0.1] * 384 for _ in texts],
            collection_name="existing",
        )


def test_model_registry_unknown_key() -> None:
    with pytest.raises(KeyError, match="Unknown"):
        get_model("missing")


def test_scifact_download_helpers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    archive_buffer = io.BytesIO()
    with zipfile.ZipFile(archive_buffer, "w") as archive:
        archive.writestr("scifact/corpus.jsonl", "{}\n")
        archive.writestr("scifact/queries.jsonl", "{}\n")
        archive.writestr("scifact/qrels/test.tsv", "q0 0 1\n")
    archive_bytes = archive_buffer.getvalue()

    class Response:
        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *_: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, chunk_size: int) -> list[bytes]:
            return [archive_bytes]

    monkeypatch.setattr("ingestion.download_scifact.requests.get", lambda *args, **kwargs: Response())
    output = tmp_path / "scifact"
    sizes = download_scifact(output_dir=output, url="https://example.test/scifact.zip")
    assert set(sizes) == {"corpus.jsonl", "queries.jsonl", "qrels/test.tsv"}
    assert (output / "qrels/test.tsv").is_file()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        assert _find_member(archive, "corpus.jsonl") == "scifact/corpus.jsonl"