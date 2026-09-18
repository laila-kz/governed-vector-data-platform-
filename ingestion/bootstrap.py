"""Build a reproducible SciFact baseline across Lance, DuckDB, and Qdrant."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from qdrant_client import QdrantClient

from catalog.db import CatalogDB
from chunking.chunker import ChunkingStrategy, DEFAULT_DATASET_PATH, read_chunks
from embedding.embedder import ingest_chunks
from embedding.model_registry import BASELINE_MODEL_V1, EmbeddingModel
from ingestion.load_documents import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_QUARANTINE_PATH,
    load_documents,
    load_valid_documents,
)

DEFAULT_DATABASE_PATH = Path("data/catalog.duckdb")
DEFAULT_COLLECTION_NAME = "scifact_v1"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_STRATEGY_PATH = Path("chunking/strategies/strategy_v1_fixed.yaml")


def _document_rows(documents: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for document in documents:
        metadata = document["metadata"]
        rows.append(
            {
                "doc_id": str(document["doc_id"]),
                "doc_version": 1,
                "title": str(document["title"]),
                "source_uri": str(metadata["source_uri"]),
                "author": str(metadata["author"]),
                "classification_level": str(metadata["classification_level"]),
                "language": str(metadata["language"]),
                "content_hash": hashlib.sha256(
                    str(document["text"]).encode("utf-8")
                ).hexdigest(),
                "pii_masked_flag": bool(document.get("pii_masked_flag", False)),
            }
        )
    return rows


def _chunk_rows(chunks: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "chunk_id": str(chunk["chunk_id"]),
            "doc_id": str(chunk["doc_id"]),
            "doc_version": int(chunk.get("doc_version", 1)),
            "strategy_name": str(chunk["strategy_name"]),
            "strategy_version": str(chunk["strategy_version"]),
            "chunk_index": int(chunk["chunk_index"]),
            "chunk_text": str(chunk["chunk_text"]),
            "chunk_hash": str(chunk["chunk_hash"]),
            "tokenizer_name": str(chunk.get("tokenizer", "")),
            "pii_masked_flag": bool(chunk.get("pii_masked_flag", False)),
        }
        for chunk in chunks
    ]


def _model_row(model: EmbeddingModel) -> dict[str, Any]:
    return {
        "model_name": model.payload_model_name,
        "model_version": model.model_version,
        "provider": model.provider,
        "dimensions": model.dimensions,
        "pricing_usd_per_1k_tokens": model.pricing_usd_per_1k_tokens,
        "latency_ms_p95": model.latency_ms_p95,
    }


def _vector_rows(
    chunks: Sequence[Mapping[str, Any]], model: EmbeddingModel, collection: str
) -> list[dict[str, Any]]:
    return [
        {
            "vector_id": f"vec_{index:04d}",
            "chunk_id": str(chunk["chunk_id"]),
            "model_name": model.payload_model_name,
            "model_version": model.model_version,
            "collection_name": collection,
            "dimension": model.dimensions,
            "distance_metric": "Cosine",
            "active": True,
            "pii_masked_flag": bool(chunk.get("pii_masked_flag", False)),
        }
        for index, chunk in enumerate(chunks, start=1)
    ]


def materialize_catalog(
    catalog: CatalogDB,
    documents: Iterable[Mapping[str, Any]],
    chunks: Sequence[Mapping[str, Any]],
    model: EmbeddingModel = BASELINE_MODEL_V1,
    collection: str = DEFAULT_COLLECTION_NAME,
) -> None:
    """Persist governed document, chunk, strategy, model, and vector metadata."""
    strategy = ChunkingStrategy.from_yaml(DEFAULT_STRATEGY_PATH)
    catalog.insert_documents(_document_rows(documents))
    catalog.insert_chunk_strategies(
        [{
            "strategy_name": strategy.strategy_name,
            "strategy_version": strategy.version,
            "splitter_type": strategy.splitter_type,
            "chunk_size_tokens": strategy.chunk_size_tokens,
            "chunk_overlap_tokens": strategy.chunk_overlap_tokens,
            "tokenizer_name": strategy.tokenizer,
            "pii_masked_flag": True,
        }]
    )
    catalog.insert_chunks(_chunk_rows(chunks))
    catalog.insert_embedding_models([_model_row(model)])
    catalog.insert_vectors(_vector_rows(chunks, model, collection))


def bootstrap_baseline(
    *,
    corpus_path: Path = DEFAULT_CORPUS_PATH,
    dataset_path: Path = DEFAULT_DATASET_PATH,
    quarantine_path: Path = DEFAULT_QUARANTINE_PATH,
    database_path: Path | str = DEFAULT_DATABASE_PATH,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    collection: str = DEFAULT_COLLECTION_NAME,
    embed: Callable[[Sequence[str]], Iterable[Sequence[float]]] | None = None,
    qdrant_client: Any | None = None,
) -> dict[str, Any]:
    """Run validation, Lance persistence, catalog materialization, and Qdrant ingest."""
    report = load_documents(corpus_path, dataset_path, quarantine_path)
    documents, invalid_records = load_valid_documents(corpus_path)
    chunks = read_chunks(dataset_path)
    model = BASELINE_MODEL_V1
    client = qdrant_client or QdrantClient(url=qdrant_url)
    owns_client = qdrant_client is None
    try:
        with CatalogDB(database_path) as catalog:
            materialize_catalog(catalog, documents, chunks, model, collection)
            inserted = ingest_chunks(
                chunks,
                client,
                model=model,
                embed=embed,
                collection_name=collection,
            )
            if inserted != len(chunks):
                raise RuntimeError(
                    f"Qdrant inserted {inserted} vectors, expected {len(chunks)}"
                )
    finally:
        if owns_client:
            client.close()
    return {
        **report,
        "invalid_records_reloaded": len(invalid_records),
        "catalog_document_count": len(documents),
        "catalog_chunk_count": len(chunks),
        "vector_count": len(chunks),
        "collection": collection,
        "model_name": model.payload_model_name,
        "model_version": model.model_version,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS_PATH)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--quarantine", type=Path, default=DEFAULT_QUARANTINE_PATH)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--collection", default=DEFAULT_COLLECTION_NAME)
    args = parser.parse_args()
    report = bootstrap_baseline(
        corpus_path=args.corpus,
        dataset_path=args.dataset,
        quarantine_path=args.quarantine,
        database_path=args.database,
        qdrant_url=args.qdrant_url,
        collection=args.collection,
    )
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if report["invalid_document_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()