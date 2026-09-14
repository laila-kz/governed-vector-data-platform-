"""Batch embed Lance chunks and ingest vectors into Qdrant."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5
from typing import TYPE_CHECKING

from qdrant_client import QdrantClient
from qdrant_client.models import (
    CollectionInfo,
    CreateAlias,
    CreateAliasOperation,
    DeleteAlias,
    DeleteAliasOperation,
    Distance,
    PointStruct,
    VectorParams,
)

from chunking.chunker import DEFAULT_DATASET_PATH, read_chunks
from embedding.model_registry import BASELINE_MODEL_V1, EmbeddingModel, get_model

if TYPE_CHECKING:
    from catalog.openlineage_emitter import OpenLineageEmitter

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_COLLECTION_NAME = "scifact_v1"
DEFAULT_ALIAS_NAME = "vectors_live"
DEFAULT_BATCH_SIZE = 64

VectorEmbedder = Callable[[Sequence[str]], Iterable[Sequence[float]]]


def fastembedder(model: EmbeddingModel = BASELINE_MODEL_V1) -> VectorEmbedder:
    """Create the production FastEmbed callable without loading it at import time."""
    from fastembed import TextEmbedding

    embedder = TextEmbedding(model_name=model.model_name)

    def embed(texts: Sequence[str]) -> Iterable[Sequence[float]]:
        return embedder.embed(list(texts))

    return embed


def _ensure_collection(
    client: QdrantClient, collection_name: str, model: EmbeddingModel
) -> None:
    if client.collection_exists(collection_name):
        info: CollectionInfo = client.get_collection(collection_name)
        vector_config = info.config.params.vectors
        if isinstance(vector_config, dict):
            vector_config = next(iter(vector_config.values()))
        if vector_config.size != model.dimensions or vector_config.distance != Distance.COSINE:
            raise ValueError(
                f"Collection {collection_name!r} does not match the model contract: "
                f"expected {model.dimensions} dimensions and Cosine distance"
            )
        return
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=model.dimensions, distance=Distance.COSINE),
    )


def _ensure_alias(client: QdrantClient, alias_name: str, collection_name: str) -> None:
    aliases = client.get_aliases().aliases
    current = next((alias for alias in aliases if alias.alias_name == alias_name), None)
    if current is None:
        client.update_collection_aliases(
            change_aliases_operations=[
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        alias_name=alias_name, collection_name=collection_name
                    )
                )
            ]
        )
    elif current.collection_name != collection_name:
        client.update_collection_aliases(
            change_aliases_operations=[
                DeleteAliasOperation(delete_alias=DeleteAlias(alias_name=alias_name)),
                CreateAliasOperation(
                    create_alias=CreateAlias(
                        alias_name=alias_name, collection_name=collection_name
                    )
                ),
            ]
        )


def _vector_id(index: int) -> str:
    return f"vec_{index:04d}"


def _payload(
    chunk: Mapping[str, Any], model: EmbeddingModel, vector_id: str, created_at: str
) -> dict[str, Any]:
    return {
        "vector_id": vector_id,
        "chunk_id": str(chunk["chunk_id"]),
        "doc_id": str(chunk["doc_id"]),
        "doc_version": int(chunk.get("doc_version", 1)),
        "strategy_name": str(chunk["strategy_name"]),
        "strategy_version": str(chunk["strategy_version"]),
        "model_name": model.payload_model_name,
        "model_version": model.model_version,
        "pii_masked_flag": bool(chunk.get("pii_masked_flag", False)),
        "created_at": created_at,
    }


def ingest_chunks(
    chunks: Iterable[Mapping[str, Any]],
    client: QdrantClient,
    model: EmbeddingModel = BASELINE_MODEL_V1,
    embed: VectorEmbedder | None = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    alias_name: str = DEFAULT_ALIAS_NAME,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lineage_emitter: "OpenLineageEmitter | None" = None,
) -> int:
    """Embed chunks in batches, upsert points, and ensure the live alias."""
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    chunk_list = list(chunks)
    if not chunk_list:
        return 0

    embed = embed or fastembedder(model)
    _ensure_collection(client, collection_name, model)
    created_at = datetime.now(timezone.utc).isoformat()
    run_id = str(uuid5(NAMESPACE_URL, f"{collection_name}:{created_at}"))
    if lineage_emitter is not None:
        lineage_emitter.emit_start(run_id, collection_name)
    inserted = 0
    try:
        for batch_start in range(0, len(chunk_list), batch_size):
            batch = chunk_list[batch_start : batch_start + batch_size]
            vectors = list(embed([str(chunk["chunk_text"]) for chunk in batch]))
            if len(vectors) != len(batch):
                raise ValueError("Embedding provider returned a different number of vectors")
            points: list[PointStruct] = []
            for offset, (chunk, vector) in enumerate(zip(batch, vectors)):
                vector_id = _vector_id(batch_start + offset + 1)
                vector_values = list(vector)
                if len(vector_values) != model.dimensions:
                    raise ValueError(
                        f"Embedding for {vector_id} has dimension {len(vector_values)}; "
                        f"expected {model.dimensions}"
                    )
                points.append(
                    PointStruct(
                        id=str(uuid5(NAMESPACE_URL, vector_id)),
                        vector=vector_values,
                        payload=_payload(chunk, model, vector_id, created_at),
                    )
                )
            client.upsert(collection_name=collection_name, points=points, wait=True)
            inserted += len(points)
    except Exception:
        raise

    _ensure_alias(client, alias_name, collection_name)
    if lineage_emitter is not None:
        lineage_emitter.emit_complete(run_id, collection_name)
    return inserted


def ingest_lance_dataset(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    model_key: str = "v1",
    lineage_emitter: "OpenLineageEmitter | None" = None,
    **kwargs: Any,
) -> int:
    """Read chunks from Lance and ingest them using a Qdrant HTTP client."""
    model = get_model(model_key)
    client = QdrantClient(url=qdrant_url)
    return ingest_chunks(
        read_chunks(dataset_path),
        client,
        model=model,
        lineage_emitter=lineage_emitter,
        **kwargs,
    )


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    parser.add_argument("--model", default="v1")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--marquez-url", help="Marquez OpenLineage endpoint")
    args = parser.parse_args()
    lineage_emitter = None
    if args.marquez_url:
        from catalog.openlineage_emitter import OpenLineageEmitter, VectorEmbeddingDatasetFacet

        lineage_emitter = OpenLineageEmitter.for_marquez(
            VectorEmbeddingDatasetFacet(
                model_name="bge-small-en-v1.5",
                model_version="1.0.0",
                dimension=384,
                strategy_name="fixed_size_v1",
                strategy_version="1.0.0",
                chunk_size_tokens=300,
            ),
            url=args.marquez_url,
        )
    count = ingest_lance_dataset(
        args.dataset,
        args.qdrant_url,
        args.model,
        batch_size=args.batch_size,
        lineage_emitter=lineage_emitter,
    )
    print(f"ingested {count} vectors into {DEFAULT_COLLECTION_NAME}")


if __name__ == "__main__":
    main()