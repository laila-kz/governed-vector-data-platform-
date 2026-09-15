"""Asynchronous-style batch worker for shadow embedding migrations."""

from __future__ import annotations

import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence
from uuid import NAMESPACE_URL, uuid5

from qdrant_client.models import Distance, PointStruct, VectorParams

from catalog.db import CatalogDB
from chunking.chunker import DEFAULT_DATASET_PATH, read_chunks
from embedding.embedder import VectorEmbedder
from embedding.model_registry import EmbeddingModel, get_model
from migration.planner import _resolve_model
from migration.progress_tracker import MigrationProgress, ProgressTracker

DEFAULT_SHADOW_COLLECTION = "scifact_v2_shadow"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_RETRIES = 3


class ShadowMigrationWorker:
    """Read a planned migration and write target vectors to a shadow collection."""

    def __init__(
        self,
        catalog: CatalogDB,
        qdrant_client: Any,
        *,
        dataset_path: Path | str = DEFAULT_DATASET_PATH,
        embed: VectorEmbedder | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_retries: int = DEFAULT_MAX_RETRIES,
        backoff_base_seconds: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
        random_value: Callable[[], float] = random.random,
        progress_tracker: ProgressTracker | None = None,
        collection_name: str = DEFAULT_SHADOW_COLLECTION,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if max_retries < 0:
            raise ValueError("max_retries must not be negative")
        if backoff_base_seconds < 0:
            raise ValueError("backoff_base_seconds must not be negative")
        self.catalog = catalog
        self.qdrant_client = qdrant_client
        self.dataset_path = Path(dataset_path)
        self.embed = embed
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.backoff_base_seconds = backoff_base_seconds
        self.sleep = sleep
        self.random_value = random_value
        self.progress_tracker = progress_tracker or ProgressTracker()
        self.collection_name = collection_name

    def run(self, migration_id: str) -> MigrationProgress:
        migration = self._load_migration(migration_id)
        source_model = _resolve_model(str(migration[0]))
        target_model = _resolve_model(str(migration[1]))
        source_version = str(migration[2])
        source_strategy = str(migration[3])
        total_vectors = int(migration[4])

        self._ensure_target_model(target_model)
        self._ensure_collection(target_model)
        chunk_ids = self._source_chunk_ids(
            source_model.payload_model_name,
            source_version,
            source_strategy,
        )
        chunks = self._load_chunks(chunk_ids)
        if len(chunks) != total_vectors:
            raise ValueError(
                f"Migration {migration_id} expects {total_vectors} chunks, "
                f"but Lance contains {len(chunks)} matching chunks"
            )

        self._set_migration_status(migration_id, "running")
        started = time.perf_counter()
        migrated = 0
        errors = 0
        completed_batches = 0
        try:
            for batch_start in range(0, len(chunks), self.batch_size):
                batch = chunks[batch_start : batch_start + self.batch_size]
                vectors = self._embed_batch(batch, target_model)
                points = self._points(batch, vectors, target_model, migration_id)
                errors += self._upsert_with_retries(points, migration_id, len(batch))
                self._insert_catalog_vectors(batch, points, target_model)
                migrated += len(batch)
                completed_batches += 1
                self._update_migration_progress(migration_id, migrated)
                self._checkpoint(
                    migration_id,
                    total_vectors,
                    migrated,
                    completed_batches,
                    errors,
                    started,
                )
        except Exception:
            self._set_migration_status(migration_id, "failed")
            raise

        self._set_migration_status(migration_id, "completed")
        return self.progress_tracker.checkpoints[-1] if self.progress_tracker.checkpoints else MigrationProgress(
            migration_id, total_vectors, 0, 0, errors, 0.0
        )

    def _load_migration(self, migration_id: str) -> tuple[Any, ...]:
        rows = self.catalog.query(
            """
            SELECT source_model_name, target_model_name, source_model_version,
                   source_strategy_name, total_vectors, status
            FROM migrations
            WHERE migration_id = ?
            """,
            [migration_id],
        )
        if not rows:
            raise ValueError(f"Unknown migration: {migration_id}")
        row = rows[0]
        if row[5] not in {"planned", "pending", "running"}:
            raise ValueError(f"Migration {migration_id} is not pending: {row[5]}")
        return row[:5]

    def _source_chunk_ids(self, model_name: str, model_version: str, strategy: str) -> list[str]:
        rows = self.catalog.query(
            """
            SELECT v.chunk_id
            FROM vectors AS v
            JOIN chunks AS c ON c.chunk_id = v.chunk_id
            WHERE v.active = TRUE AND v.model_name = ? AND v.model_version = ?
              AND c.strategy_name = ?
            ORDER BY c.chunk_id
            """,
            [model_name, model_version, strategy],
        )
        return [str(row[0]) for row in rows]

    def _load_chunks(self, chunk_ids: Iterable[str]) -> list[dict[str, Any]]:
        wanted = set(chunk_ids)
        return [chunk for chunk in read_chunks(self.dataset_path) if str(chunk["chunk_id"]) in wanted]

    def _ensure_target_model(self, model: EmbeddingModel) -> None:
        if not self.catalog.query(
            "SELECT 1 FROM embedding_models WHERE model_name = ? AND model_version = ?",
            [model.payload_model_name, model.model_version],
        ):
            self.catalog.insert_embedding_models(
                [
                    {
                        "model_name": model.payload_model_name,
                        "model_version": model.model_version,
                        "provider": model.provider,
                        "dimensions": model.dimensions,
                        "pricing_usd_per_1k_tokens": model.pricing_usd_per_1k_tokens,
                        "latency_ms_p95": model.latency_ms_p95,
                    }
                ]
            )

    def _ensure_collection(self, model: EmbeddingModel) -> None:
        if self.qdrant_client.collection_exists(self.collection_name):
            return
        self.qdrant_client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=model.dimensions, distance=Distance.COSINE),
        )

    def _embed_batch(
        self, batch: Sequence[Mapping[str, Any]], model: EmbeddingModel
    ) -> list[Sequence[float]]:
        embed = self.embed
        if embed is None:
            from embedding.embedder import fastembedder

            embed = fastembedder(model)
        vectors = list(embed([str(chunk["chunk_text"]) for chunk in batch]))
        if len(vectors) != len(batch):
            raise ValueError("Embedding provider returned a different number of vectors")
        for index, vector in enumerate(vectors):
            if len(vector) != model.dimensions:
                raise ValueError(
                    f"Embedding at batch offset {index} has dimension {len(vector)}; "
                    f"expected {model.dimensions}"
                )
        return vectors

    def _points(
        self,
        batch: Sequence[Mapping[str, Any]],
        vectors: Sequence[Sequence[float]],
        model: EmbeddingModel,
        migration_id: str,
    ) -> list[PointStruct]:
        created_at = datetime.now(timezone.utc).isoformat()
        points: list[PointStruct] = []
        for chunk, vector in zip(batch, vectors):
            vector_id = f"{migration_id}:{chunk['chunk_id']}"
            points.append(
                PointStruct(
                    id=str(uuid5(NAMESPACE_URL, vector_id)),
                    vector=list(vector),
                    payload={
                        "vector_id": vector_id,
                        "chunk_id": str(chunk["chunk_id"]),
                        "doc_id": str(chunk["doc_id"]),
                        "doc_version": int(chunk.get("doc_version", 1)),
                        "strategy_name": str(chunk["strategy_name"]),
                        "strategy_version": str(chunk["strategy_version"]),
                        "model_name": model.payload_model_name,
                        "model_version": model.model_version,
                        "is_active": False,
                        "pii_masked_flag": True,
                        "created_at": created_at,
                    },
                )
            )
        return points

    def _upsert_with_retries(
        self, points: Sequence[PointStruct], migration_id: str, batch_size: int
    ) -> int:
        errors = 0
        for attempt in range(self.max_retries + 1):
            try:
                self.qdrant_client.upsert(
                    collection_name=self.collection_name, points=list(points), wait=True
                )
                return errors
            except Exception:
                errors += 1
                if attempt == self.max_retries:
                    raise
                delay = self.backoff_base_seconds * (2**attempt)
                self.sleep(delay * (0.5 + self.random_value()))
        return errors

    def _insert_catalog_vectors(
        self,
        chunks: Sequence[Mapping[str, Any]],
        points: Sequence[PointStruct],
        model: EmbeddingModel,
    ) -> None:
        self.catalog.insert_vectors(
            [
                {
                    "vector_id": str(point.payload["vector_id"]),
                    "chunk_id": str(chunk["chunk_id"]),
                    "model_name": model.payload_model_name,
                    "model_version": model.model_version,
                    "collection_name": self.collection_name,
                    "dimension": model.dimensions,
                    "active": False,
                    "pii_masked_flag": True,
                }
                for chunk, point in zip(chunks, points)
            ]
        )

    def _update_migration_progress(self, migration_id: str, migrated: int) -> None:
        self.catalog.connection.execute(
            "UPDATE migrations SET migrated_vectors = ? WHERE migration_id = ?",
            [migrated, migration_id],
        )

    def _set_migration_status(self, migration_id: str, status: str) -> None:
        self.catalog.connection.execute(
            "UPDATE migrations SET status = ? WHERE migration_id = ?",
            [status, migration_id],
        )

    def _checkpoint(
        self,
        migration_id: str,
        total: int,
        migrated: int,
        batches: int,
        errors: int,
        started: float,
    ) -> None:
        elapsed = max(time.perf_counter() - started, 1e-9)
        self.progress_tracker.checkpoint(
            MigrationProgress(
                migration_id,
                total,
                migrated,
                batches,
                errors,
                migrated / elapsed,
            )
        )