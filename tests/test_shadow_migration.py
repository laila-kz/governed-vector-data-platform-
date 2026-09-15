from pathlib import Path
from typing import Any

from catalog.db import CatalogDB
from chunking.chunker import write_chunks
from migration.planner import MigrationPlanner
from migration.progress_tracker import ProgressTracker
from migration.worker import ShadowMigrationWorker


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.upserts: list[dict[str, Any]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, **kwargs: Any) -> None:
        self.collections.add(kwargs["collection_name"])

    def upsert(self, **kwargs: Any) -> None:
        self.upserts.append(kwargs)


def _seed_catalog(catalog: CatalogDB) -> None:
    catalog.insert_documents(
        [{
            "doc_id": "doc-1", "doc_version": 1, "title": "Title",
            "source_uri": "test://doc-1", "author": "Test",
            "classification_level": "public", "language": "en",
        }]
    )
    catalog.insert_chunk_strategies(
        [{
            "strategy_name": "fixed_size_v1", "strategy_version": "1.0.0",
            "splitter_type": "recursive_character", "chunk_size_tokens": 300,
            "chunk_overlap_tokens": 30, "tokenizer_name": "cl100k_base",
        }]
    )
    catalog.insert_chunks(
        [{
            "chunk_id": "doc-1:0000", "doc_id": "doc-1", "doc_version": 1,
            "strategy_name": "fixed_size_v1", "strategy_version": "1.0.0",
            "chunk_index": 0, "chunk_text": "one", "chunk_hash": "hash",
        }]
    )
    catalog.insert_embedding_models(
        [{
            "model_name": "bge-small-en-v1.5", "model_version": "1.0.0",
            "provider": "fastembed", "dimensions": 384,
        }]
    )
    catalog.insert_vectors(
        [{
            "vector_id": "vec-1", "chunk_id": "doc-1:0000",
            "model_name": "bge-small-en-v1.5", "model_version": "1.0.0",
            "collection_name": "scifact_v1", "dimension": 384,
            "active": True,
        }]
    )


def test_shadow_worker_writes_inactive_target_vectors_and_checkpoints(tmp_path: Path) -> None:
    dataset_path = tmp_path / "chunks.lance"
    write_chunks(
        [{
            "chunk_id": "doc-1:0000", "doc_id": "doc-1", "doc_version": 1,
            "chunk_index": 0, "chunk_text": "one", "chunk_hash": "hash",
            "strategy_name": "fixed_size_v1", "strategy_version": "1.0.0",
            "pii_masked_flag": False,
        }],
        dataset_path,
    )
    with CatalogDB(":memory:") as catalog:
        _seed_catalog(catalog)
        plan = MigrationPlanner(catalog, batch_size=1).plan_migration(
            "v1", "v2", "fixed_size_v1", "fixed_size_v1"
        )
        qdrant = FakeQdrant()
        tracker = ProgressTracker()
        result = ShadowMigrationWorker(
            catalog,
            qdrant,
            dataset_path=dataset_path,
            embed=lambda texts: [[0.0] * 1024 for _ in texts],
            batch_size=1,
            progress_tracker=tracker,
        ).run(plan.migration_id)

        assert qdrant.collections == {"scifact_v2_shadow"}
        assert result.migrated_vectors == 1
        point = qdrant.upserts[0]["points"][0]
        assert point.payload["model_version"] == "2.0.0"
        assert point.payload["is_active"] is False
        assert point.payload["pii_masked_flag"] is True
        assert catalog.query(
            "SELECT status, migrated_vectors FROM migrations WHERE migration_id = ?",
            [plan.migration_id],
        ) == [("completed", 1)]
        assert catalog.query(
            "SELECT active, pii_masked_flag FROM vectors WHERE model_version = '2.0.0'"
        ) == [(False, True)]
        assert len(tracker.checkpoints) == 1


def test_shadow_worker_retries_transient_upsert_failures(tmp_path: Path) -> None:
    dataset_path = tmp_path / "chunks.lance"
    write_chunks(
        [{
            "chunk_id": "doc-1:0000", "doc_id": "doc-1", "doc_version": 1,
            "chunk_index": 0, "chunk_text": "one", "chunk_hash": "hash",
            "strategy_name": "fixed_size_v1", "strategy_version": "1.0.0",
        }],
        dataset_path,
    )
    with CatalogDB(":memory:") as catalog:
        _seed_catalog(catalog)
        plan = MigrationPlanner(catalog).plan_migration(
            "v1", "v2", "fixed_size_v1", "fixed_size_v1"
        )
        qdrant = FakeQdrant()
        original_upsert = qdrant.upsert
        attempts = 0

        def flaky_upsert(**kwargs: Any) -> None:
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise ConnectionError("temporary")
            original_upsert(**kwargs)

        qdrant.upsert = flaky_upsert
        result = ShadowMigrationWorker(
            catalog,
            qdrant,
            dataset_path=dataset_path,
            embed=lambda texts: [[0.0] * 1024 for _ in texts],
            sleep=lambda _: None,
            random_value=lambda: 0.0,
        ).run(plan.migration_id)

        assert result.error_count == 1
        assert attempts == 2