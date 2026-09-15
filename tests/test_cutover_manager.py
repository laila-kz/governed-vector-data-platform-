from pathlib import Path
from typing import Any

import yaml

from catalog.db import CatalogDB
from chunking.chunker import write_chunks
from migration.cutover_manager import CutoverManager
from migration.planner import MigrationPlanner
from migration.worker import ShadowMigrationWorker
from tests.test_shadow_migration import _seed_catalog


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.alias_operations: list[list[Any]] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, **kwargs: Any) -> None:
        self.collections.add(str(kwargs["collection_name"]))

    def upsert(self, **kwargs: Any) -> None:
        return None

    def update_collection_aliases(self, **kwargs: Any) -> None:
        self.alias_operations.append(kwargs["change_aliases_operations"])


def _prepare_migration(tmp_path: Path) -> tuple[CatalogDB, FakeQdrant, str, Path]:
    dataset_path = tmp_path / "chunks.lance"
    write_chunks(
        [{
            "chunk_id": "doc-1:0000", "doc_id": "doc-1", "doc_version": 1,
            "chunk_index": 0, "chunk_text": "one", "chunk_hash": "hash",
            "strategy_name": "fixed_size_v1", "strategy_version": "1.0.0",
        }],
        dataset_path,
    )
    catalog = CatalogDB(":memory:")
    _seed_catalog(catalog)
    plan = MigrationPlanner(catalog).plan_migration(
        "v1", "v2", "fixed_size_v1", "fixed_size_v1"
    )
    ShadowMigrationWorker(
        catalog,
        FakeQdrant(),
        dataset_path=dataset_path,
        embed=lambda texts: [[0.0] * 1024 for _ in texts],
    ).run(plan.migration_id)
    policy_path = tmp_path / "routing_policy.yaml"
    policy_path.write_text(
        "active_alias: vectors_live\nactive_model: bge-small-en-v1.5\nactive_version: 1.0.0\n",
        encoding="utf-8",
    )
    return catalog, FakeQdrant(), plan.migration_id, policy_path


def test_execute_cutover_atomically_updates_alias_catalog_and_policy(tmp_path: Path) -> None:
    catalog, qdrant, migration_id, policy_path = _prepare_migration(tmp_path)
    try:
        manager = CutoverManager(catalog, qdrant, policy_path=policy_path)
        assert manager.verify_migration_completeness(migration_id) is True
        manager.execute_cutover(migration_id)

        assert len(qdrant.alias_operations) == 1
        operations = qdrant.alias_operations[0]
        assert operations[0].delete_alias.alias_name == "vectors_live"
        assert operations[1].create_alias.collection_name == "scifact_v2_shadow"
        assert catalog.query(
            "SELECT active FROM vectors WHERE vector_id = 'vec-1'"
        ) == [(False,)]
        assert catalog.query(
            "SELECT active FROM vectors WHERE model_version = '2.0.0'"
        ) == [(True,)]
        assert catalog.query(
            "SELECT status FROM migrations WHERE migration_id = ?", [migration_id]
        ) == [("completed",)]
        assert yaml.safe_load(policy_path.read_text(encoding="utf-8"))["active_model"] == (
            "bge-large-en-v1.5"
        )
    finally:
        catalog.close()


def test_rollback_restores_baseline_and_records_event(tmp_path: Path) -> None:
    catalog, qdrant, migration_id, policy_path = _prepare_migration(tmp_path)
    try:
        manager = CutoverManager(catalog, qdrant, policy_path=policy_path)
        manager.execute_cutover(migration_id)
        manager.rollback_cutover(migration_id)

        assert len(qdrant.alias_operations) == 2
        assert qdrant.alias_operations[1][1].create_alias.collection_name == "scifact_v1"
        assert catalog.query(
            "SELECT active FROM vectors WHERE vector_id = 'vec-1'"
        ) == [(True,)]
        assert catalog.query(
            "SELECT active FROM vectors WHERE model_version = '2.0.0'"
        ) == [(False,)]
        assert catalog.query(
            "SELECT status FROM migrations WHERE migration_id = ?", [migration_id]
        ) == [("rolled_back",)]
        assert catalog.query(
            "SELECT event_type FROM migration_events WHERE migration_id = ? ORDER BY created_at",
            [migration_id],
        ) == [("cutover_completed",), ("cutover_rolled_back",)]
        assert yaml.safe_load(policy_path.read_text(encoding="utf-8"))["active_model"] == (
            "bge-small-en-v1.5"
        )
    finally:
        catalog.close()


def test_verify_cutover_rejects_incomplete_migration(tmp_path: Path) -> None:
    catalog, qdrant, migration_id, policy_path = _prepare_migration(tmp_path)
    try:
        catalog.connection.execute(
            "UPDATE migrations SET migrated_vectors = 0 WHERE migration_id = ?",
            [migration_id],
        )
        manager = CutoverManager(catalog, qdrant, policy_path=policy_path)
        try:
            manager.verify_migration_completeness(migration_id)
        except ValueError as error:
            assert "incomplete" in str(error)
        else:
            raise AssertionError("incomplete migration should not pass verification")
    finally:
        catalog.close()