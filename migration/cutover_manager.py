"""Atomic Qdrant alias cutovers for completed shadow migrations."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import yaml
from qdrant_client.http import models

from catalog.db import CatalogDB

DEFAULT_ALIAS = "vectors_live"
DEFAULT_SOURCE_COLLECTION = "scifact_v1"
DEFAULT_SHADOW_COLLECTION = "scifact_v2_shadow"
DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "configs" / "routing_policy.yaml"


class CutoverManager:
    """Coordinate catalog, routing policy, and Qdrant cutover state."""

    def __init__(
        self,
        catalog: CatalogDB,
        qdrant_client: Any,
        *,
        policy_path: Path | str = DEFAULT_POLICY_PATH,
        alias_name: str = DEFAULT_ALIAS,
        source_collection: str = DEFAULT_SOURCE_COLLECTION,
        shadow_collection: str = DEFAULT_SHADOW_COLLECTION,
    ) -> None:
        self.catalog = catalog
        self.qdrant_client = qdrant_client
        self.policy_path = Path(policy_path)
        self.alias_name = alias_name
        self.source_collection = source_collection
        self.shadow_collection = shadow_collection

    def _migration(self, migration_id: str) -> tuple[Any, ...]:
        rows = self.catalog.query(
            """
            SELECT source_model_name, source_model_version, target_model_name,
                   target_model_version, total_vectors, migrated_vectors,
                   status, error_count
            FROM migrations WHERE migration_id = ?
            """,
            [migration_id],
        )
        if not rows:
            raise ValueError(f"Unknown migration: {migration_id}")
        return rows[0]

    def verify_migration_completeness(self, migration_id: str) -> bool:
        """Assert that every planned vector migrated without recorded errors."""
        migration = self._migration(migration_id)
        total_vectors = int(migration[4])
        migrated_vectors = int(migration[5])
        error_count = int(migration[7] or 0)
        if migrated_vectors != total_vectors:
            raise ValueError(
                f"Migration {migration_id} is incomplete: "
                f"migrated {migrated_vectors} of {total_vectors} vectors"
            )
        if error_count != 0:
            raise ValueError(
                f"Migration {migration_id} has {error_count} recorded errors"
            )
        if migration[6] != "completed":
            raise ValueError(f"Migration {migration_id} is not cutover-ready: status is {migration[6]}")
        return True

    def _swap_alias(self, collection_name: str) -> None:
        if not self.qdrant_client.collection_exists(collection_name):
            raise ValueError(f"Qdrant collection does not exist: {collection_name}")
        self.qdrant_client.update_collection_aliases(
            change_aliases_operations=[
                models.DeleteAliasOperation(
                    delete_alias=models.DeleteAlias(alias_name=self.alias_name)
                ),
                models.CreateAliasOperation(
                    create_alias=models.CreateAlias(
                        collection_name=collection_name,
                        alias_name=self.alias_name,
                    )
                ),
            ]
        )

    def _update_policy(self, model_name: str, model_version: str, *, shadow: bool = False) -> None:
        policy = yaml.safe_load(self.policy_path.read_text(encoding="utf-8")) or {}
        policy.update(
            {
                "active_alias": self.alias_name,
                "active_model": model_name,
                "active_version": model_version,
                "shadow_collection": self.shadow_collection if shadow else None,
                "shadow_model": model_name if shadow else None,
                "shadow_version": model_version if shadow else None,
            }
        )
        self.policy_path.write_text(
            yaml.safe_dump(policy, sort_keys=False), encoding="utf-8"
        )

    def _record_event(self, migration_id: str, event_type: str, details: dict[str, Any]) -> None:
        self.catalog.insert_rows(
            "migration_events",
            [{
                "event_id": str(uuid.uuid4()),
                "migration_id": migration_id,
                "event_type": event_type,
                "details_json": json.dumps(details),
            }],
        )

    def execute_cutover(self, migration_id: str) -> None:
        """Atomically route the live alias to the completed shadow collection."""
        self.verify_migration_completeness(migration_id)
        migration = self._migration(migration_id)
        source_model, source_version, target_model, target_version = migration[:4]
        self._swap_alias(self.shadow_collection)
        try:
            with self.catalog.transaction():
                self.catalog.connection.execute(
                    "UPDATE vectors SET active = FALSE WHERE active = TRUE AND model_name = ? AND model_version = ?",
                    [source_model, source_version],
                )
                self.catalog.connection.execute(
                    "UPDATE vectors SET active = TRUE WHERE collection_name = ? AND model_name = ? AND model_version = ?",
                    [self.shadow_collection, target_model, target_version],
                )
                self.catalog.connection.execute(
                    "UPDATE migrations SET status = 'completed', completed_at = current_timestamp WHERE migration_id = ?",
                    [migration_id],
                )
                self._record_event(
                    migration_id,
                    "cutover_completed",
                    {"alias": self.alias_name, "collection": self.shadow_collection},
                )
            self._update_policy(str(target_model), str(target_version))
        except Exception:
            self._swap_alias(self.source_collection)
            raise

    def rollback_cutover(self, migration_id: str) -> None:
        """Restore the baseline collection and record the rollback event."""
        migration = self._migration(migration_id)
        source_model, source_version, target_model, target_version = migration[:4]
        self._swap_alias(self.source_collection)
        try:
            with self.catalog.transaction():
                self.catalog.connection.execute(
                    "UPDATE vectors SET active = FALSE WHERE model_name = ? AND model_version = ?",
                    [target_model, target_version],
                )
                self.catalog.connection.execute(
                    "UPDATE vectors SET active = TRUE WHERE collection_name = ? AND model_name = ? AND model_version = ?",
                    [self.source_collection, source_model, source_version],
                )
                self.catalog.connection.execute(
                    "UPDATE migrations SET status = 'rolled_back' WHERE migration_id = ?",
                    [migration_id],
                )
                self._record_event(
                    migration_id,
                    "cutover_rolled_back",
                    {"alias": self.alias_name, "collection": self.source_collection},
                )
            self._update_policy(str(source_model), str(source_version))
        except Exception:
            self._swap_alias(self.shadow_collection)
            raise