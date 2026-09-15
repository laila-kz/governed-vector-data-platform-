from pathlib import Path

from catalog.db import CatalogDB
from chunking.chunker import write_chunks
from cli.chaos import (
    ChaosFaultInjector,
    ChaosMalformedPayloadError,
    ChaosRateLimitError,
    ChaosTelemetry,
    run_chaos_injection,
)
from migration.planner import MigrationPlanner
from tests.test_shadow_migration import _seed_catalog


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.upserts = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, **kwargs: object) -> None:
        self.collections.add(str(kwargs["collection_name"]))

    def upsert(self, **kwargs: object) -> None:
        self.upserts.append(kwargs)


def test_chaos_injector_supports_all_failure_modes() -> None:
    for selector, expected in (
        (0.0, ChaosRateLimitError),
        (0.5, ConnectionError),
        (0.9, ChaosMalformedPayloadError),
    ):
        try:
            ChaosFaultInjector(
                1.0,
                random_value=lambda: 0.0,
                fault_selector=lambda selector=selector: selector,
            ).before_upsert(1, 0)
        except expected:
            pass
        else:
            raise AssertionError(f"expected {expected.__name__}")


def test_chaos_run_opens_circuit_and_preserves_sources(tmp_path: Path) -> None:
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
        report = run_chaos_injection(
            catalog,
            FakeQdrant(),
            plan.migration_id,
            1.0,
            dataset_path=dataset_path,
            random_value=lambda: 0.0,
            fault_selector=lambda: 0.0,
            embed=lambda texts: [[0.0] * 1024 for _ in texts],
        )

        assert report.circuit_breaker_opened is True
        assert report.telemetry.rate_limit_failures == 4
        assert report.telemetry.backoffs
        assert report.active_catalog_unchanged is True
        assert report.lance_unchanged is True
        assert catalog.query(
            "SELECT status, migrated_vectors FROM migrations WHERE migration_id = ?",
            [plan.migration_id],
        ) == [("failed", 0)]