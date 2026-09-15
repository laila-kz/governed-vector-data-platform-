"""Chaos simulation for shadow migrations."""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from catalog.db import CatalogDB
from chunking.chunker import read_chunks
from migration.worker import CircuitBreakerOpenError, ShadowMigrationWorker


class ChaosRateLimitError(RuntimeError):
    """Simulate an upstream HTTP 429 response."""

    status_code = 429


class ChaosMalformedPayloadError(ValueError):
    """Simulate an upstream response with an invalid payload."""


@dataclass
class ChaosTelemetry:
    rate_limit_failures: int = 0
    dropped_connections: int = 0
    malformed_payloads: int = 0
    backoffs: list[float] = field(default_factory=list)
    circuit_breaker_opened: bool = False


class ChaosFaultInjector:
    """Probabilistically inject one of the migration's upstream failure modes."""

    def __init__(
        self,
        fail_rate: float,
        *,
        random_value: Callable[[], float] = random.random,
        fault_selector: Callable[[], float] = random.random,
        telemetry: ChaosTelemetry | None = None,
    ) -> None:
        if not 0.0 <= fail_rate <= 1.0:
            raise ValueError("fail_rate must be between 0.0 and 1.0")
        self.fail_rate = fail_rate
        self.random_value = random_value
        self.fault_selector = fault_selector
        self.telemetry = telemetry or ChaosTelemetry()

    def before_upsert(self, batch_size: int, attempt: int) -> None:
        if self.random_value() >= self.fail_rate:
            return
        fault = self.fault_selector()
        if fault < 1 / 3:
            self.telemetry.rate_limit_failures += 1
            raise ChaosRateLimitError("simulated upstream rate limit (HTTP 429)")
        if fault < 2 / 3:
            self.telemetry.dropped_connections += 1
            raise ConnectionError("simulated dropped batch connection")
        self.telemetry.malformed_payloads += 1
        raise ChaosMalformedPayloadError("simulated malformed embedding payload")


@dataclass(frozen=True)
class ChaosReport:
    migration_id: str
    fail_rate: float
    migrated_vectors: int
    circuit_breaker_opened: bool
    telemetry: ChaosTelemetry
    active_catalog_unchanged: bool
    lance_unchanged: bool


def _lance_fingerprint(dataset_path: Path | str) -> str:
    digest = hashlib.sha256()
    for chunk in read_chunks(dataset_path):
        digest.update(repr(sorted(chunk.items())).encode("utf-8"))
    return digest.hexdigest()


def run_chaos_injection(
    catalog: CatalogDB,
    qdrant_client: Any,
    migration_id: str,
    fail_rate: float,
    *,
    dataset_path: Path | str = "data/lance_lakehouse/chunks.lance",
    random_value: Callable[[], float] = random.random,
    fault_selector: Callable[[], float] = random.random,
    sleep: Callable[[float], None] = lambda _: None,
    embed: Callable[[list[str]], list[list[float]]] | None = None,
) -> ChaosReport:
    """Run a fault-injected shadow migration and verify source integrity."""
    active_before = catalog.query(
        "SELECT vector_id, chunk_id, model_name, model_version, active FROM vectors "
        "WHERE active = TRUE ORDER BY vector_id"
    )
    lance_before = _lance_fingerprint(dataset_path)
    telemetry = ChaosTelemetry()
    injector = ChaosFaultInjector(
        fail_rate,
        random_value=random_value,
        fault_selector=fault_selector,
        telemetry=telemetry,
    )
    worker = ShadowMigrationWorker(
        catalog,
        qdrant_client,
        dataset_path=dataset_path,
        embed=embed,
        fault_injector=injector,
        sleep=sleep,
        on_backoff=lambda delay, _count, _error: telemetry.backoffs.append(delay),
    )
    circuit_opened = False
    migrated_vectors = 0
    try:
        result = worker.run(migration_id)
        migrated_vectors = result.migrated_vectors
    except CircuitBreakerOpenError:
        circuit_opened = True
        telemetry.circuit_breaker_opened = True
    active_after = catalog.query(
        "SELECT vector_id, chunk_id, model_name, model_version, active FROM vectors "
        "WHERE active = TRUE ORDER BY vector_id"
    )
    return ChaosReport(
        migration_id,
        fail_rate,
        migrated_vectors,
        circuit_opened,
        telemetry,
        active_before == active_after,
        lance_before == _lance_fingerprint(dataset_path),
    )


def chaos_inject(
    fail_rate: float = 0.25,
    migration_id: str = "",
    qdrant_url: str = "http://localhost:6333",
    dataset_path: str = "data/lance_lakehouse/chunks.lance",
) -> None:
    """Inject upstream failures into a shadow migration and report integrity checks."""
    if not migration_id:
        raise ValueError("migration_id is required")
    from qdrant_client import QdrantClient

    with CatalogDB() as catalog, QdrantClient(url=qdrant_url) as qdrant:
        report = run_chaos_injection(
            catalog, qdrant, migration_id, fail_rate, dataset_path=dataset_path
        )
    print(
        f"Chaos migration {report.migration_id}: "
        f"circuit_breaker={report.circuit_breaker_opened}, "
        f"backoffs={len(report.telemetry.backoffs)}, "
        f"active_index_unchanged={report.active_catalog_unchanged}, "
        f"lance_unchanged={report.lance_unchanged}"
    )