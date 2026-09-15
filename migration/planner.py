"""Pre-flight planning for embedding model and chunk-strategy migrations."""

from __future__ import annotations

import math
import uuid
from dataclasses import dataclass
from typing import Any

from catalog.db import CatalogDB
from embedding.model_registry import EmbeddingModel, get_model


@dataclass(frozen=True)
class MigrationPlan:
    """Estimated scope and cost of a planned migration."""

    migration_id: str
    from_model: str
    to_model: str
    from_model_version: str
    to_model_version: str
    from_strategy: str
    to_strategy: str
    total_vectors: int
    total_characters: int
    estimated_tokens: int
    estimated_cost_usd: float
    estimated_duration_seconds: float
    predicted_retrieval_drift: float
    batch_size: int
    rate_limit_per_second: float
    status: str = "planned"


def _resolve_model(model: str) -> EmbeddingModel:
    try:
        return get_model(model)
    except KeyError:
        for candidate in (get_model("v1"), get_model("v2")):
            if model in {candidate.model_name, candidate.payload_model_name}:
                return candidate
        raise ValueError(f"Unknown embedding model: {model}") from None


def _estimate_tokens(text: str) -> int:
    try:
        import tiktoken

        return max(1, len(tiktoken.get_encoding("cl100k_base").encode(text)))
    except Exception:
        return max(1, math.ceil(len(text.split()) * 1.3))


class MigrationPlanner:
    """Build and persist migration plans from the metadata catalog."""

    def __init__(
        self,
        catalog: CatalogDB,
        *,
        batch_size: int = 64,
        rate_limit_per_second: float = 10.0,
        predicted_retrieval_drift: float = 0.0,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        if rate_limit_per_second <= 0:
            raise ValueError("rate_limit_per_second must be greater than zero")
        self.catalog = catalog
        self.batch_size = batch_size
        self.rate_limit_per_second = rate_limit_per_second
        self.predicted_retrieval_drift = predicted_retrieval_drift

    def plan_migration(
        self,
        from_model: str,
        to_model: str,
        from_strategy: str,
        to_strategy: str,
    ) -> MigrationPlan:
        source = _resolve_model(from_model)
        target = _resolve_model(to_model)
        self._ensure_model_registered(source)
        self._ensure_model_registered(target)
        rows = self.catalog.query(
            """
            SELECT c.chunk_text
            FROM vectors AS v
            JOIN chunks AS c ON c.chunk_id = v.chunk_id
            WHERE v.active = TRUE
              AND v.model_name = ?
              AND v.model_version = ?
              AND c.strategy_name = ?
            ORDER BY c.chunk_id
            """,
            [source.payload_model_name, source.model_version, from_strategy],
        )
        total_characters = sum(len(str(row[0])) for row in rows)
        estimated_tokens = sum(_estimate_tokens(str(row[0])) for row in rows)
        total_vectors = len(rows)
        estimated_cost = estimated_tokens / 1000 * target.pricing_usd_per_1k_tokens
        batches = math.ceil(total_vectors / self.batch_size) if total_vectors else 0
        estimated_duration = batches / self.rate_limit_per_second
        migration_id = f"mig_{uuid.uuid4().hex[:12]}"
        plan = MigrationPlan(
            migration_id=migration_id,
            from_model=source.payload_model_name,
            to_model=target.payload_model_name,
            from_model_version=source.model_version,
            to_model_version=target.model_version,
            from_strategy=from_strategy,
            to_strategy=to_strategy,
            total_vectors=total_vectors,
            total_characters=total_characters,
            estimated_tokens=estimated_tokens,
            estimated_cost_usd=estimated_cost,
            estimated_duration_seconds=estimated_duration,
            predicted_retrieval_drift=self.predicted_retrieval_drift,
            batch_size=self.batch_size,
            rate_limit_per_second=self.rate_limit_per_second,
        )
        self.catalog.insert_migrations([_migration_row(plan)])
        return plan

    def _ensure_model_registered(self, model: EmbeddingModel) -> None:
        existing = self.catalog.query(
            "SELECT 1 FROM embedding_models WHERE model_name = ? AND model_version = ?",
            [model.payload_model_name, model.model_version],
        )
        if not existing:
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


def _migration_row(plan: MigrationPlan) -> dict[str, Any]:
    return {
        "migration_id": plan.migration_id,
        "source_model_name": plan.from_model,
        "source_model_version": plan.from_model_version,
        "target_model_name": plan.to_model,
        "target_model_version": plan.to_model_version,
        "source_strategy_name": plan.from_strategy,
        "source_strategy_version": None,
        "target_strategy_name": plan.to_strategy,
        "target_strategy_version": None,
        "status": plan.status,
        "total_vectors": plan.total_vectors,
        "migrated_vectors": 0,
        "total_characters": plan.total_characters,
        "estimated_tokens": plan.estimated_tokens,
        "estimated_cost_usd": plan.estimated_cost_usd,
        "estimated_duration_seconds": plan.estimated_duration_seconds,
        "predicted_retrieval_drift": plan.predicted_retrieval_drift,
        "batch_size": plan.batch_size,
        "rate_limit_per_second": plan.rate_limit_per_second,
    }