"""Versioned embedding model definitions used by the ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingModel:
    """Operational and lineage metadata for an embedding model."""

    registry_key: str
    model_name: str
    model_version: str
    dimensions: int
    provider: str
    pricing_usd_per_1k_tokens: float
    latency_ms_p95: float
    status: str = "active"

    @property
    def payload_model_name(self) -> str:
        """Return the short model name used in vector payloads."""
        return self.model_name.rsplit("/", maxsplit=1)[-1]


BASELINE_MODEL_V1 = EmbeddingModel(
    registry_key="v1",
    model_name="BAAI/bge-small-en-v1.5",
    model_version="1.0.0",
    dimensions=384,
    provider="fastembed",
    pricing_usd_per_1k_tokens=0.0,
    latency_ms_p95=100.0,
)

MODEL_REGISTRY: dict[str, EmbeddingModel] = {BASELINE_MODEL_V1.registry_key: BASELINE_MODEL_V1}


def get_model(registry_key: str = "v1") -> EmbeddingModel:
    """Look up a registered model by its stable registry key."""
    try:
        return MODEL_REGISTRY[registry_key]
    except KeyError as error:
        raise KeyError(f"Unknown embedding model registry key: {registry_key}") from error