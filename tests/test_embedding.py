from typing import Any, Sequence

from embedding.embedder import ingest_chunks
from embedding.model_registry import BASELINE_MODEL_V1, TARGET_MODEL_V2, get_model


class FakeQdrant:
    def __init__(self) -> None:
        self.collections: set[str] = set()
        self.operations: list[tuple[str, Any]] = []
        self.aliases: list[Any] = []

    def collection_exists(self, collection_name: str) -> bool:
        return collection_name in self.collections

    def create_collection(self, **kwargs: Any) -> None:
        self.collections.add(kwargs["collection_name"])
        self.operations.append(("create_collection", kwargs))

    def get_aliases(self) -> Any:
        return type("Aliases", (), {"aliases": self.aliases})()

    def update_collection_aliases(self, **kwargs: Any) -> None:
        self.operations.append(("aliases", kwargs))

    def upsert(self, **kwargs: Any) -> None:
        self.operations.append(("upsert", kwargs))


def test_baseline_model_registry_contract() -> None:
    model = get_model("v1")

    assert model == BASELINE_MODEL_V1
    assert model.model_name == "BAAI/bge-small-en-v1.5"
    assert model.payload_model_name == "bge-small-en-v1.5"
    assert model.dimensions == 384
    assert model.pricing_usd_per_1k_tokens == 0.0


def test_target_model_v2_registry_contract() -> None:
    model = get_model("v2")

    assert model == TARGET_MODEL_V2
    assert model.model_name == "BAAI/bge-large-en-v1.5"
    assert model.model_version == "2.0.0"
    assert model.payload_model_name == "bge-large-en-v1.5"
    assert model.dimensions == 1024
    assert model.provider == "fastembed"
    assert model.pricing_usd_per_1k_tokens == 0.0
    assert model.latency_ms_p95 == 250.0


def test_ingest_batches_vectors_and_builds_payload() -> None:
    client = FakeQdrant()
    chunks = [
        {
            "chunk_id": "doc-1:0000",
            "chunk_text": "first",
            "doc_id": "doc-1",
            "doc_version": 1,
            "strategy_name": "fixed_size_v1",
            "strategy_version": "1.0.0",
            "pii_masked_flag": True,
        },
        {
            "chunk_id": "doc-1:0001",
            "chunk_text": "second",
            "doc_id": "doc-1",
            "doc_version": 1,
            "strategy_name": "fixed_size_v1",
            "strategy_version": "1.0.0",
            "pii_masked_flag": False,
        },
    ]

    def fake_embed(texts: Sequence[str]) -> list[list[float]]:
        return [[float(index)] * 384 for index, _ in enumerate(texts, start=1)]

    count = ingest_chunks(chunks, client, embed=fake_embed, batch_size=1)

    assert count == 2
    upserts = [operation for name, operation in client.operations if name == "upsert"]
    assert len(upserts) == 2
    payload = upserts[0]["points"][0].payload
    assert payload["vector_id"] == "vec_0001"
    assert payload["model_name"] == "bge-small-en-v1.5"
    assert payload["model_version"] == "1.0.0"
    assert payload["pii_masked_flag"] is True
    alias_operations = [operation for name, operation in client.operations if name == "aliases"]
    assert len(alias_operations) == 1