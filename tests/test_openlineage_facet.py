from openlineage.client.run import RunState
from openlineage.client.serde import Serde

from catalog.openlineage_emitter import (
    DEFAULT_MARQUEZ_URL,
    PRODUCER_URL,
    SCHEMA_URL,
    OpenLineageEmitter,
    VectorEmbeddingDatasetFacet,
    event_to_json,
)
from embedding.embedder import ingest_chunks


class FakeOpenLineageClient:
    def __init__(self) -> None:
        self.events = []

    def emit(self, event) -> None:
        self.events.append(event)


def _facet() -> VectorEmbeddingDatasetFacet:
    return VectorEmbeddingDatasetFacet(
        model_name="bge-small-en-v1.5",
        model_version="1.0.0",
        dimension=384,
        strategy_name="fixed_size_v1",
        strategy_version="1.0.0",
        chunk_size_tokens=300,
    )


def test_custom_facet_matches_canonical_metadata() -> None:
    facet = _facet()
    payload = Serde.to_dict(facet)

    assert payload["_producer"] == PRODUCER_URL
    assert payload["_schemaURL"] == SCHEMA_URL
    assert payload["dimension"] == 384
    assert payload["distance_metric"] == "Cosine"


def test_marquez_factory_uses_configured_endpoint(monkeypatch) -> None:
    captured = {}

    class FakeClient:
        def __init__(self, url: str) -> None:
            captured["url"] = url

        def emit(self, event) -> None:
            return None

    monkeypatch.setattr("catalog.openlineage_emitter.OpenLineageClient", FakeClient)
    OpenLineageEmitter.for_marquez(_facet())
    assert captured["url"] == DEFAULT_MARQUEZ_URL


def test_emitter_sends_start_and_complete_with_dataset_lineage() -> None:
    client = FakeOpenLineageClient()
    emitter = OpenLineageEmitter(client, _facet())

    start = emitter.emit_start("run-1", "scifact_v1")
    complete = emitter.emit_complete("run-1", "scifact_v1")

    assert [event.eventType for event in client.events] == [RunState.START, RunState.COMPLETE]
    assert start.inputs[0].namespace == "dataset"
    assert start.inputs[0].name == "scifact_corpus"
    assert complete.outputs[0].name == "qdrant_scifact_v1"
    assert "gvp_VectorEmbeddingDatasetFacet" in complete.outputs[0].outputFacets
    assert '"eventType": "START"' in event_to_json(start)


def test_emitter_wraps_batch_operation() -> None:
    client = FakeOpenLineageClient()
    emitter = OpenLineageEmitter(client, _facet())

    result = emitter.emit_batch("run-2", "scifact_v1", lambda: "done")

    assert result == "done"
    assert [event.eventType for event in client.events] == [RunState.START, RunState.COMPLETE]


def test_batch_embedding_emits_start_and_complete() -> None:
    qdrant = type("FakeQdrant", (), {})()
    qdrant.collections = set()
    qdrant.operations = []
    qdrant.aliases = []
    qdrant.collection_exists = lambda name: name in qdrant.collections
    qdrant.create_collection = lambda **kwargs: qdrant.collections.add(kwargs["collection_name"])
    qdrant.get_aliases = lambda: type("Aliases", (), {"aliases": qdrant.aliases})()
    qdrant.update_collection_aliases = lambda **kwargs: None
    qdrant.upsert = lambda **kwargs: qdrant.operations.append(kwargs)

    client = FakeOpenLineageClient()
    emitter = OpenLineageEmitter(client, _facet())
    count = ingest_chunks(
        [
            {
                "chunk_id": "doc:0000",
                "chunk_text": "text",
                "doc_id": "doc",
                "strategy_name": "fixed_size_v1",
                "strategy_version": "1.0.0",
            }
        ],
        qdrant,
        embed=lambda texts: [[0.1] * 384 for _ in texts],
        lineage_emitter=emitter,
    )

    assert count == 1
    assert [event.eventType for event in client.events] == [RunState.START, RunState.COMPLETE]