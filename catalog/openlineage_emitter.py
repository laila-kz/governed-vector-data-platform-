"""Emit OpenLineage events for governed embedding batches."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

import attr
from openlineage.client.facet import BaseFacet
from openlineage.client import OpenLineageClient
from openlineage.client.run import (
    InputDataset,
    Job,
    OutputDataset,
    Run,
    RunEvent,
    RunState,
)
from openlineage.client.serde import Serde

SCHEMA_URL = (
    "https://raw.githubusercontent.com/laila-kz/governed-vector-platform/"
    "main/schemas/gvp_VectorEmbeddingDatasetFacet.json"
)
PRODUCER_URL = "https://github.com/laila-kz/governed-vector-platform"
OPENLINEAGE_SCHEMA_URL = "https://openlineage.io/spec/1-0-5/OpenLineage.json"
DEFAULT_NAMESPACE = "governed-vector-platform"
DEFAULT_JOB_NAME = "batch_embedding"
DEFAULT_INPUT_DATASET = "scifact_corpus"
DEFAULT_MARQUEZ_URL = "http://localhost:5000/api/v1/lineage"


@attr.s(init=False)
class VectorEmbeddingDatasetFacet(BaseFacet):
    """Custom dataset facet for embedding model and chunking provenance."""

    model_name: str = attr.ib()
    model_version: str = attr.ib()
    dimension: int = attr.ib()
    strategy_name: str = attr.ib()
    strategy_version: str = attr.ib()
    chunk_size_tokens: int = attr.ib()
    distance_metric: str = attr.ib(default="Cosine")
    tokenizer_name: str = attr.ib(default="cl100k_base")

    def __init__(
        self,
        model_name: str,
        model_version: str,
        dimension: int,
        strategy_name: str,
        strategy_version: str,
        chunk_size_tokens: int,
        distance_metric: str = "Cosine",
        tokenizer_name: str = "cl100k_base",
        **_: Any,
    ) -> None:
        if distance_metric not in {"Cosine", "Dot", "Euclidean"}:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")
        if dimension <= 0 or chunk_size_tokens <= 0:
            raise ValueError("dimension and chunk_size_tokens must be greater than zero")
        self.model_name = model_name
        self.model_version = model_version
        self.dimension = dimension
        self.strategy_name = strategy_name
        self.strategy_version = strategy_version
        self.chunk_size_tokens = chunk_size_tokens
        self.distance_metric = distance_metric
        self.tokenizer_name = tokenizer_name
        self._producer = PRODUCER_URL
        self._schemaURL = SCHEMA_URL


class EventClient(Protocol):
    def emit(self, event: RunEvent) -> None:
        """Send an OpenLineage event."""


class OpenLineageEmitter:
    """Build and emit START/COMPLETE events for one embedding batch run."""

    def __init__(
        self,
        client: EventClient,
        facet: VectorEmbeddingDatasetFacet,
        namespace: str = DEFAULT_NAMESPACE,
        job_name: str = DEFAULT_JOB_NAME,
        input_dataset: str = DEFAULT_INPUT_DATASET,
    ) -> None:
        self.client = client
        self.facet = facet
        self.namespace = namespace
        self.job_name = job_name
        self.input_dataset = input_dataset

    @classmethod
    def for_marquez(
        cls,
        facet: VectorEmbeddingDatasetFacet,
        url: str = DEFAULT_MARQUEZ_URL,
        **kwargs: Any,
    ) -> "OpenLineageEmitter":
        """Create an emitter backed by the Marquez OpenLineage endpoint."""
        return cls(OpenLineageClient(url=url), facet, **kwargs)

    def _event(
        self,
        state: RunState,
        run_id: str,
        collection_name: str,
        event_time: str | None = None,
    ) -> RunEvent:
        try:
            normalized_run_id = str(UUID(run_id))
        except ValueError:
            normalized_run_id = str(uuid5(NAMESPACE_URL, run_id))
        output = OutputDataset(
            namespace="dataset",
            name=f"qdrant_{collection_name}",
            outputFacets={"gvp_VectorEmbeddingDatasetFacet": self.facet},
        )
        return RunEvent(
            eventType=state,
            eventTime=event_time or datetime.now(timezone.utc).isoformat(),
            run=Run(runId=normalized_run_id),
            job=Job(namespace=self.namespace, name=self.job_name),
            producer=PRODUCER_URL,
            inputs=[InputDataset(namespace="dataset", name=self.input_dataset)],
            outputs=[output],
            schemaURL=OPENLINEAGE_SCHEMA_URL,
        )

    def emit_start(self, run_id: str, collection_name: str) -> RunEvent:
        event = self._event(RunState.START, run_id, collection_name)
        self.client.emit(event)
        return event

    def emit_complete(self, run_id: str, collection_name: str) -> RunEvent:
        event = self._event(RunState.COMPLETE, run_id, collection_name)
        self.client.emit(event)
        return event

    def emit_batch(
        self,
        run_id: str,
        collection_name: str,
        operation: Any,
    ) -> Any:
        """Emit START, execute an operation, then emit COMPLETE on success."""
        self.emit_start(run_id, collection_name)
        result = operation()
        self.emit_complete(run_id, collection_name)
        return result


def event_to_json(event: RunEvent) -> str:
    """Serialize a RunEvent using the OpenLineage client serializer."""
    return Serde.to_json(event)


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=str(uuid4()))
    parser.add_argument("--collection", default="scifact_v1")
    args = parser.parse_args()
    facet = VectorEmbeddingDatasetFacet(
        model_name="bge-small-en-v1.5",
        model_version="1.0.0",
        dimension=384,
        strategy_name="fixed_size_v1",
        strategy_version="1.0.0",
        chunk_size_tokens=300,
    )
    print(event_to_json(
        OpenLineageEmitter(
            client=_NoopClient(),
            facet=facet,
        )._event(RunState.START, args.run_id, args.collection)
    ))


class _NoopClient:
    def emit(self, event: RunEvent) -> None:
        return None


if __name__ == "__main__":
    main()