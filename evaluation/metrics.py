"""Information-retrieval metrics and latency benchmarking utilities."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

from qdrant_client import QdrantClient

DEFAULT_CATALOG_PATH = Path("evaluation/beir_scifact_qrels.json")
Ranking = Sequence[str | tuple[str, float] | Mapping[str, Any]]


@dataclass(frozen=True)
class LatencyBenchmark:
    """Latency summary in milliseconds for one evaluated model."""

    model: str
    sample_count: int
    average_ms: float
    p50_ms: float
    p95_ms: float


@dataclass(frozen=True)
class EvaluationMetrics:
    """Aggregate IR quality metrics for one model."""

    model: str
    query_count: int
    recall_at_5: float
    recall_at_10: float
    ndcg_at_10: float
    mrr: float
    latency: LatencyBenchmark | None = None


def _document_id(item: str | tuple[str, float] | Mapping[str, Any]) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, tuple):
        if not item:
            raise ValueError("ranking tuple must contain a document id")
        return str(item[0])
    if isinstance(item, Mapping):
        for key in ("id", "doc_id", "document_id"):
            if key in item:
                return str(item[key])
    raise TypeError(f"unsupported ranking item: {item!r}")


def _recall_at_k(relevant: set[str], ranking: Ranking, k: int) -> float:
    if not relevant:
        return 0.0
    retrieved = {_document_id(item) for item in ranking[:k]}
    return len(relevant & retrieved) / len(relevant)


def _ndcg_at_k(relevant: set[str], ranking: Ranking, k: int) -> float:
    if not relevant:
        return 0.0
    gains = [1.0 if _document_id(item) in relevant else 0.0 for item in ranking[:k]]
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal_count = min(len(relevant), k)
    ideal_dcg = sum(1.0 / math.log2(index + 2) for index in range(ideal_count))
    return dcg / ideal_dcg if ideal_dcg else 0.0


def _reciprocal_rank(relevant: set[str], ranking: Ranking) -> float:
    for index, item in enumerate(ranking, start=1):
        if _document_id(item) in relevant:
            return 1.0 / index
    return 0.0


def percentile(values: Sequence[float], percentile_value: float) -> float:
    """Return an interpolated percentile, using the nearest-rank endpoints."""
    if not values:
        raise ValueError("cannot calculate a percentile from no values")
    if not 0.0 <= percentile_value <= 100.0:
        raise ValueError("percentile must be between 0 and 100")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile_value / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def benchmark_latency(
    model: str,
    queries: Iterable[str],
    retrieve: Callable[[str], Any],
) -> LatencyBenchmark:
    """Measure average, p50, and p95 retrieval latency in milliseconds."""
    durations: list[float] = []
    for query in queries:
        started = time.perf_counter()
        retrieve(query)
        durations.append((time.perf_counter() - started) * 1000.0)
    if not durations:
        raise ValueError("latency benchmark requires at least one query")
    return LatencyBenchmark(
        model=model,
        sample_count=len(durations),
        average_ms=statistics.fmean(durations),
        p50_ms=percentile(durations, 50.0),
        p95_ms=percentile(durations, 95.0),
    )


def evaluate_rankings(
    model: str,
    qrels: Mapping[str, Iterable[str]],
    rankings: Mapping[str, Ranking],
    *,
    latency: LatencyBenchmark | None = None,
) -> EvaluationMetrics:
    """Compute Recall@5/10, NDCG@10, and MRR over ranked query results.

    Queries missing from ``rankings`` are evaluated as empty rankings, while
    extra ranking entries are ignored. This keeps the denominator fixed to the
    benchmark qrels and makes incomplete retrieval runs visible in the scores.
    """
    if not qrels:
        raise ValueError("qrels must contain at least one query")
    totals = {"recall_at_5": 0.0, "recall_at_10": 0.0, "ndcg_at_10": 0.0, "mrr": 0.0}
    for query_id, relevant_ids in qrels.items():
        relevant = {str(doc_id) for doc_id in relevant_ids}
        ranking = rankings.get(query_id, ())
        totals["recall_at_5"] += _recall_at_k(relevant, ranking, 5)
        totals["recall_at_10"] += _recall_at_k(relevant, ranking, 10)
        totals["ndcg_at_10"] += _ndcg_at_k(relevant, ranking, 10)
        totals["mrr"] += _reciprocal_rank(relevant, ranking)
    query_count = len(qrels)
    return EvaluationMetrics(
        model=model,
        query_count=query_count,
        recall_at_5=totals["recall_at_5"] / query_count,
        recall_at_10=totals["recall_at_10"] / query_count,
        ndcg_at_10=totals["ndcg_at_10"] / query_count,
        mrr=totals["mrr"] / query_count,
        latency=latency,
    )


def load_catalog(path: Path | str = DEFAULT_CATALOG_PATH) -> list[dict[str, Any]]:
    """Load and minimally validate the canonical SciFact evaluation catalog."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("evaluation catalog must be a JSON list")
    for record in records:
        if not {"query_id", "text", "relevant_doc_ids"}.issubset(record):
            raise ValueError("evaluation catalog record is missing required fields")
    return records


def qdrant_retriever(
    qdrant_client: Any,
    embed: Callable[[list[str]], Iterable[Sequence[float]]],
    collection: str,
) -> Callable[[str, int], Ranking]:
    """Build a real Qdrant retriever compatible with client API generations."""
    def retrieve(query: str, limit: int) -> Ranking:
        vector = list(embed([query]))[0]
        query_points = getattr(qdrant_client, "query_points", None)
        if callable(query_points):
            result = query_points(
                collection_name=collection,
                query=list(vector),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
            points = getattr(result, "points", result)
        else:
            points = qdrant_client.search(
                collection_name=collection,
                query_vector=list(vector),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )
        rankings: list[dict[str, Any]] = []
        for point in points or []:
            payload = getattr(point, "payload", None)
            point_id = getattr(point, "id", None)
            score = getattr(point, "score", None)
            if isinstance(point, Mapping):
                payload = point.get("payload", {})
                point_id = point.get("id")
                score = point.get("score", 0.0)
            payload = payload or {}
            rankings.append({"id": str(payload.get("doc_id", point_id or "")), "score": score or 0.0})
        return rankings

    return retrieve


def evaluate_qdrant_collection(
    model: str,
    records: Sequence[Mapping[str, Any]],
    qdrant_client: Any,
    embed: Callable[[list[str]], Iterable[Sequence[float]]],
    collection: str,
) -> EvaluationMetrics:
    """Evaluate one populated Qdrant collection against catalog qrels."""
    qrels = {
        str(record["query_id"]): record["relevant_doc_ids"] for record in records
    }
    retrieve = qdrant_retriever(qdrant_client, embed, collection)
    rankings: dict[str, Ranking] = {}
    durations: list[float] = []
    for record in records:
        started = time.perf_counter()
        rankings[str(record["query_id"])] = retrieve(str(record["text"]), 10)
        durations.append((time.perf_counter() - started) * 1000.0)
    latency = LatencyBenchmark(
        model=model,
        sample_count=len(durations),
        average_ms=statistics.fmean(durations),
        p50_ms=percentile(durations, 50.0),
        p95_ms=percentile(durations, 95.0),
    )
    return evaluate_rankings(model, qrels, rankings, latency=latency)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG_PATH)
    parser.add_argument("--model-v1", default="bge-small-en-v1.5")
    parser.add_argument("--model-v2", default="bge-large-en-v1.5")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection-v1", default="scifact_v1")
    parser.add_argument("--collection-v2", default="scifact_v2_shadow")
    args = parser.parse_args()
    catalog = load_catalog(args.catalog)
    qrels = {str(record["query_id"]): record["relevant_doc_ids"] for record in catalog}
    from embedding.embedder import fastembedder
    from embedding.model_registry import find_model

    client = QdrantClient(url=args.qdrant_url)
    try:
        for collection in (args.collection_v1, args.collection_v2):
            if not client.collection_exists(collection):
                raise RuntimeError(f"Required Qdrant collection is missing: {collection}")
        output = {
            args.model_v1: asdict(evaluate_qdrant_collection(
                args.model_v1, catalog, client, fastembedder(find_model(args.model_v1)), args.collection_v1
            )),
            args.model_v2: asdict(evaluate_qdrant_collection(
                args.model_v2, catalog, client, fastembedder(find_model(args.model_v2)), args.collection_v2
            )),
        }
    finally:
        client.close()
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()