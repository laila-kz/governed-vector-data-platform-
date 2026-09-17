"""Quality gates that authorize safe vector migration cutovers."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from catalog.db import CatalogDB
from evaluation.metrics import EvaluationMetrics, Ranking, evaluate_rankings
from evaluation.metrics import load_catalog
from embedding.model_registry import get_model

DEFAULT_EVALUATION_CATALOG = Path("evaluation/beir_scifact_qrels.json")
BASELINE_COLLECTION = "scifact_v1"
SHADOW_COLLECTION = "scifact_v2_shadow"

Retriever = Callable[[str, str, int], Ranking]


@dataclass(frozen=True)
class QualityGateDecision:
    migration_id: str
    passed: bool
    reason: str
    baseline: EvaluationMetrics
    shadow: EvaluationMetrics
    recall_at_10_delta: float
    ndcg_at_10_delta: float
    error_rate: float


class QualityGate:
    """Evaluate baseline and shadow retrieval before cutover."""

    def __init__(
        self,
        catalog: CatalogDB,
        retrieve: Retriever,
        *,
        evaluation_catalog: Path | str = DEFAULT_EVALUATION_CATALOG,
        recall_tolerance: float = 0.02,
    ) -> None:
        if recall_tolerance < 0:
            raise ValueError("recall_tolerance must not be negative")
        self.catalog = catalog
        self.retrieve = retrieve
        self.evaluation_catalog = Path(evaluation_catalog)
        self.recall_tolerance = recall_tolerance

    def _migration(self, migration_id: str) -> tuple[Any, ...]:
        rows = self.catalog.query(
            """
            SELECT source_model_name, source_model_version, target_model_name,
                     target_model_version, status, error_count, total_vectors
            FROM migrations WHERE migration_id = ?
            """,
            [migration_id],
        )
        if not rows:
            raise ValueError(f"Unknown migration: {migration_id}")
        return rows[0]

    def _queries(self) -> dict[str, dict[str, Any]]:
        records = load_catalog(self.evaluation_catalog)
        return {str(record["query_id"]): record for record in records}

    def _evaluate_collection(
        self,
        model: str,
        collection: str,
        queries: Mapping[str, Mapping[str, Any]],
    ) -> tuple[EvaluationMetrics, float]:
        qrels = {
            query_id: record["relevant_doc_ids"] for query_id, record in queries.items()
        }
        rankings: dict[str, Ranking] = {}
        errors = 0
        latencies: list[float] = []
        for query_id, record in queries.items():
            started = time.perf_counter()
            try:
                rankings[query_id] = self.retrieve(collection, str(record["text"]), 10)
            except Exception:
                errors += 1
                rankings[query_id] = []
            latencies.append((time.perf_counter() - started) * 1000.0)
        model_metrics = evaluate_rankings(model, qrels, rankings)
        error_rate = errors / len(queries) if queries else 1.0
        return model_metrics, error_rate

    def _persist_run(
        self,
        migration_id: str,
        metrics: EvaluationMetrics,
        error_rate: float,
        status: str,
    ) -> None:
        try:
            model = get_model(metrics.model)
        except KeyError:
            model = get_model("v1" if "small" in str(metrics.model).lower() else "v2")
        self.catalog.insert_retrieval_eval_runs(
            [{
                "run_id": str(uuid.uuid4()),
                "migration_id": migration_id,
                "model_name": model.payload_model_name,
                "model_version": model.model_version,
                "strategy_name": None,
                "strategy_version": None,
                "dataset_name": "beir_scifact",
                "recall_at_5": metrics.recall_at_5,
                "recall_at_10": metrics.recall_at_10,
                "ndcg_at_10": metrics.ndcg_at_10,
                "mrr": metrics.mrr,
                "cosine_drift": None,
                "error_rate": error_rate,
                "status": status,
                "metrics_json": json.dumps(asdict(metrics)),
                "started_at": None,
                "completed_at": None,
                "pii_masked_flag": True,
            }]
        )

    def check(self, migration_id: str) -> QualityGateDecision:
        """Evaluate both collections and persist pass/fail evaluation runs."""
        migration = self._migration(migration_id)
        source_model, target_model = str(migration[0]), str(migration[2])
        queries = self._queries()
        baseline, baseline_errors = self._evaluate_collection(
            source_model, BASELINE_COLLECTION, queries
        )
        shadow, shadow_errors = self._evaluate_collection(
            target_model, SHADOW_COLLECTION, queries
        )
        migration_error_rate = (
            int(migration[5] or 0) / int(migration[6])
            if int(migration[6] or 0) > 0
            else float(int(migration[5] or 0) > 0)
        )
        error_rate = max(baseline_errors, shadow_errors, migration_error_rate)
        recall_delta = shadow.recall_at_10 - baseline.recall_at_10
        ndcg_delta = shadow.ndcg_at_10 - baseline.ndcg_at_10
        passed = (
            recall_delta >= -self.recall_tolerance
            and ndcg_delta >= 0.0
            and error_rate == 0.0
        )
        reason = "quality gate passed" if passed else (
            "quality gate failed: "
            f"recall_delta={recall_delta:.4f}, "
            f"ndcg_delta={ndcg_delta:.4f}, error_rate={error_rate:.4f}"
        )
        status = "passed" if passed else "failed"
        self._persist_run(migration_id, baseline, baseline_errors, status)
        self._persist_run(migration_id, shadow, shadow_errors, status)
        return QualityGateDecision(
            migration_id,
            passed,
            reason,
            baseline,
            shadow,
            recall_delta,
            ndcg_delta,
            error_rate,
        )


def qdrant_retriever(qdrant_client: Any, embedders: Mapping[str, Callable[[list[str]], list[list[float]]]]) -> Retriever:
    """Build a retriever for Qdrant collections using model-specific embedders."""
    def retrieve(collection: str, query: str, limit: int) -> Ranking:
        model_name = "bge-large-en-v1.5" if collection == SHADOW_COLLECTION else "bge-small-en-v1.5"
        vector = list(embedders[model_name]([query])[0])
        result = qdrant_client.query_points(
            collection_name=collection,
            query=vector,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        points = getattr(result, "points", result)
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