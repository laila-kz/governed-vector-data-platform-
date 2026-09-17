from pathlib import Path
from typing import Any

import pytest

from catalog.db import CatalogDB
from evaluation.quality_gate import QualityGate, qdrant_retriever
from migration.planner import MigrationPlanner
from tests.test_shadow_migration import _seed_catalog


def _write_catalog(path: Path) -> None:
    path.write_text(
        '[{"query_id":"q1","text":"claim","relevant_doc_ids":["doc-1"]}]\n',
        encoding="utf-8",
    )


def _migration(catalog: CatalogDB) -> str:
    _seed_catalog(catalog)
    return MigrationPlanner(catalog).plan_migration(
        "v1", "v2", "fixed_size_v1", "fixed_size_v1"
    ).migration_id


def test_quality_gate_passes_and_persists_both_runs(tmp_path: Path) -> None:
    evaluation_catalog = tmp_path / "catalog.json"
    _write_catalog(evaluation_catalog)
    with CatalogDB(":memory:") as catalog:
        migration_id = _migration(catalog)

        def retrieve(collection: str, query: str, limit: int) -> list[dict[str, Any]]:
            return [{"id": "doc-1", "score": 1.0}]

        decision = QualityGate(
            catalog, retrieve, evaluation_catalog=evaluation_catalog
        ).check(migration_id)

        assert decision.passed is True
        assert catalog.query(
            "SELECT status, error_rate FROM retrieval_eval_runs "
            "WHERE migration_id = ? ORDER BY model_name",
            [migration_id],
        ) == [("passed", 0.0), ("passed", 0.0)]


def test_quality_gate_blocks_ndcg_regression_and_persists_failure(tmp_path: Path) -> None:
    evaluation_catalog = tmp_path / "catalog.json"
    _write_catalog(evaluation_catalog)
    with CatalogDB(":memory:") as catalog:
        migration_id = _migration(catalog)

        def retrieve(collection: str, query: str, limit: int) -> list[str]:
            if collection == "scifact_v1":
                return ["doc-1"]
            return ["irrelevant", "doc-1"]

        decision = QualityGate(
            catalog, retrieve, evaluation_catalog=evaluation_catalog
        ).check(migration_id)

        assert decision.passed is False
        assert "ndcg_delta" in decision.reason
        assert catalog.query(
            "SELECT status FROM retrieval_eval_runs WHERE migration_id = ?",
            [migration_id],
        ) == [("failed",), ("failed",)]


def test_quality_gate_fails_when_retrieval_errors_occur(tmp_path: Path) -> None:
    evaluation_catalog = tmp_path / "catalog.json"
    _write_catalog(evaluation_catalog)
    with CatalogDB(":memory:") as catalog:
        migration_id = _migration(catalog)

        def retrieve(collection: str, query: str, limit: int) -> list[str]:
            if collection == "scifact_v2_shadow":
                raise ConnectionError("shadow unavailable")
            return ["doc-1"]

        decision = QualityGate(
            catalog, retrieve, evaluation_catalog=evaluation_catalog
        ).check(migration_id)

        assert decision.passed is False
        assert decision.error_rate == 1.0


def test_quality_gate_rejects_invalid_configuration_and_unknown_migration() -> None:
    with CatalogDB(":memory:") as catalog:
        with pytest.raises(ValueError, match="must not be negative"):
            QualityGate(catalog, lambda collection, query, limit: [], recall_tolerance=-1)
        with pytest.raises(ValueError, match="Unknown migration"):
            QualityGate(catalog, lambda collection, query, limit: []).check("missing")


def test_qdrant_retriever_supports_object_and_mapping_points() -> None:
    class Point:
        id = 7
        payload = {"doc_id": "object-doc"}
        score = 0.9

    class Result:
        points = [Point(), {"id": 8, "payload": {}, "score": 0.8}]

    class Client:
        def query_points(self, **kwargs: Any) -> Result:
            assert kwargs["collection_name"] == "scifact_v2_shadow"
            assert kwargs["limit"] == 3
            return Result()

    retriever = qdrant_retriever(
        Client(),
        {
            "bge-small-en-v1.5": lambda texts: [[1.0]],
            "bge-large-en-v1.5": lambda texts: [[2.0]],
        },
    )

    assert retriever("scifact_v2_shadow", "claim", 3) == [
        {"id": "object-doc", "score": 0.9},
        {"id": "8", "score": 0.8},
    ]