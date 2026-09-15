from typer.testing import CliRunner

from catalog.db import CatalogDB
from cli.gvpctl import app
from migration.planner import MigrationPlanner


def _seed_catalog(catalog: CatalogDB) -> None:
    catalog.insert_documents(
        [
            {
                "doc_id": "doc-1",
                "doc_version": 1,
                "title": "Title",
                "source_uri": "test://doc-1",
                "author": "Test",
                "classification_level": "public",
                "language": "en",
            }
        ]
    )
    catalog.insert_chunk_strategies(
        [
            {
                "strategy_name": "fixed_size_v1",
                "strategy_version": "1.0.0",
                "splitter_type": "recursive_character",
                "chunk_size_tokens": 300,
                "chunk_overlap_tokens": 30,
                "tokenizer_name": "cl100k_base",
            }
        ]
    )
    catalog.insert_chunks(
        [
            {
                "chunk_id": "doc-1:0000",
                "doc_id": "doc-1",
                "doc_version": 1,
                "strategy_name": "fixed_size_v1",
                "strategy_version": "1.0.0",
                "chunk_index": 0,
                "chunk_text": "one two three",
                "chunk_hash": "hash-1",
            },
            {
                "chunk_id": "doc-1:0001",
                "doc_id": "doc-1",
                "doc_version": 1,
                "strategy_name": "fixed_size_v1",
                "strategy_version": "1.0.0",
                "chunk_index": 1,
                "chunk_text": "four five",
                "chunk_hash": "hash-2",
            },
        ]
    )
    catalog.insert_embedding_models(
        [
            {
                "model_name": "bge-small-en-v1.5",
                "model_version": "1.0.0",
                "provider": "fastembed",
                "dimensions": 384,
            }
        ]
    )
    catalog.insert_vectors(
        [
            {
                "vector_id": "vec-1",
                "chunk_id": "doc-1:0000",
                "model_name": "bge-small-en-v1.5",
                "model_version": "1.0.0",
                "collection_name": "scifact_v1",
                "dimension": 384,
                "active": True,
            },
            {
                "vector_id": "vec-2",
                "chunk_id": "doc-1:0001",
                "model_name": "bge-small-en-v1.5",
                "model_version": "1.0.0",
                "collection_name": "scifact_v1",
                "dimension": 384,
                "active": False,
            },
        ]
    )


def test_plan_migration_counts_only_active_source_vectors() -> None:
    with CatalogDB(":memory:") as catalog:
        _seed_catalog(catalog)

        plan = MigrationPlanner(catalog, batch_size=1, rate_limit_per_second=2).plan_migration(
            "v1", "v2", "fixed_size_v1", "fixed_size_v2"
        )

        assert plan.total_vectors == 1
        assert plan.total_characters == len("one two three")
        assert plan.estimated_tokens > 0
        assert plan.estimated_cost_usd == 0.0
        assert plan.estimated_duration_seconds == 0.5
        assert catalog.query(
            "SELECT status, total_vectors FROM migrations WHERE migration_id = ?",
            [plan.migration_id],
        ) == [("planned", 1)]


def test_migrate_plan_cli_renders_with_approval(monkeypatch, tmp_path) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["migrate", "plan", "v1", "v2", "--approve"])

    assert result.exit_code == 0
    assert "Migration Plan mig_" in result.output
    assert "vector count" in result.output