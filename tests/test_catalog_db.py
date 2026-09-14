from pathlib import Path

import duckdb
import pytest

from catalog.db import CatalogDB


def test_catalog_initializes_all_tables_and_supports_lineage_inserts() -> None:
    with CatalogDB(":memory:") as catalog:
        tables = {
            row[0]
            for row in catalog.query(
                "SELECT table_name FROM information_schema.tables "
                "WHERE table_schema = 'main'"
            )
        }
        assert {
            "documents",
            "chunk_strategies",
            "chunks",
            "embedding_models",
            "vectors",
            "migrations",
            "retrieval_eval_runs",
        } <= tables

        with catalog.transaction():
            assert catalog.insert_documents(
                [
                    {
                        "doc_id": "doc-1",
                        "doc_version": 1,
                        "title": "Title",
                        "source_uri": "beir://scifact/doc-1",
                        "author": "BEIR/SciFact",
                        "classification_level": "public",
                        "language": "en",
                        "pii_masked_flag": False,
                    }
                ]
            ) == 1
            catalog.insert_chunk_strategies(
                [
                    {
                        "strategy_name": "fixed_size_v1",
                        "strategy_version": "1.0.0",
                        "splitter_type": "recursive_character",
                        "chunk_size_tokens": 300,
                        "chunk_overlap_tokens": 30,
                        "tokenizer_name": "cl100k_base",
                        "pii_masked_flag": False,
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
                        "chunk_text": "Text",
                        "chunk_hash": "hash",
                        "pii_masked_flag": False,
                    }
                ]
            )
            catalog.insert_embedding_models(
                [
                    {
                        "model_name": "bge-small-en-v1.5",
                        "model_version": "1.0.0",
                        "provider": "fastembed",
                        "dimensions": 384,
                        "pii_masked_flag": False,
                    }
                ]
            )
            catalog.insert_vectors(
                [
                    {
                        "vector_id": "vec_0001",
                        "chunk_id": "doc-1:0000",
                        "model_name": "bge-small-en-v1.5",
                        "model_version": "1.0.0",
                        "collection_name": "scifact_v1",
                        "dimension": 384,
                        "pii_masked_flag": False,
                    }
                ]
            )

        assert catalog.query("SELECT count(*) FROM vectors")[0][0] == 1


def test_catalog_transaction_rolls_back_on_foreign_key_error() -> None:
    with CatalogDB(":memory:") as catalog:
        with pytest.raises(duckdb.ConstraintException):
            with catalog.transaction():
                catalog.insert_vectors(
                    [
                        {
                            "vector_id": "orphan",
                            "chunk_id": "missing",
                            "model_name": "missing",
                            "model_version": "1.0.0",
                            "collection_name": "scifact_v1",
                            "dimension": 384,
                            "pii_masked_flag": False,
                        }
                    ]
                )
        assert catalog.query("SELECT count(*) FROM vectors")[0][0] == 0


def test_bulk_insert_rejects_unknown_or_inconsistent_rows(tmp_path: Path) -> None:
    with CatalogDB(tmp_path / "catalog.duckdb") as catalog:
        with pytest.raises(ValueError, match="Unknown catalog table"):
            catalog.insert_rows("unknown", [])
        with pytest.raises(ValueError, match="same columns"):
            catalog.insert_documents([{"doc_id": "one"}, {"doc_id": "two", "title": "Two"}])