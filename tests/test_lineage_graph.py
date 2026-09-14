from catalog.db import CatalogDB
from catalog.lineage_service import LineageService


def _seed_catalog(catalog: CatalogDB) -> None:
    with catalog.transaction():
        catalog.insert_documents(
            [
                {
                    "doc_id": "doc-1",
                    "doc_version": 1,
                    "title": "Title",
                    "source_uri": "beir://scifact/doc-1",
                    "author": "BEIR/SciFact",
                    "classification_level": "public",
                    "language": "en",
                    "content_hash": "doc-hash",
                    "pii_masked_flag": True,
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
                    "pii_masked_flag": False,
                },
                {
                    "strategy_name": "fixed_size_v1",
                    "strategy_version": "2.0.0",
                    "splitter_type": "recursive_character",
                    "chunk_size_tokens": 500,
                    "chunk_overlap_tokens": 50,
                    "tokenizer_name": "cl100k_base",
                    "pii_masked_flag": False,
                },
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
                    "chunk_hash": "chunk-hash",
                    "pii_masked_flag": True,
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
                },
                {
                    "model_name": "bge-small-en-v1.5",
                    "model_version": "2.0.0",
                    "provider": "fastembed",
                    "dimensions": 384,
                    "pii_masked_flag": False,
                },
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
                    "active": False,
                    "pii_masked_flag": False,
                },
                {
                    "vector_id": "vec_0002",
                    "chunk_id": "doc-1:0000",
                    "model_name": "bge-small-en-v1.5",
                    "model_version": "2.0.0",
                    "collection_name": "scifact_v2",
                    "dimension": 384,
                    "active": True,
                    "pii_masked_flag": False,
                },
            ]
        )


def test_backward_trace_returns_complete_typed_provenance() -> None:
    with CatalogDB(":memory:") as catalog:
        _seed_catalog(catalog)
        record = LineageService(catalog).get_vector_lineage("vec_0001")

    assert record.collection_name == "scifact_v1"
    assert record.provider == "fastembed"
    assert record.chunk_text == "Text"
    assert record.source_uri == "beir://scifact/doc-1"
    assert record.content_hash == "doc-hash"
    assert record.pii_masked_flag is True


def test_forward_trace_and_staleness_resolver() -> None:
    with CatalogDB(":memory:") as catalog:
        _seed_catalog(catalog)
        service = LineageService(catalog)
        document = service.get_document_lineage("doc-1", 1)
        stale = service.get_stale_vectors("2.0.0", "1.0.0")

    assert len(document.chunks) == 1
    assert {vector.vector_id for vector in document.chunks[0].vectors} == {
        "vec_0001",
        "vec_0002",
    }
    assert [vector.vector_id for vector in stale] == ["vec_0001"]


def test_lineage_not_found_errors_are_explicit() -> None:
    with CatalogDB(":memory:") as catalog:
        service = LineageService(catalog)
        try:
            service.get_vector_lineage("missing")
        except KeyError as error:
            assert "missing" in str(error)
        else:
            raise AssertionError("missing vector should raise KeyError")