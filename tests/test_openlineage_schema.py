import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, FormatChecker


SCHEMA_PATH = Path(__file__).parents[1] / "schemas" / "gvp_VectorEmbeddingDatasetFacet.json"


@pytest.fixture
def facet_schema() -> dict[str, object]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def test_canonical_facet_schema_accepts_valid_facet(
    facet_schema: dict[str, object],
) -> None:
    facet = {
        "_producer": "https://github.com/laila-kz/governed-vector-platform",
        "_schemaURL": "https://raw.githubusercontent.com/laila-kz/governed-vector-platform/main/schemas/gvp_VectorEmbeddingDatasetFacet.json",
        "model_name": "bge-small-en-v1.5",
        "model_version": "1.0.0",
        "dimension": 384,
        "distance_metric": "Cosine",
        "strategy_name": "fixed_size_v1",
        "strategy_version": "1.0.0",
        "chunk_size_tokens": 300,
        "chunk_overlap_tokens": 30,
        "tokenizer_name": "cl100k_base",
    }

    Draft202012Validator(
        facet_schema, format_checker=FormatChecker()
    ).validate(facet)


def test_canonical_facet_schema_rejects_invalid_lineage_metadata(
    facet_schema: dict[str, object],
) -> None:
    validator = Draft202012Validator(facet_schema, format_checker=FormatChecker())
    incomplete_facet = {
        "_producer": "https://github.com/laila-kz/governed-vector-platform",
        "_schemaURL": "https://example.com/schema.json",
        "model_name": "bge-small-en-v1.5",
        "distance_metric": "Manhattan",
    }

    errors = list(validator.iter_errors(incomplete_facet))

    assert any(error.validator == "required" for error in errors)
    assert any(error.validator == "enum" for error in errors)


def test_schema_id_matches_canonical_location(facet_schema: dict[str, object]) -> None:
    assert facet_schema["$id"] == (
        "https://raw.githubusercontent.com/laila-kz/governed-vector-platform/"
        "main/schemas/gvp_VectorEmbeddingDatasetFacet.json"
    )