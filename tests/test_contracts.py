import json
from pathlib import Path

import pytest

from contracts.contract_validator import validate_document, validate_file


@pytest.fixture
def valid_document() -> dict[str, object]:
    return {
        "doc_id": "doc-1",
        "title": "A paper",
        "text": "An abstract.",
        "metadata": {
            "source_uri": "https://example.test/doc-1",
            "author": "Researcher",
            "classification_level": "public",
            "language": "en",
        },
    }


def test_valid_document_is_parsed(valid_document: dict[str, object]) -> None:
    document = validate_document(valid_document)

    assert document.doc_id == "doc-1"
    assert document.metadata.classification_level == "public"


def test_disallowed_metadata_value_is_rejected(
    valid_document: dict[str, object],
) -> None:
    metadata = valid_document["metadata"]
    assert isinstance(metadata, dict)
    metadata["classification_level"] = "top-secret"

    with pytest.raises(ValueError, match="classification_level"):
        validate_document(valid_document)


def test_invalid_records_are_quarantined(
    tmp_path: Path, valid_document: dict[str, object]
) -> None:
    invalid_document = {
        **valid_document,
        "metadata": {**valid_document["metadata"], "language": "xx"},
    }
    input_path = tmp_path / "documents.jsonl"
    input_path.write_text(
        "\n".join(json.dumps(record) for record in [valid_document, invalid_document])
        + "\n",
        encoding="utf-8",
    )

    report = validate_file(input_path, quarantine_dir=tmp_path / "quarantine")

    assert report["valid_count"] == 1
    assert report["invalid_count"] == 1
    report_path = Path(report["quarantine_report"])
    saved_report = json.loads(report_path.read_text(encoding="utf-8"))
    assert saved_report["invalid_records"][0]["record_number"] == 2
    assert saved_report["quarantine_report"] == str(report_path)