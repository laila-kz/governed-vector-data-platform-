"""Validate governed document records and quarantine invalid input."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

DEFAULT_CONTRACT_PATH = Path(__file__).with_name("document_contract.yaml")
DEFAULT_QUARANTINE_DIR = Path("data/quarantine")


class DocumentMetadata(BaseModel):
    """Metadata required for governance and lineage."""

    model_config = ConfigDict(extra="forbid")

    source_uri: str = Field(min_length=1)
    author: str = Field(min_length=1)
    classification_level: str = Field(min_length=1)
    language: str = Field(min_length=2)


class DocumentRecord(BaseModel):
    """Canonical document envelope accepted by downstream ingestion."""

    model_config = ConfigDict(extra="forbid")

    doc_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: DocumentMetadata


from functools import lru_cache


@lru_cache(maxsize=16)
def _load_contract_cached(resolved_path: str) -> dict[str, Any]:
    with Path(resolved_path).open(encoding="utf-8") as contract_file:
        contract = yaml.safe_load(contract_file)

    if not isinstance(contract, dict):
        raise ValueError("The document contract must contain a YAML mapping.")
    for key in ("required_fields", "required_metadata_fields", "allowed_values", "character_limits"):
        if key not in contract:
            raise ValueError(f"The document contract is missing {key!r}.")
    return contract


def load_contract(contract_path: Path | str = DEFAULT_CONTRACT_PATH) -> dict[str, Any]:
    """Load and minimally validate the YAML contract configuration."""
    return _load_contract_cached(str(Path(contract_path).resolve()))


def _validate_contract_rules(
    document: DocumentRecord, contract: dict[str, Any]
) -> list[dict[str, Any]]:
    """Apply policy rules that are intentionally configured in YAML."""
    errors: list[dict[str, Any]] = []
    limits = contract["character_limits"]
    values = contract["allowed_values"]

    fields = {
        "doc_id": document.doc_id,
        "title": document.title,
        "text": document.text,
        "source_uri": document.metadata.source_uri,
        "author": document.metadata.author,
        "classification_level": document.metadata.classification_level,
        "language": document.metadata.language,
    }
    for field_name, value in fields.items():
        field_limits = limits.get(field_name)
        if field_limits is None:
            continue
        length = len(value)
        if length < field_limits["min"] or length > field_limits["max"]:
            errors.append(
                {
                    "type": "string_length",
                    "loc": [field_name],
                    "msg": (
                        f"length must be between {field_limits['min']} and "
                        f"{field_limits['max']} characters; got {length}"
                    ),
                }
            )

    for field_name, value in {
        "classification_level": document.metadata.classification_level,
        "language": document.metadata.language,
    }.items():
        allowed = values.get(field_name, [])
        if value not in allowed:
            errors.append(
                {
                    "type": "literal_error",
                    "loc": [field_name],
                    "msg": f"value must be one of {allowed}; got {value!r}",
                }
            )
    return errors


def validate_document(
    record: dict[str, Any], contract: dict[str, Any] | None = None
) -> DocumentRecord:
    """Validate one record and return its typed document model."""
    active_contract = contract or load_contract()
    document = DocumentRecord.model_validate(record)
    rule_errors = _validate_contract_rules(document, active_contract)
    if rule_errors:
        raise ValueError(json.dumps(rule_errors, ensure_ascii=True))
    return document


def _validation_errors(
    record: dict[str, Any], contract: dict[str, Any]
) -> list[dict[str, Any]]:
    try:
        validate_document(record, contract)
    except ValidationError as error:
        return error.errors(include_url=False)
    except ValueError as error:
        try:
            return json.loads(str(error))
        except json.JSONDecodeError:
            return [{"type": "value_error", "loc": [], "msg": str(error)}]
    return []


def _read_records(input_path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    suffix = input_path.suffix.lower()
    if suffix == ".jsonl":
        with input_path.open(encoding="utf-8") as input_file:
            for line_number, line in enumerate(input_file, start=1):
                if line.strip():
                    yield line_number, json.loads(line)
        return
    if suffix == ".json":
        with input_path.open(encoding="utf-8") as input_file:
            payload = json.load(input_file)
        records = payload if isinstance(payload, list) else [payload]
        yield from enumerate(records, start=1)
        return
    raise ValueError("Input must be a .json or .jsonl file.")


def validate_file(
    input_path: Path,
    contract_path: Path = DEFAULT_CONTRACT_PATH,
    quarantine_dir: Path = DEFAULT_QUARANTINE_DIR,
    report_name: str = "document_contract_errors.json",
) -> dict[str, Any]:
    """Validate a JSON/JSONL file and write rejected records to quarantine."""
    contract = load_contract(contract_path)
    valid_count = 0
    invalid_records: list[dict[str, Any]] = []
    for record_number, record in _read_records(Path(input_path)):
        if not isinstance(record, dict):
            errors = [{"type": "mapping_type", "loc": [], "msg": "record must be an object"}]
        else:
            errors = _validation_errors(record, contract)
        if errors:
            invalid_records.append(
                {"record_number": record_number, "record": record, "errors": errors}
            )
        else:
            valid_count += 1

    report = {
        "contract_name": contract.get("contract_name"),
        "contract_version": contract.get("contract_version"),
        "source_file": str(input_path),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "valid_count": valid_count,
        "invalid_count": len(invalid_records),
        "invalid_records": invalid_records,
    }
    if invalid_records:
        quarantine_path = Path(quarantine_dir)
        quarantine_path.mkdir(parents=True, exist_ok=True)
        report_path = quarantine_path / report_name
        report["quarantine_report"] = str(report_path)
        with report_path.open("w", encoding="utf-8") as report_file:
            json.dump(report, report_file, indent=2, ensure_ascii=True)
            report_file.write("\n")
    return report


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON or JSONL document file to validate")
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT_PATH)
    parser.add_argument("--quarantine-dir", type=Path, default=DEFAULT_QUARANTINE_DIR)
    args = parser.parse_args()
    report = validate_file(args.input, args.contract, args.quarantine_dir)
    print(json.dumps(report, indent=2, ensure_ascii=True))
    if report["invalid_count"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()