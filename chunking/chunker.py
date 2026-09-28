"""Create deterministic token chunks and persist them as a Lance dataset."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import lancedb
import tiktoken
import yaml

DEFAULT_STRATEGY_PATH = Path(__file__).parent / "strategies" / "strategy_v1_fixed.yaml"
DEFAULT_DATASET_PATH = Path("data/lance_lakehouse/chunks.lance")


@dataclass(frozen=True)
class ChunkingStrategy:
    strategy_name: str
    version: str
    splitter_type: str
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    tokenizer: str

    @classmethod
    def from_yaml(cls, strategy_path: Path) -> "ChunkingStrategy":
        with Path(strategy_path).open(encoding="utf-8") as strategy_file:
            values = yaml.safe_load(strategy_file)
        if not isinstance(values, dict):
            raise ValueError("Chunking strategy must contain a YAML mapping.")

        required = (
            "strategy_name",
            "version",
            "splitter_type",
            "chunk_size_tokens",
            "chunk_overlap_tokens",
            "tokenizer",
        )
        missing = [field_name for field_name in required if field_name not in values]
        if missing:
            raise ValueError(f"Chunking strategy is missing fields: {missing}")

        chunk_size = int(values["chunk_size_tokens"])
        overlap = int(values["chunk_overlap_tokens"])
        if chunk_size <= 0:
            raise ValueError("chunk_size_tokens must be greater than zero")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("chunk_overlap_tokens must be >= 0 and less than chunk size")
        return cls(
            strategy_name=str(values["strategy_name"]),
            version=str(values["version"]),
            splitter_type=str(values["splitter_type"]),
            chunk_size_tokens=chunk_size,
            chunk_overlap_tokens=overlap,
            tokenizer=str(values["tokenizer"]),
        )


from functools import lru_cache


@lru_cache(maxsize=8)
def _get_encoding(tokenizer_name: str) -> Any:
    return tiktoken.get_encoding(tokenizer_name)


def _chunk_text(
    text: str, strategy: ChunkingStrategy, encoding: Any | None = None
) -> Iterable[str]:
    if encoding is None:
        encoding = _get_encoding(strategy.tokenizer)
    tokens = encoding.encode(text, disallowed_special=())
    step = strategy.chunk_size_tokens - strategy.chunk_overlap_tokens
    for start in range(0, len(tokens), step):
        chunk = encoding.decode(tokens[start : start + strategy.chunk_size_tokens])
        if chunk:
            yield chunk
        if start + strategy.chunk_size_tokens >= len(tokens):
            break


def _chunk_hash(chunk_text: str, strategy_version: str) -> str:
    value = f"{chunk_text}{strategy_version}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def chunk_documents(
    documents: Iterable[Mapping[str, Any]],
    strategy_path: Path = DEFAULT_STRATEGY_PATH,
) -> list[dict[str, Any]]:
    """Chunk document mappings into deterministic, Lance-ready records."""
    strategy = ChunkingStrategy.from_yaml(strategy_path)
    encoding = _get_encoding(strategy.tokenizer)
    chunks: list[dict[str, Any]] = []
    for document in documents:
        if "doc_id" not in document or "text" not in document:
            raise ValueError("Each document must contain doc_id and text")
        doc_id = str(document["doc_id"])
        text = document["text"]
        if not isinstance(text, str):
            raise TypeError("Document text must be a string")
        for chunk_index, chunk_text in enumerate(_chunk_text(text, strategy, encoding=encoding)):
            chunk_hash = _chunk_hash(chunk_text, strategy.version)
            chunks.append(
                {
                    "chunk_id": f"{doc_id}:{chunk_index:04d}",
                    "chunk_index": chunk_index,
                    "chunk_text": chunk_text,
                    "chunk_hash": chunk_hash,
                    "doc_id": doc_id,
                    "doc_version": int(document.get("doc_version", 1)),
                    "strategy_name": strategy.strategy_name,
                    "strategy_version": strategy.version,
                    "tokenizer": strategy.tokenizer,
                    "pii_masked_flag": bool(document.get("pii_masked_flag", False)),
                }
            )
    return chunks


def write_chunks(
    chunks: Iterable[Mapping[str, Any]],
    dataset_path: Path = DEFAULT_DATASET_PATH,
    mode: str = "overwrite",
) -> Path:
    """Write chunk records to ``data/lance_lakehouse/chunks.lance``."""
    rows = [dict(chunk) for chunk in chunks]
    if not rows:
        raise ValueError("Cannot create a Lance dataset without chunk records")
    dataset_path = Path(dataset_path)
    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    database = lancedb.connect(str(dataset_path.parent))
    database.create_table(dataset_path.stem, data=rows, mode=mode)
    return dataset_path


def read_chunks(dataset_path: Path = DEFAULT_DATASET_PATH) -> list[dict[str, Any]]:
    """Read all persisted chunk records from a Lance dataset."""
    dataset_path = Path(dataset_path)
    database = lancedb.connect(str(dataset_path.parent))
    table = database.open_table(dataset_path.stem)
    return table.to_arrow().to_pylist()


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="JSON document file containing a list of records")
    parser.add_argument("--strategy", type=Path, default=DEFAULT_STRATEGY_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_DATASET_PATH)
    args = parser.parse_args()

    import json

    with args.input.open(encoding="utf-8") as input_file:
        documents = json.load(input_file)
    chunks = chunk_documents(documents, args.strategy)
    write_chunks(chunks, args.output)
    print(f"wrote {len(chunks)} chunks to {args.output}")


if __name__ == "__main__":
    main()