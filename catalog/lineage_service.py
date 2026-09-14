"""Bidirectional provenance queries over the DuckDB metadata catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from catalog.db import DEFAULT_DATABASE_PATH, CatalogDB


class VectorLineageRecord(BaseModel):
    """Complete backward provenance for one vector."""

    model_config = ConfigDict(frozen=True)

    vector_id: str
    collection_name: str
    dimension: int
    model_name: str
    model_version: str
    provider: str
    chunk_id: str
    doc_id: str
    doc_version: int
    chunk_text: str
    strategy_name: str
    strategy_version: str
    source_uri: str
    content_hash: str | None
    pii_masked_flag: bool


class VectorLineageSummary(BaseModel):
    """Vector provenance attached to a forward document trace."""

    model_config = ConfigDict(frozen=True)

    vector_id: str
    collection_name: str
    dimension: int
    model_name: str
    model_version: str
    active: bool
    pii_masked_flag: bool


class ChunkLineageRecord(BaseModel):
    """Chunk and all vectors generated from it."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    chunk_index: int
    chunk_text: str
    chunk_hash: str
    strategy_name: str
    strategy_version: str
    pii_masked_flag: bool
    vectors: list[VectorLineageSummary]


class DocumentLineageRecord(BaseModel):
    """Forward provenance from one document version to its vectors."""

    model_config = ConfigDict(frozen=True)

    doc_id: str
    doc_version: int
    source_uri: str
    content_hash: str | None
    pii_masked_flag: bool
    chunks: list[ChunkLineageRecord]


class LineageService:
    """Resolve backward, forward, and stale-vector catalog lineage."""

    def __init__(self, catalog: CatalogDB) -> None:
        self.catalog = catalog
        self._vector_lineage_cache: dict[str, VectorLineageRecord] = {}

    def clear_cache(self) -> None:
        """Invalidate cached traces after catalog writes."""
        self._vector_lineage_cache.clear()

    def get_vector_lineage(self, vector_id: str) -> VectorLineageRecord:
        """Resolve vector -> model -> chunk -> document provenance."""
        cached = self._vector_lineage_cache.get(vector_id)
        if cached is not None:
            return cached
        rows = self.catalog.query(
            """
            SELECT
                v.vector_id, v.collection_name, v.dimension,
                v.model_name, v.model_version, em.provider,
                v.chunk_id, c.doc_id, c.doc_version, c.chunk_text,
                c.strategy_name, c.strategy_version,
                d.source_uri, d.content_hash,
                (v.pii_masked_flag OR c.pii_masked_flag OR d.pii_masked_flag)
            FROM vectors AS v
            JOIN embedding_models AS em
              ON em.model_name = v.model_name
             AND em.model_version = v.model_version
            JOIN chunks AS c ON c.chunk_id = v.chunk_id
            JOIN documents AS d
              ON d.doc_id = c.doc_id
             AND d.doc_version = c.doc_version
            WHERE v.vector_id = ?
            """,
            [vector_id],
        )
        if not rows:
            raise KeyError(f"Vector not found: {vector_id}")
        record = VectorLineageRecord.model_validate(
            dict(zip(VectorLineageRecord.model_fields, rows[0]))
        )
        self._vector_lineage_cache[vector_id] = record
        return record

    def get_document_lineage(self, doc_id: str, version: int) -> DocumentLineageRecord:
        """Resolve document -> chunks -> vectors for one document version."""
        document_rows = self.catalog.query(
            """
            SELECT doc_id, doc_version, source_uri, content_hash, pii_masked_flag
            FROM documents
            WHERE doc_id = ? AND doc_version = ?
            """,
            [doc_id, version],
        )
        if not document_rows:
            raise KeyError(f"Document not found: {doc_id} version {version}")

        chunk_rows = self.catalog.query(
            """
            SELECT
                c.chunk_id, c.chunk_index, c.chunk_text, c.chunk_hash,
                c.strategy_name, c.strategy_version, c.pii_masked_flag
            FROM chunks AS c
            WHERE c.doc_id = ? AND c.doc_version = ?
            ORDER BY c.chunk_index
            """,
            [doc_id, version],
        )
        chunks: list[ChunkLineageRecord] = []
        for chunk_row in chunk_rows:
            chunk_id = chunk_row[0]
            vector_rows = self.catalog.query(
                """
                SELECT
                    v.vector_id, v.collection_name, v.dimension,
                    v.model_name, v.model_version, v.active,
                    (v.pii_masked_flag OR c.pii_masked_flag OR d.pii_masked_flag)
                FROM vectors AS v
                JOIN chunks AS c ON c.chunk_id = v.chunk_id
                JOIN documents AS d
                  ON d.doc_id = c.doc_id AND d.doc_version = c.doc_version
                WHERE v.chunk_id = ?
                ORDER BY v.vector_id
                """,
                [chunk_id],
            )
            chunks.append(
                ChunkLineageRecord(
                    chunk_id=chunk_row[0],
                    chunk_index=chunk_row[1],
                    chunk_text=chunk_row[2],
                    chunk_hash=chunk_row[3],
                    strategy_name=chunk_row[4],
                    strategy_version=chunk_row[5],
                    pii_masked_flag=chunk_row[6],
                    vectors=[
                        VectorLineageSummary(
                            vector_id=row[0],
                            collection_name=row[1],
                            dimension=row[2],
                            model_name=row[3],
                            model_version=row[4],
                            active=row[5],
                            pii_masked_flag=row[6],
                        )
                        for row in vector_rows
                    ],
                )
            )

        document_row = document_rows[0]
        return DocumentLineageRecord(
            doc_id=document_row[0],
            doc_version=document_row[1],
            source_uri=document_row[2],
            content_hash=document_row[3],
            pii_masked_flag=document_row[4],
            chunks=chunks,
        )

    def get_stale_vectors(
        self,
        current_model_version: str,
        current_strategy_version: str | None = None,
    ) -> list[VectorLineageSummary]:
        """Return vectors outside the active model or strategy versions."""
        if current_strategy_version is None:
            strategy_rows = self.catalog.query(
                """
                SELECT strategy_version
                FROM chunk_strategies
                ORDER BY created_at DESC, strategy_version DESC
                LIMIT 1
                """
            )
            current_strategy_version = strategy_rows[0][0] if strategy_rows else ""

        rows = self.catalog.query(
            """
            SELECT
                v.vector_id, v.collection_name, v.dimension,
                v.model_name, v.model_version, v.active,
                (v.pii_masked_flag OR c.pii_masked_flag OR d.pii_masked_flag)
            FROM vectors AS v
            JOIN chunks AS c ON c.chunk_id = v.chunk_id
            JOIN documents AS d
              ON d.doc_id = c.doc_id AND d.doc_version = c.doc_version
            WHERE v.model_version <> ? OR c.strategy_version <> ?
            ORDER BY v.vector_id
            """,
            [current_model_version, current_strategy_version],
        )
        return [
            VectorLineageSummary(
                vector_id=row[0],
                collection_name=row[1],
                dimension=row[2],
                model_name=row[3],
                model_version=row[4],
                active=row[5],
                pii_masked_flag=row[6],
            )
            for row in rows
        ]


def main() -> None:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE_PATH)
    parser.add_argument("--trace-vector")
    parser.add_argument("--trace-document")
    args = parser.parse_args()
    if not args.trace_vector and not args.trace_document:
        parser.error("provide --trace-vector or --trace-document DOC_ID:VERSION")

    with CatalogDB(args.database) as catalog:
        service = LineageService(catalog)
        if args.trace_vector:
            result: Any = service.get_vector_lineage(args.trace_vector).model_dump(mode="json")
        else:
            doc_id, version = args.trace_document.rsplit(":", maxsplit=1)
            result = service.get_document_lineage(doc_id, int(version)).model_dump(mode="json")
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()