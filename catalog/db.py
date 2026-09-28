"""DuckDB connection manager and bulk metadata insertion helpers."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping

import duckdb

DEFAULT_DATABASE_PATH = Path("data/catalog.duckdb")
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("schema.sql")

TABLES = {
    "documents",
    "chunk_strategies",
    "chunks",
    "embedding_models",
    "vectors",
    "migrations",
    "migration_events",
    "retrieval_eval_runs",
}


class CatalogDB:
    """Own a DuckDB connection and initialize the catalog on construction."""

    def __init__(
        self,
        database_path: Path | str = DEFAULT_DATABASE_PATH,
        schema_path: Path | str = DEFAULT_SCHEMA_PATH,
    ) -> None:
        self.database_path = str(database_path)
        self.schema_path = Path(schema_path)
        if self.database_path != ":memory:":
            Path(self.database_path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = duckdb.connect(self.database_path)
        self.initialize()

    def initialize(self) -> None:
        """Apply the idempotent catalog schema to the active connection."""
        self.connection.execute(self.schema_path.read_text(encoding="utf-8"))
        for column, definition in (
            ("error_count", "INTEGER DEFAULT 0"),
            ("migration_id", "VARCHAR"),
            ("status", "VARCHAR DEFAULT 'completed'"),
            ("error_rate", "DOUBLE DEFAULT 0"),
            ("total_characters", "BIGINT DEFAULT 0"),
            ("estimated_tokens", "BIGINT DEFAULT 0"),
            ("estimated_cost_usd", "DOUBLE DEFAULT 0"),
            ("estimated_duration_seconds", "DOUBLE DEFAULT 0"),
            ("predicted_retrieval_drift", "DOUBLE DEFAULT 0"),
            ("batch_size", "INTEGER DEFAULT 64"),
            ("rate_limit_per_second", "DOUBLE DEFAULT 1"),
        ):
            self.connection.execute(
                f'ALTER TABLE migrations ADD COLUMN IF NOT EXISTS "{column}" {definition}'
            )

    @contextmanager
    def transaction(self) -> Iterator[duckdb.DuckDBPyConnection]:
        """Run a block atomically and roll back if it raises."""
        self.connection.execute("BEGIN TRANSACTION")
        try:
            yield self.connection
        except Exception:
            self.connection.execute("ROLLBACK")
            raise
        else:
            self.connection.execute("COMMIT")

    def insert_rows(
        self, table: str, rows: Iterable[Mapping[str, Any]]
    ) -> int:
        """Bulk insert mappings into a known catalog table."""
        if table not in TABLES:
            raise ValueError(f"Unknown catalog table: {table}")
        records = [dict(row) for row in rows]
        if not records:
            return 0
        columns = list(records[0])
        col_set = set(columns)
        if not columns or any(set(record) != col_set for record in records):
            raise ValueError("All rows must have the same columns")
        quoted_columns = ", ".join(f'"{column}"' for column in columns)
        import pyarrow as pa

        tbl = pa.Table.from_pylist(records)
        self.connection.execute(
            f'INSERT INTO "{table}" ({quoted_columns}) SELECT * FROM tbl'
        )
        return len(records)

    def insert_documents(self, rows: Iterable[Mapping[str, Any]]) -> int:
        return self.insert_rows("documents", rows)

    def insert_chunk_strategies(self, rows: Iterable[Mapping[str, Any]]) -> int:
        return self.insert_rows("chunk_strategies", rows)

    def insert_chunks(self, rows: Iterable[Mapping[str, Any]]) -> int:
        return self.insert_rows("chunks", rows)

    def insert_embedding_models(self, rows: Iterable[Mapping[str, Any]]) -> int:
        return self.insert_rows("embedding_models", rows)

    def insert_vectors(self, rows: Iterable[Mapping[str, Any]]) -> int:
        return self.insert_rows("vectors", rows)

    def insert_migrations(self, rows: Iterable[Mapping[str, Any]]) -> int:
        return self.insert_rows("migrations", rows)

    def insert_retrieval_eval_runs(self, rows: Iterable[Mapping[str, Any]]) -> int:
        return self.insert_rows("retrieval_eval_runs", rows)

    def query(self, sql: str, parameters: Iterable[Any] | None = None) -> list[tuple[Any, ...]]:
        """Execute a read query and return rows as tuples."""
        result = self.connection.execute(sql, parameters or ())
        return result.fetchall()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> "CatalogDB":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()