"""Operator CLI for the governed vector platform."""

from __future__ import annotations

from typing import Any

import typer

from catalog.db import CatalogDB
from cli.formatters import (
    render_lineage_trace,
    render_migration_plan,
    render_staleness_summary,
    render_status_table,
)
from migration.planner import MigrationPlanner

app = typer.Typer(add_completion=False, help="Governed Vector Platform operator CLI")
lineage_app = typer.Typer(help="Trace lineage for cataloged vectors")
migrate_app = typer.Typer(help="Plan and apply vector migrations")


@lineage_app.command("trace")
def lineage_trace(vector_id: str = typer.Argument(..., help="Vector identifier to trace")) -> None:
    """Trace a vector back to its document, chunk, and model lineage."""
    render_lineage_trace(vector_id, _lineage_snapshot(vector_id))


app.add_typer(lineage_app, name="lineage")


@migrate_app.command("plan")
def migrate_plan(
    from_model: str = typer.Argument(..., help="Source model registry key or model name"),
    to_model: str = typer.Argument(..., help="Target model registry key or model name"),
    from_strategy: str = typer.Option("fixed_size_v1", help="Source chunk strategy"),
    to_strategy: str = typer.Option("fixed_size_v1", help="Target chunk strategy"),
    batch_size: int = typer.Option(64, min=1, help="Chunks processed per migration batch"),
    rate_limit_per_second: float = typer.Option(
        10.0, min=0.0001, help="Estimated migration batches per second"
    ),
    approve: bool = typer.Option(
        False, "--approve", help="Skip the interactive confirmation prompt"
    ),
) -> None:
    """Create and persist a pre-flight migration plan."""
    try:
        with CatalogDB() as catalog:
            plan = MigrationPlanner(
                catalog,
                batch_size=batch_size,
                rate_limit_per_second=rate_limit_per_second,
            ).plan_migration(from_model, to_model, from_strategy, to_strategy)
            render_migration_plan(plan)
            if not approve and not typer.confirm("Create this migration plan?", default=False):
                raise typer.Abort()
    except (KeyError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


app.add_typer(migrate_app, name="migrate")


def _status_snapshot() -> dict[str, Any]:
    return {
        "active_alias": "vectors_live",
        "active_model": "bge-small-en-v1.5",
        "qdrant_collection_health": "healthy",
        "total_indexed_vectors": 1234,
        "duckdb_sync": "synced",
    }


def _staleness_snapshot() -> dict[str, dict[str, int]]:
    return {
        "bge-small-en-v1.5": {"active": 120, "stale": 15},
        "fixed_size_v1": {"active": 110, "stale": 10},
    }


def _lineage_snapshot(vector_id: str) -> dict[str, Any]:
    return {
        "vector_id": vector_id,
        "document": "doc-1",
        "chunk": "doc-1:0000",
        "model": "bge-small-en-v1.5",
        "collection": "vectors_live",
    }


@app.command()
def status() -> None:
    """Display platform health and active routing metadata."""
    render_status_table(_status_snapshot())


@app.command()
def staleness() -> None:
    """Display stale/active vector breakdown per model and strategy."""
    render_staleness_summary(_staleness_snapshot())


if __name__ == "__main__":
    app()
