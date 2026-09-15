"""Operator CLI for the governed vector platform."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
import typer

from api.search_proxy import _load_policy
from catalog.db import DEFAULT_DATABASE_PATH, CatalogDB
from catalog.lineage_service import LineageService
from cli.formatters import (
    render_lineage_trace,
    render_migration_plan,
    render_staleness_summary,
    render_status_table,
)
from migration.planner import MigrationPlanner

APP_ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = APP_ROOT / "configs" / "routing_policy.yaml"

app = typer.Typer(add_completion=False, help="Governed Vector Platform operator CLI")
lineage_app = typer.Typer(help="Trace lineage for cataloged vectors")
migrate_app = typer.Typer(help="Plan and apply embedding migrations")


@lineage_app.command("trace")
def lineage_trace(vector_id: str = typer.Argument(..., help="Vector identifier to trace")) -> None:
    """Trace a vector back to its document, chunk, and model lineage."""
    render_lineage_trace(vector_id, _lineage_snapshot(vector_id))


app.add_typer(lineage_app, name="lineage")


@migrate_app.command("plan")
def migration_plan(
    from_model: str = typer.Argument(..., help="Source registry key or model name"),
    to_model: str = typer.Argument(..., help="Target registry key or model name"),
    from_strategy: str = typer.Option("fixed_size_v1", help="Source chunk strategy"),
    to_strategy: str = typer.Option("fixed_size_v1", help="Target chunk strategy"),
    approve: bool = typer.Option(False, "--approve", help="Approve the planned migration without prompting"),
) -> None:
    """Create and display a pre-flight migration plan."""
    with CatalogDB(DEFAULT_DATABASE_PATH) as catalog:
        plan = MigrationPlanner(catalog).plan_migration(
            from_model,
            to_model,
            from_strategy,
            to_strategy,
        )
    render_migration_plan(plan)
    if not approve and not typer.confirm("Approve this migration plan?", default=False):
        typer.echo(f"Migration {plan.migration_id} remains planned; approval was not granted.")
        return
    typer.echo(f"Migration {plan.migration_id} approved.")


app.add_typer(migrate_app, name="migrate")


def _status_snapshot() -> dict[str, Any]:
    policy = _load_policy(POLICY_PATH)
    try:
        with CatalogDB(DEFAULT_DATABASE_PATH) as catalog:
            total_vectors = int(catalog.query("SELECT COUNT(*) FROM vectors")[0][0])
            duckdb_sync = "synced"
    except Exception:
        total_vectors = 0
        duckdb_sync = "unavailable"

    qdrant_health = "not configured"
    if os.getenv("USE_REAL_QDRANT", "true").lower() == "true":
        try:
            QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333")).get_collections()
            qdrant_health = "healthy"
        except Exception:
            qdrant_health = "unavailable"
    return {
        "active_alias": policy["active_alias"],
        "active_model": policy["active_model"],
        "qdrant_collection_health": qdrant_health,
        "total_indexed_vectors": total_vectors,
        "duckdb_sync": duckdb_sync,
    }


def _staleness_snapshot() -> dict[str, dict[str, int]]:
    policy = _load_policy(POLICY_PATH)
    with CatalogDB(DEFAULT_DATABASE_PATH) as catalog:
        strategy_rows = catalog.query(
            "SELECT strategy_version FROM chunk_strategies ORDER BY created_at DESC, strategy_version DESC LIMIT 1"
        )
        current_strategy = str(strategy_rows[0][0]) if strategy_rows else None
        stale_condition = "v.model_name <> ? OR v.model_version <> ?"
        parameters: list[Any] = [policy["active_model"], policy["active_version"]]
        if current_strategy is not None:
            stale_condition += " OR c.strategy_version <> ?"
            parameters.append(current_strategy)
        rows = catalog.query(
            f"""
            SELECT v.model_name, COUNT(*) FILTER (WHERE {stale_condition}), COUNT(*)
            FROM vectors AS v
            JOIN chunks AS c ON c.chunk_id = v.chunk_id
            GROUP BY v.model_name
            ORDER BY v.model_name
            """,
            parameters,
        )
    return {
        str(model): {"active": int(total - stale), "stale": int(stale)}
        for model, stale, total in rows
    }


def _lineage_snapshot(vector_id: str) -> dict[str, Any]:
    with CatalogDB(DEFAULT_DATABASE_PATH) as catalog:
        try:
            return LineageService(catalog).get_vector_lineage(vector_id).model_dump(mode="json")
        except KeyError:
            return {"vector_id": vector_id, "status": "not found"}


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
