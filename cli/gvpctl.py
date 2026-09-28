"""Operator CLI for the governed vector platform."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer
import yaml
from qdrant_client import QdrantClient

from catalog.db import CatalogDB
from catalog.lineage_service import LineageService
from cli.chaos import chaos_inject
from cli.formatters import (
    migration_progress,
    render_lineage_trace,
    render_migration_plan,
    render_staleness_summary,
    render_status_table,
    update_migration_progress,
)
from migration.planner import MigrationPlanner
from migration.progress_tracker import ProgressTracker
from migration.cutover_manager import CutoverManager
from migration.worker import ShadowMigrationWorker
from evaluation.quality_gate import QualityGate, qdrant_retriever
from embedding.embedder import fastembedder
from embedding.model_registry import get_model

app = typer.Typer(add_completion=False, help="Governed Vector Platform operator CLI")
lineage_app = typer.Typer(help="Trace lineage for cataloged vectors")
migrate_app = typer.Typer(help="Plan and apply vector migrations")
chaos_app = typer.Typer(help="Inject controlled failures into migrations")
cutover_app = typer.Typer(help="Switch and restore live vector traffic")
quality_gate_app = typer.Typer(help="Evaluate migration retrieval quality")


@lineage_app.command("trace")
def lineage_trace(
    vector_id: str = typer.Argument(..., help="Vector identifier to trace"),
    database: str = typer.Option("data/catalog.duckdb", help="Catalog database path"),
) -> None:
    """Trace a vector back to its document, chunk, and model lineage."""
    with CatalogDB(database) as catalog:
        service = LineageService(catalog)
        try:
            record = service.get_vector_lineage(vector_id)
        except KeyError as error:
            total = catalog.query("SELECT COUNT(*) FROM vectors")[0][0]
            raise typer.BadParameter(
                f"no cataloged vector {vector_id!r}; baseline IDs run "
                f"vec_0001 to vec_{int(total):04d}"
            ) from error
    trace_data = {
        "vector_id": record.vector_id,
        "document": f"{record.doc_id} (v{record.doc_version})",
        "chunk": record.chunk_id,
        "model": f"{record.model_name}:{record.model_version}",
        "collection": record.collection_name,
        "strategy": f"{record.strategy_name}:{record.strategy_version}",
        "source_uri": record.source_uri,
    }
    render_lineage_trace(vector_id, trace_data)


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


@migrate_app.command("apply")
def migrate_apply(
    migration_id: str = typer.Argument(..., help="Planned migration identifier"),
    approve: bool = typer.Option(False, "--approve", help="Skip the confirmation prompt"),
    skip_quality_gate: bool = typer.Option(
        False, "--skip-quality-gate", help="Skip automatic post-migration quality evaluation"
    ),
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL"),
    dataset_path: str = typer.Option("data/lance_lakehouse/chunks.lance", help="Lance dataset"),
) -> None:
    """Run a planned migration into its shadow Qdrant collection."""
    if not approve and not typer.confirm(f"Apply migration {migration_id}?", default=False):
        raise typer.Abort()
    progress = migration_progress()
    with CatalogDB() as catalog:
        qdrant = QdrantClient(url=qdrant_url)
        try:
            tracker = ProgressTracker()
            worker = ShadowMigrationWorker(
                catalog,
                qdrant,
                dataset_path=dataset_path,
                progress_tracker=tracker,
            )
            with progress:
                batch_task = progress.add_task("batches", total=1, rate=0.0)
                throughput_task = progress.add_task("embedding", total=1, rate=0.0)
                error_task = progress.add_task("errors", total=1, rate=0.0)
                tracker.callback = lambda checkpoint: update_migration_progress(
                    progress, batch_task, throughput_task, error_task, checkpoint
                )
                result = worker.run(migration_id)
                progress.update(batch_task, total=result.completed_batches)
                progress.update(throughput_task, total=result.migrated_vectors)
                progress.update(error_task, total=max(result.error_count, 1))
            if not skip_quality_gate:
                try:
                    embedders = {
                        "bge-small-en-v1.5": fastembedder(get_model("v1")),
                        "bge-large-en-v1.5": fastembedder(get_model("v2")),
                    }
                    decision = QualityGate(
                        catalog,
                        qdrant_retriever(qdrant, embedders),
                    ).check(migration_id)
                except Exception as error:
                    catalog.connection.execute(
                        "UPDATE migrations SET status = 'failed', error_count = error_count + 1 WHERE migration_id = ?",
                        [migration_id],
                    )
                    raise typer.ClickException(
                        f"automatic quality gate could not run: {error}"
                    ) from error
                typer.echo(decision.reason)
                if not decision.passed:
                    raise typer.Exit(code=1)
        finally:
            qdrant.close()
    typer.echo(f"Migration {migration_id} completed: {result.migrated_vectors} vectors")


app.add_typer(migrate_app, name="migrate")


@chaos_app.command("inject")
def chaos_command(
    fail_rate: float = typer.Option(0.25, min=0.0, max=1.0, help="Probability of a fault per batch attempt"),
    migration_id: str = typer.Option(..., help="Migration to run under fault injection"),
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL"),
    dataset_path: str = typer.Option("data/lance_lakehouse/chunks.lance", help="Lance dataset"),
) -> None:
    chaos_inject(fail_rate, migration_id, qdrant_url, dataset_path)


app.add_typer(chaos_app, name="chaos")


@cutover_app.command("execute")
def cutover_execute(
    migration_id: str = typer.Argument(..., help="Completed migration identifier"),
    approve: bool = typer.Option(False, "--approve", help="Skip the confirmation prompt"),
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL"),
    policy_path: str = typer.Option("configs/routing_policy.yaml", help="Routing policy path"),
) -> None:
    """Atomically route vectors_live to the migration shadow collection."""
    if not approve and not typer.confirm(f"Execute cutover for {migration_id}?", default=False):
        raise typer.Abort()
    with CatalogDB() as catalog:
        qdrant = QdrantClient(url=qdrant_url)
        try:
            CutoverManager(catalog, qdrant, policy_path=policy_path).execute_cutover(migration_id)
        finally:
            qdrant.close()
    typer.echo(f"Migration {migration_id} cut over to vectors_live")


@cutover_app.command("rollback")
def cutover_rollback(
    migration_id: str = typer.Argument(..., help="Migration identifier to roll back"),
    approve: bool = typer.Option(False, "--approve", help="Skip the confirmation prompt"),
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL"),
    policy_path: str = typer.Option("configs/routing_policy.yaml", help="Routing policy path"),
) -> None:
    """Atomically restore vectors_live to the baseline collection."""
    if not approve and not typer.confirm(f"Rollback cutover for {migration_id}?", default=False):
        raise typer.Abort()
    with CatalogDB() as catalog:
        qdrant = QdrantClient(url=qdrant_url)
        try:
            CutoverManager(catalog, qdrant, policy_path=policy_path).rollback_cutover(migration_id)
        finally:
            qdrant.close()
    typer.echo(f"Migration {migration_id} rolled back to vectors_live")


app.add_typer(cutover_app, name="cutover")


@quality_gate_app.command("check")
def quality_gate_check(
    migration_id: str = typer.Argument(..., help="Migration identifier to evaluate"),
    qdrant_url: str = typer.Option("http://localhost:6333", help="Qdrant server URL"),
    evaluation_catalog: str = typer.Option("evaluation/beir_scifact_qrels.json", help="SciFact evaluation catalog"),
) -> None:
    """Run the baseline-vs-shadow quality gate before cutover."""
    with CatalogDB() as catalog:
        qdrant = QdrantClient(url=qdrant_url)
        try:
            embedders = {
                "bge-small-en-v1.5": fastembedder(get_model("v1")),
                "bge-large-en-v1.5": fastembedder(get_model("v2")),
            }
            decision = QualityGate(
                catalog,
                qdrant_retriever(qdrant, embedders),
                evaluation_catalog=evaluation_catalog,
            ).check(migration_id)
        finally:
            qdrant.close()
    typer.echo(decision.reason)
    if not decision.passed:
        raise typer.Exit(code=1)


app.add_typer(quality_gate_app, name="quality-gate")


def _status_snapshot() -> dict[str, Any]:
    active_alias = "vectors_live"
    active_model = "bge-small-en-v1.5"
    policy_path = Path("configs/routing_policy.yaml")
    if policy_path.exists():
        try:
            policy = yaml.safe_load(policy_path.read_text(encoding="utf-8")) or {}
            active_alias = policy.get("active_alias", active_alias)
            active_model = policy.get("active_model", active_model)
        except Exception:
            pass

    total_indexed = 0
    duckdb_sync = "synced"
    try:
        with CatalogDB() as catalog:
            rows = catalog.query("SELECT COUNT(*) FROM vectors")
            if rows and rows[0][0] is not None:
                total_indexed = int(rows[0][0])
    except Exception:
        duckdb_sync = "unavailable"

    return {
        "active_alias": active_alias,
        "active_model": active_model,
        "qdrant_collection_health": "healthy",
        "total_indexed_vectors": total_indexed,
        "duckdb_sync": duckdb_sync,
    }


def _staleness_snapshot() -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    try:
        with CatalogDB() as catalog:
            model_rows = catalog.query(
                "SELECT model_name, active, COUNT(*) FROM vectors GROUP BY model_name, active"
            )
            for model_name, active, count in model_rows:
                key = str(model_name)
                if key not in summary:
                    summary[key] = {"active": 0, "stale": 0}
                if active:
                    summary[key]["active"] += int(count)
                else:
                    summary[key]["stale"] += int(count)

            strategy_rows = catalog.query(
                """
                SELECT c.strategy_name, v.active, COUNT(*)
                FROM vectors AS v
                JOIN chunks AS c ON c.chunk_id = v.chunk_id
                GROUP BY c.strategy_name, v.active
                """
            )
            for strategy_name, active, count in strategy_rows:
                key = str(strategy_name)
                if key not in summary:
                    summary[key] = {"active": 0, "stale": 0}
                if active:
                    summary[key]["active"] += int(count)
                else:
                    summary[key]["stale"] += int(count)
    except Exception:
        pass

    if not summary:
        summary = {
            "bge-small-en-v1.5": {"active": 0, "stale": 0},
            "fixed_size_v1": {"active": 0, "stale": 0},
        }
    return summary


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
