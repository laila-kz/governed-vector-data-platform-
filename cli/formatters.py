"""Rich formatting helpers for the gvpctl operator CLI."""

from __future__ import annotations

from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from migration.planner import MigrationPlan
from migration.progress_tracker import MigrationProgress

console = Console()


def render_status_table(status: dict[str, object]) -> None:
    table = Table(title="Governed Vector Platform Status")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in status.items():
        table.add_row(str(key), str(value))
    console.print(table)


def render_staleness_summary(summary: dict[str, object]) -> None:
    table = Table(title="Staleness Summary")
    table.add_column("Model")
    table.add_column("Active")
    table.add_column("Stale")
    table.add_column("Risk")
    for model, values in summary.items():
        active = values.get("active", 0)
        stale = values.get("stale", 0)
        risk = "low" if stale <= active else "high"
        table.add_row(str(model), str(active), str(stale), risk)
    console.print(table)


def render_lineage_trace(vector_id: str, trace: dict[str, object]) -> None:
    console.print(f"[bold]Vector[/bold]: {vector_id}")
    console.print("[cyan]Document -> Chunk -> Model -> Qdrant Index[/cyan]")
    for key, value in trace.items():
        console.print(f"{key}: {value}")


def render_migration_plan(plan: MigrationPlan) -> None:
    """Render a migration plan as an operator-friendly Terraform-style diff."""
    table = Table(title=f"Migration Plan {plan.migration_id}")
    table.add_column("Change")
    table.add_column("Value")
    table.add_row("model", f"{plan.from_model} -> {plan.to_model}")
    table.add_row("strategy", f"{plan.from_strategy} -> {plan.to_strategy}")
    table.add_row("vector count", str(plan.total_vectors))
    table.add_row("token volume", f"{plan.estimated_tokens:,}")
    table.add_row("estimated API cost", f"${plan.estimated_cost_usd:.6f}")
    table.add_row("estimated duration", f"{plan.estimated_duration_seconds:.2f}s")
    table.add_row("predicted retrieval drift", f"{plan.predicted_retrieval_drift:.4f}")
    table.add_row("status", plan.status)
    console.print(table)


def migration_progress() -> Progress:
    """Build the live progress display used by ``migrate apply``."""
    return Progress(
        SpinnerColumn(),
        TextColumn("{task.description}"),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        TextColumn("{task.fields[rate]:.2f} chunks/s"),
        TimeElapsedColumn(),
    )


def update_migration_progress(
    progress: Progress,
    batch_task: int,
    throughput_task: int,
    error_task: int,
    checkpoint: MigrationProgress,
) -> None:
    """Update the three live migration metrics from one checkpoint."""
    progress.update(batch_task, completed=checkpoint.completed_batches)
    progress.update(
        throughput_task,
        completed=checkpoint.migrated_vectors,
        rate=checkpoint.chunks_per_second,
    )
    progress.update(error_task, completed=checkpoint.error_count)
