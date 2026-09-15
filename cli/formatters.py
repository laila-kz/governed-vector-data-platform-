"""Rich formatting helpers for the gvpctl operator CLI."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

console = Console()


def render_migration_plan(plan: object) -> None:
    """Render a migration plan as an operator-friendly Terraform-style diff."""
    table = Table(title=f"Migration Plan: {plan.migration_id}")
    table.add_column("Metric")
    table.add_column("Current")
    table.add_column("Planned")
    table.add_row("Model", f"{plan.from_model} ({plan.from_model_version})", f"{plan.to_model} ({plan.to_model_version})")
    table.add_row("Strategy", plan.from_strategy, plan.to_strategy)
    table.add_row("Vectors", str(plan.total_vectors), str(plan.total_vectors))
    table.add_row("Characters", str(plan.total_characters), str(plan.total_characters))
    table.add_row("Estimated tokens", "-", str(plan.estimated_tokens))
    table.add_row("Estimated cost (USD)", "-", f"{plan.estimated_cost_usd:.6f}")
    table.add_row("Estimated duration (s)", "-", f"{plan.estimated_duration_seconds:.2f}")
    table.add_row("Predicted retrieval drift", "-", f"{plan.predicted_retrieval_drift:.4f}")
    table.add_row("Status", "-", plan.status)
    console.print(table)


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
    table = Table(title=f"Lineage Trace: {vector_id}")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in trace.items():
        table.add_row(str(key), str(value))
    console.print(table)
