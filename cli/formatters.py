"""Rich formatting helpers for the gvpctl operator CLI."""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

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
    table = Table(title=f"Lineage Trace: {vector_id}")
    table.add_column("Field")
    table.add_column("Value")
    for key, value in trace.items():
        table.add_row(str(key), str(value))
    console.print(table)
