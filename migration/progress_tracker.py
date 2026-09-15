"""Progress checkpoints emitted while a shadow migration is running."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class MigrationProgress:
    migration_id: str
    total_vectors: int
    migrated_vectors: int
    completed_batches: int
    error_count: int
    chunks_per_second: float


ProgressCallback = Callable[[MigrationProgress], None]


class ProgressTracker:
    """Emit immutable progress snapshots to an optional consumer."""

    def __init__(self, callback: ProgressCallback | None = None) -> None:
        self.callback = callback
        self.checkpoints: list[MigrationProgress] = []

    def checkpoint(self, progress: MigrationProgress) -> None:
        self.checkpoints.append(progress)
        if self.callback is not None:
            self.callback(progress)