# Migration Runbook

## Prerequisites

```powershell
python -m pip install -r requirements.txt
docker compose up -d qdrant prometheus grafana postgres marquez marquez-web
```

Confirm Qdrant responds at `http://localhost:6333` before applying a migration. Marquez and Grafana are optional for the migration itself but required for the portfolio observability captures.

## Plan and Apply

```powershell
python -m cli.gvpctl migrate plan "bge-small-en-v1.5" "bge-large-en-v1.5" --approve
python -m cli.gvpctl migrate apply "<migration_id>" --approve
```

The apply command performs shadow writing, records progress, and runs the quality gate automatically. The gate evaluates the baseline and shadow collections against the 300 qrel-bearing SciFact queries currently materialized in the repository.

## Quality Gate

```powershell
python -m cli.gvpctl quality-gate check "<migration_id>"
```

The command records two `retrieval_eval_runs` rows. It fails when Recall@10 regresses by more than 0.02, NDCG@10 decreases, or either collection has retrieval errors. A failed check must be investigated before any cutover.

## Cutover and Rollback

```powershell
python -m cli.gvpctl cutover execute "<migration_id>" --approve
python -m cli.gvpctl cutover rollback "<migration_id>" --approve
```

Cutover changes the Qdrant alias and catalog activation state as one operator workflow. Rollback points the alias back to `scifact_v1`, restores source vectors as active, updates the routing policy, and records a migration event.

## Chaos Demonstration

```powershell
python -m cli.gvpctl chaos inject --fail-rate 0.4 --migration-id "<migration_id>"
```

The chaos runner simulates HTTP 429 responses, dropped connections, and malformed payloads. It verifies retry backoff, circuit breaking, unchanged active vectors, and unchanged Lance content.

## Portfolio Captures

- [demo_walkthrough.gif](images/demo_walkthrough.gif) is the terminal-style lifecycle walkthrough.
- [marquez_lineage.png](images/marquez_lineage.png) records the Marquez console. Select the `governed-vector-platform` namespace in the Jobs view to inspect the `batch_embedding` run; the UI defaults to the `dataset` namespace.
- [grafana_dashboard.png](images/grafana_dashboard.png) records the provisioned Grafana dashboard when the local stack is available.

Do not present placeholder or synthetic screenshots as live operational evidence. Re-capture the two service images after OpenLineage events and Prometheus metrics are available in the local stack.