# Operator CLI Reference (`gvpctl`)

The `gvpctl` command-line utility provides Kubernetes/Terraform-style controls for inspecting platform health, tracing vector provenance, planning and applying shadow migrations, running quality gates, executing atomic cutovers, and injecting chaos.

---

## 1. Global Commands

### `gvpctl status`
Displays real-time platform health, active alias, active model, and total indexed vectors.

```powershell
python -m cli.gvpctl status
```

### `gvpctl staleness`
Displays active vs. stale vector counts broken down by embedding model and chunking strategy.

```powershell
python -m cli.gvpctl staleness
```

---

## 2. Lineage Subcommands (`gvpctl lineage`)

### `gvpctl lineage trace <vector_id>`
Traces a vector ID back through its embedding model, chunk text snippet, and source document version.

```powershell
python -m cli.gvpctl lineage trace "vec_scifact_0001"
```

---

## 3. Migration Subcommands (`gvpctl migrate`)

### `gvpctl migrate plan <from_model> <to_model>`
Generates a pre-flight migration plan calculating vector counts, token volume, estimated cost ($USD), runtime, and predicted vector drift.

**Options**:
- `--from-strategy`: Source chunking strategy (default: `fixed_size_v1`).
- `--to-strategy`: Target chunking strategy (default: `fixed_size_v1`).
- `--batch-size`: Batch size for shadow migration (default: `64`).
- `--approve`: Skip interactive confirmation prompt.

```powershell
python -m cli.gvpctl migrate plan "bge-small-en-v1.5" "bge-large-en-v1.5" --approve
```

### `gvpctl migrate apply <migration_id>`
Executes shadow writing into Qdrant collection `scifact_v2_shadow`, displaying live Rich progress bars, and automatically runs the quality gate upon completion.

**Options**:
- `--approve`: Skip interactive confirmation prompt.
- `--skip-quality-gate`: Skip automatic post-migration quality gate.
- `--qdrant-url`: Qdrant endpoint (default: `http://localhost:6333`).

```powershell
python -m cli.gvpctl migrate apply "mig_scifact_v1_to_v2" --approve
```

---

## 4. Quality Gate Subcommands (`gvpctl quality-gate`)

### `gvpctl quality-gate check <migration_id>`
Evaluates baseline vs. shadow collections against the BEIR SciFact ground-truth dataset.

```powershell
python -m cli.gvpctl quality-gate check "mig_scifact_v1_to_v2"
```

---

## 5. Cutover Subcommands (`gvpctl cutover`)

### `gvpctl cutover execute <migration_id>`
Atomically swaps `vectors_live` to the shadow collection and updates catalog state.

```powershell
python -m cli.gvpctl cutover execute "mig_scifact_v1_to_v2" --approve
```

### `gvpctl cutover rollback <migration_id>`
Atomically points `vectors_live` back to the baseline collection and restores active catalog state.

```powershell
python -m cli.gvpctl cutover rollback "mig_scifact_v1_to_v2" --approve
```

---

## 6. Chaos Subcommands (`gvpctl chaos`)

### `gvpctl chaos inject`
Injects transient HTTP 429 throttling, connection drops, and payload corruption during migration execution.

**Options**:
- `--fail-rate`: Fault injection probability per batch (default: `0.25`).
- `--migration-id`: Target migration ID.

```powershell
python -m cli.gvpctl chaos inject --fail-rate 0.4 --migration-id "mig_scifact_v1_to_v2"
```
