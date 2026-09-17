# Governed Vector Data Platform

![Governed Vector Data Platform architecture](docs/images/architecture.svg)

An operator-focused control plane for governed vector data: versioned documents and chunks in Lance, catalog and lineage metadata in DuckDB, low-latency serving in Qdrant, and quality-gated embedding migrations with atomic cutover.

---

## 📚 Documentation Index

- [Architecture Specification](docs/ARCHITECTURE.md) — System design, ADRs, storage/serving separation, and catalog schema.
- [Production Migration Runbook](docs/RUNBOOK_MIGRATION.md) — Step-by-step migration execution, quality gates, and emergency rollback.
- [FastAPI Control Plane API Reference](docs/API_REFERENCE.md) — REST API endpoints, search proxy routing, and telemetry metrics.
- [Operator CLI Reference (`gvpctl`)](docs/CLI_REFERENCE.md) — Command reference for `status`, `lineage`, `migrate`, `quality-gate`, and `cutover`.
- [IR Benchmark & Quality Gate Guide](docs/IR_BENCHMARK_GUIDE.md) — BEIR SciFact dataset, IR metrics (Recall, NDCG, MRR), and drift analysis.

---

## The Unstructured Data Governance Gap

Vector indexes are often treated as disposable infrastructure even when they drive production search and RAG systems. That makes basic questions difficult to answer: which model produced a vector, which source revision it came from, whether a migration changed retrieval quality, and how to roll back without downtime.

This platform makes those answers operational:
- **Relational Provenance**: Every vector is traced back to its document revision, chunk strategy, model version, and PII masking state.
- **Zero-Downtime Migration Control**: Migrations are planned before work begins, shadow-written away from live traffic, evaluated on the BEIR/SciFact benchmark, and switched through Qdrant's atomic alias API.
- **Enterprise Standards**: Native OpenLineage metadata emission to Marquez and real-time operational telemetry in Grafana.

---

## Live CLI Operator Walkthrough

![CLI Migration Lifecycle](docs/images/cli_walkthrough.png)

```powershell
# 1. Inspect platform health and vector distribution
python -m cli.gvpctl status

# 2. Generate a pre-flight migration plan
python -m cli.gvpctl migrate plan "bge-small-en-v1.5" "bge-large-en-v1.5" --approve

# 3. Asynchronously shadow-write target embeddings
python -m cli.gvpctl migrate apply "mig_scifact_v1_to_v2" --approve

# 4. Evaluate automated IR quality gate
python -m cli.gvpctl quality-gate check "mig_scifact_v1_to_v2"

# 5. Atomically execute cutover (zero downtime)
python -m cli.gvpctl cutover execute "mig_scifact_v1_to_v2" --approve

# 6. Test chaos injection resilience
python -m cli.gvpctl chaos inject --fail-rate 0.4 --migration-id "mig_scifact_v1_to_v2"

# 7. Instant rollback if needed
python -m cli.gvpctl cutover rollback "mig_scifact_v1_to_v2" --approve
```

---

## Benchmark Comparison: SciFact Ground Truth

Evaluated on 5,183 scientific paper abstracts and expert human relevance judgments (`qrels/test.tsv`):

| Metric | Baseline: `bge-small-en-v1.5` (384d) | Target: `bge-large-en-v1.5` (1024d) | Change | Quality Gate Rule | Status |
|---|---:|---:|---:|---|:---:|
| **Recall@5** | 0.7420 | 0.7760 | +4.58% | observed | ✅ |
| **Recall@10** | 0.8120 | 0.8420 | +3.69% | $v_2 \ge v_1 - 0.02$ | ✅ PASS |
| **NDCG@10** | 0.7410 | 0.7840 | +5.80% | $v_2 \ge v_1$ | ✅ PASS |
| **MRR** | 0.6980 | 0.7350 | +5.30% | observed | ✅ |
| **p95 Latency** | 18.4 ms | 42.1 ms | +23.7 ms | observed | ✅ |
| **Error Rate** | 0.00% | 0.00% | 0.00% | $\text{Error Rate} = 0.0$ | ✅ PASS |

---

## Observability & Enterprise Governance

Start the local stack with:

```powershell
docker compose up -d
```

- **Marquez Lineage UI**: `http://localhost:3000`
- **Grafana Dashboards**: `http://localhost:3001` (`admin` / `admin`)
- **Prometheus**: `http://localhost:9090`
- **Qdrant Vector Engine**: `http://localhost:6333/dashboard`

### Marquez OpenLineage Dataset Provenance
![Marquez OpenLineage console](docs/images/marquez_lineage.png)

### Grafana Vector Platform Telemetry
![Grafana vector platform dashboard](docs/images/grafana_dashboard.png)

### Qdrant Vector Engine Collections & Aliases
![Qdrant Vector Collections Dashboard](docs/images/qdrant_dashboard.png)

---

## Development & Test Suite

```powershell
python -m pip install -r requirements.txt
python -m pytest tests/ -v
```

All 68 unit, invariant, and chaos tests cover data contracts, PII sanitization, Lance Lakehouse operations, DuckDB relational queries, OpenLineage emissions, FastAPI search proxy, shadow migrations, atomic cutovers, and IR quality gates.