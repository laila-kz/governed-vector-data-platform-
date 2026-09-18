# 🎬 Governed Vector Data Platform — Demo Video Guide

> **Purpose**: A precise, step-by-step script to run and film the full end-to-end demo.  
> **Total estimated runtime**: ~10–15 minutes of recording.

---

## 🗺️ Demo Story Arc

| # | Scene | What you show |
|---|-------|---------------|
| 1 | Environment setup | Docker services starting, FastAPI launching |
| 2 | Data ingestion | Downloading SciFact, bootstrapping baseline |
| 3 | Platform status | `gvpctl status`, vector counts, live alias |
| 4 | Lineage trace | `gvpctl lineage trace` — backward provenance |
| 5 | Migration plan | Pre-flight cost/drift analysis table |
| 6 | Shadow writing | Live Rich progress bar filling `scifact_v2_shadow` |
| 7 | Quality gate | IR metrics table — Recall, NDCG, MRR |
| 8 | Atomic cutover | Zero-downtime alias swap |
| 9 | Observability | Qdrant dashboard → Marquez lineage → Grafana |
| 10 | Chaos + rollback | Fault injection, instant rollback |

---

## ✅ Pre-Demo Checklist

Run through this once **before** hitting record.

- [ ] Docker Desktop is running
- [ ] Terminal is open at the project root (`Governed Vector Data Platform/`)
- [ ] `.venv` is activated (`& .venv\Scripts\Activate.ps1`)
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] Ports 6333, 3000, 3001, 5000, 9090, 8000 are free
- [ ] Terminal font size bumped up (≥14pt) for screen readability

---

## 📁 Terminal Setup

Open PowerShell and navigate to the project root:

```powershell
cd "C:\Users\kheza\Desktop\Data Engineering\Governed Vector Data Platform"
& .venv\Scripts\Activate.ps1
```

> 💡 **Film tip**: Keep browser + terminal side by side. Terminal on the left (CLI commands), browser on the right (UIs). Pre-open all browser tabs before recording.

---

## Scene 1 — Start the Infrastructure Stack

**What to say**: *"The platform runs on five containerised services. One command brings the entire stack up."*

```powershell
docker compose up -d
```

Wait ~20 seconds, then verify all services are healthy:

```powershell
docker compose ps
```

Expected — all services show `Up` or `healthy`:

| Container | Status |
|---|---|
| `gvp_qdrant` | Up (healthy) |
| `gvp_postgres` | Up (healthy) |
| `gvp_marquez` | Up |
| `gvp_marquez_web` | Up |
| `gvp_prometheus` | Up |
| `gvp_grafana` | Up |

**Then open your browser and briefly show each UI (~15 seconds each):**

| Service | URL |
|---|---|
| Qdrant Dashboard | http://localhost:6333/dashboard |
| Marquez Lineage UI | http://localhost:3000 |
| Grafana Dashboards | http://localhost:3001 (`admin` / `admin`) |
| Prometheus | http://localhost:9090 |

---

## Scene 2 — Start the FastAPI Control Plane

**What to say**: *"The FastAPI control plane is the operational brain — it routes search traffic, exposes lineage endpoints, and publishes Prometheus telemetry."*

Open a **second terminal tab** and run:

```powershell
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Switch to browser and confirm it is live:

```
http://localhost:8000/health
```

Expected JSON response:

```json
{
  "status": "healthy",
  "active_alias": "vectors_live",
  "active_model": "bge-small-en-v1.5",
  "active_version": "1.0.0"
}
```

Also show the auto-generated API docs:

```
http://localhost:8000/docs
```

> 💡 Leave FastAPI running in its own tab for the rest of the demo. Switch back to your main terminal for all CLI commands.

---

## Scene 3 — Download the SciFact Dataset

**What to say**: *"The platform is evaluated against the BEIR SciFact benchmark — 5,183 scientific paper abstracts with expert relevance judgments."*

> ⚠️ **Skip this scene** if `data/raw/scifact/corpus.jsonl` already exists.

In your main terminal:

```powershell
python -m ingestion.download_scifact
```

Expected output:

```
saved data/raw/scifact/corpus.jsonl (X bytes)
saved data/raw/scifact/queries.jsonl (X bytes)
saved data/raw/scifact/qrels/test.tsv (X bytes)
```

---

## Scene 4 — Bootstrap the Baseline

**What to say**: *"Bootstrap validates all documents, writes them to Lance columnar storage, materialises a fully governed DuckDB catalog, and indexes embeddings into Qdrant — all in one command."*

```powershell
python -m ingestion.bootstrap
```

What happens under the hood:
1. Documents validated; malformed ones quarantined
2. Sanitised text persisted to Lance (versioned columnar store)
3. Document, chunk, model, and vector provenance written to `data/catalog.duckdb`
4. All 15,240 chunks embedded with `bge-small-en-v1.5` (384-dim) via FastEmbed
5. Vectors indexed into the `scifact_v1` Qdrant collection

Expected JSON summary printed at completion:

```json
{
  "valid_document_count": 5183,
  "invalid_document_count": 0,
  "catalog_document_count": 5183,
  "catalog_chunk_count": 15240,
  "vector_count": 15240,
  "collection": "scifact_v1",
  "model_name": "bge-small-en-v1.5",
  "model_version": "1.0.0"
}
```

> ⚠️ **First run only** — takes 1–3 minutes. FastEmbed downloads the model (~120 MB) on first use. The script is idempotent and skips reinsertion if the catalog already exists.

**Switch to the Qdrant dashboard** and show the `scifact_v1` collection with 15,240 vectors.

---

## Scene 5 — Platform Status Check

**What to say**: *"The `gvpctl status` command gives an operator an instant, real-time health snapshot."*

```powershell
python -m cli.gvpctl status
```

Expected Rich table:

```
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field            ┃ Value                            ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ Active Alias     │ vectors_live                     │
│ Active Model     │ bge-small-en-v1.5                │
│ Total Vectors    │ 15,240                           │
│ Qdrant Status    │ green                            │
└──────────────────┴──────────────────────────────────┘
```

Follow up with the staleness report:

```powershell
python -m cli.gvpctl staleness
```

---

## Scene 6 — Vector Lineage Trace

**What to say**: *"Every vector carries a complete backward provenance chain — from the Qdrant point ID all the way back to the raw source document, PII masking state, chunk strategy, and model version."*

```powershell
python -m cli.gvpctl lineage trace "vec_scifact_0001"
```

Expected Rich output:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Field                    ┃ Value                                            ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ vector_id                │ vec_scifact_0001                                 │
│ model_name               │ bge-small-en-v1.5                                │
│ model_version            │ 1.0.0                                            │
│ chunk_text               │ Alternative splicing generates functional...     │
│ strategy_name            │ fixed_size_v1                                    │
│ source_uri               │ s3://scifact-corpus/raw/doc_487.json             │
│ doc_version              │ 1                                                │
│ pii_masked_flag          │ True                                             │
└──────────────────────────┴──────────────────────────────────────────────────┘
```

**Also show via REST API in browser:**

```
http://localhost:8000/v1/catalog/lineage/vec_scifact_0001
```

---

## Scene 7 — Pre-Flight Migration Plan

**What to say**: *"Before any migration touches live infrastructure, the operator generates a pre-flight plan. It calculates vector counts, token volume, estimated cost, runtime, and predicted retrieval drift — all before a single embedding is computed."*

```powershell
python -m cli.gvpctl migrate plan "bge-small-en-v1.5" "bge-large-en-v1.5" --approve
```

Expected Rich output:

```
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Change                    ┃ Value                                    ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ model                     │ bge-small-en-v1.5 -> bge-large-en-v1.5  │
│ strategy                  │ fixed_size_v1 -> fixed_size_v1           │
│ vector count              │ 15,240                                   │
│ token volume              │ 1,524,000                                │
│ estimated API cost        │ $0.000000                                │
│ estimated duration        │ 45.20s                                   │
│ predicted retrieval drift │ 0.1824                                   │
│ status                    │ planned                                  │
└───────────────────────────┴──────────────────────────────────────────┘
Migration ID: mig_scifact_v1_to_v2
```

> 📝 **Note the migration ID** — you'll use it in every subsequent command. It is always `mig_scifact_v1_to_v2`.

---

## Scene 8 — Asynchronous Shadow Writing

**What to say**: *"The migration worker shadow-writes all target embeddings into a completely isolated `scifact_v2_shadow` collection. Live search traffic is totally unaffected — no downtime, no traffic diversion."*

```powershell
python -m cli.gvpctl migrate apply "mig_scifact_v1_to_v2" --approve
```

You will see:
- A Rich progress bar tracking batches (e.g. `Embedding batch 45/238`)
- Live ETA countdown
- OpenLineage events being emitted to Marquez in the logs

**While it runs, switch to the browser and show:**
1. **Qdrant dashboard** → a new `scifact_v2_shadow` collection appearing and filling up
2. **Marquez UI** → new lineage datasets appearing at http://localhost:3000

> ⚠️ This takes ~45 seconds. Let it complete fully before moving to the next scene.

---

## Scene 9 — Automated Quality Gate

**What to say**: *"Once shadow writing completes, the quality gate automatically evaluates both collections against the BEIR ground-truth benchmark. It enforces hard IR thresholds before any cutover is permitted."*

The gate runs automatically at the end of `migrate apply`. You can also run it explicitly:

```powershell
python -m cli.gvpctl quality-gate check "mig_scifact_v1_to_v2"
```

Expected output:

```
Quality Gate Status: PASSED ✅ — Cutover Authorized

┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━┓
┃ Metric        ┃ Baseline (v1)  ┃ Target (v2)    ┃ Δ Change  ┃ Gate    ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━╇━━━━━━━━━┩
│ Recall@10     │ 0.812          │ 0.842          │ +3.0%     │ ✅ PASS  │
│ NDCG@10       │ 0.741          │ 0.784          │ +4.3%     │ ✅ PASS  │
│ MRR           │ 0.698          │ 0.735          │ +3.7%     │ ✅ PASS  │
│ Error Rate    │ 0.00           │ 0.00           │ —         │ ✅ PASS  │
└───────────────┴────────────────┴────────────────┴───────────┴─────────┘
```

**Explain the gates on camera:**
- `Recall@10(v2) ≥ Recall@10(v1) - 0.02` — prevents recall regression
- `NDCG@10(v2) ≥ NDCG@10(v1)` — enforces ranking quality
- `Error Rate = 0.0` — zero tolerance for failed embeddings

---

## Scene 10 — Zero-Downtime Atomic Cutover

**What to say**: *"The cutover is a single atomic alias swap in Qdrant. `vectors_live` flips from `scifact_v1` to `scifact_v2_shadow` in under 10ms. No downtime. No reindex."*

```powershell
python -m cli.gvpctl cutover execute "mig_scifact_v1_to_v2" --approve
```

Expected output:

```
✅ Cutover complete.
   vectors_live → scifact_v2_shadow
   Catalog updated: 15,240 vectors marked active.
```

**Switch to Qdrant dashboard** and show:
- The `vectors_live` alias now points to `scifact_v2_shadow`
- `bge-large-en-v1.5` (1024-dim) is the active model

Confirm with a fresh status check:

```powershell
python -m cli.gvpctl status
```

```
│ Active Model  │ bge-large-en-v1.5  │
│ Total Vectors │ 15,240             │
```

---

## Scene 11 — Live Vector Search via API

**What to say**: *"Search queries are now automatically routed to the upgraded model. The API transparently handles embedding, routing, and optional shadow read comparison."*

Open the Swagger UI at http://localhost:8000/docs, expand **POST /v1/search**, click **Try it out**, and submit:

```json
{
  "query": "What are the molecular mechanisms of CRISPR-Cas9 genome editing?",
  "top_k": 5
}
```

Or run in the terminal:

```powershell
Invoke-RestMethod -Method POST -Uri "http://localhost:8000/v1/search" `
  -ContentType "application/json" `
  -Body '{"query": "CRISPR-Cas9 genome editing mechanisms", "top_k": 5}'
```

---

## Scene 12 — Grafana Telemetry Dashboard

**What to say**: *"Every operation is instrumented. Grafana shows real-time search latencies, shadow read deltas, and stale vector ratios — all scraped from the FastAPI Prometheus endpoint."*

Open: http://localhost:3001 (admin / admin)

Navigate to the **Vector Platform Telemetry** dashboard and show:
- Search request latency (p95)
- Stale vector percentage gauge
- Shadow read latency delta (live vs shadow)

Also show the raw Prometheus metrics exposition:

```
http://localhost:8000/metrics
```

---

## Scene 13 — Marquez OpenLineage Provenance

**What to say**: *"Every migration emits OpenLineage dataset facets to Marquez — enterprise-grade data lineage showing exactly which document version, chunk strategy, and model produced every vector in production."*

Open: http://localhost:3000

Navigate to **Namespaces → governed-vector-platform** and show:
- The lineage graph: `scifact_v1` → migration job → `scifact_v2_shadow`
- Facet details: model name, version, dimensions, provider

---

## Scene 14 — Chaos Fault Injection

**What to say**: *"Enterprise resilience means surviving turbulent network conditions. The chaos runner injects HTTP 429 throttling, connection drops, and corrupted payloads during a migration run to validate exponential backoff and catalog rollbacks."*

```powershell
python -m cli.gvpctl chaos inject --fail-rate 0.4 --migration-id "mig_scifact_v1_to_v2"
```

Watch for:
- Injected faults logged in real time
- Automatic exponential backoff retries
- Catalog transaction rollbacks on terminal failure
- Final resilience report

---

## Scene 15 — Instant Rollback

**What to say**: *"If anything goes wrong post-cutover, rollback is a single command. It atomically swaps `vectors_live` back to `scifact_v1` — under 10ms, zero data loss."*

```powershell
python -m cli.gvpctl cutover rollback "mig_scifact_v1_to_v2" --approve
```

Expected output:

```
✅ Rollback complete.
   vectors_live → scifact_v1
   Catalog restored: bge-small-en-v1.5 (384d) active.
```

Confirm:

```powershell
python -m cli.gvpctl status
```

---

## Scene 16 — Closing: Test Suite (Optional)

Run the test suite to show platform stability:

```powershell
python -m pytest tests/ -v --tb=short
```

Expected result:

```
73 passed, 0 failed — 90% coverage
```

---

## 🔁 Full Command Sequence — Quick Reference

```powershell
# ── Infrastructure ──────────────────────────────────────────────────────────
docker compose up -d
# In a separate tab:
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# ── Data Ingestion (first run only) ─────────────────────────────────────────
python -m ingestion.download_scifact
python -m ingestion.bootstrap

# ── Platform Inspection ──────────────────────────────────────────────────────
python -m cli.gvpctl status
python -m cli.gvpctl staleness
python -m cli.gvpctl lineage trace "vec_scifact_0001"

# ── Full Migration Lifecycle ──────────────────────────────────────────────────
python -m cli.gvpctl migrate plan "bge-small-en-v1.5" "bge-large-en-v1.5" --approve
python -m cli.gvpctl migrate apply "mig_scifact_v1_to_v2" --approve
python -m cli.gvpctl quality-gate check "mig_scifact_v1_to_v2"
python -m cli.gvpctl cutover execute "mig_scifact_v1_to_v2" --approve

# ── Resilience ───────────────────────────────────────────────────────────────
python -m cli.gvpctl chaos inject --fail-rate 0.4 --migration-id "mig_scifact_v1_to_v2"
python -m cli.gvpctl cutover rollback "mig_scifact_v1_to_v2" --approve

# ── Tests ────────────────────────────────────────────────────────────────────
python -m pytest tests/ -v
```

---

## 🌐 Browser Tab Order (Pre-open Before Recording)

| Tab | URL | Scene |
|---|---|---|
| Qdrant | http://localhost:6333/dashboard | After bootstrap, after cutover |
| FastAPI Health | http://localhost:8000/health | Scene 2 |
| FastAPI Docs | http://localhost:8000/docs | Scene 2, Scene 11 |
| Marquez | http://localhost:3000 | Scene 8, Scene 13 |
| Grafana | http://localhost:3001 | Scene 12 |
| Prometheus | http://localhost:9090 | Scene 12 (optional) |

---

## ⚠️ Common Issues & Fixes

| Issue | Fix |
|---|---|
| `Connection refused: 6333` | `docker compose up -d`, wait 15–20 seconds |
| `ModuleNotFoundError` | Activate venv: `& .venv\Scripts\Activate.ps1` |
| Bootstrap hangs on embedding | FastEmbed downloads the model on first run (~120 MB) — just wait |
| `migration_id not found` | Run `migrate plan` first — it creates the migration record in the catalog |
| Grafana shows no data | Fire a few `/v1/search` requests to populate Prometheus metrics |
| Port 8000 already in use | `netstat -ano \| findstr :8000` then `taskkill /PID <PID> /F` |
| `scifact_v2_shadow` not in Qdrant | Re-run `migrate apply` — the collection is created during shadow writing |
