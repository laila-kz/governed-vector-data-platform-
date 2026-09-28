# 🎬 Governed Vector Data Platform — Demo Guide

> **Purpose**: run and film the end-to-end demo without re-doing slow setup work.
> **Recording runtime**: ~10–15 minutes.
>
> ## ⚡ Read this first
>
> This guide is split into two parts, and the split matters:
>
> | Part | What it is | How often |
> |---|---|---|
> | **Part A — One-Time Build** | Expensive setup that creates the baseline state. **Already completed.** | Never again |
> | **Part B — Daily Startup** | Fast service restart. No rebuilding. | Once per filming day |
> | **Part C — The Demo** | The actual runbook, scene by scene. | Every recording |
>
> The single most important rule: **never run `python -m ingestion.bootstrap` while filming.** It takes ~40 minutes and cannot be interrupted cheaply. Part A already did it for you.

---

## 📊 Part A — One-Time Build (✅ ALREADY DONE — DO NOT RE-RUN)

Everything in this section has **already been executed and verified** against this machine. The state it produced is committed to disk and survives restarts.

### A.1 What was built, and the verified result

| # | Command | Status | Verified result |
|---|---|---|---|
| 1 | `docker compose up -d` | ✅ Done | 6 containers up; `gvp_qdrant` + `gvp_postgres` healthy |
| 2 | `python -m ingestion.download_scifact` | ✅ Done | `corpus.jsonl` 8.1 MB, `queries.jsonl` 210 KB, `qrels/test.tsv` 5.4 KB |
| 3 | `python -m ingestion.bootstrap` | ✅ Done | 5,183 documents · 7,620 chunks · 7,620 vectors · catalog synced |
| 4 | `python -m cli.gvpctl migrate plan ...` | ✅ Done | Plan `mig_77407a443e4d` created, status `planned` |

The build wrote state to three places, all persistent:

| Artifact | Location | Survives restart? |
|---|---|---|
| Sanitised corpus + chunks | `data/lance_lakehouse/` | ✅ plain files |
| Governed provenance catalog | `data/catalog.duckdb` | ✅ plain file |
| 7,620 embedded vectors | `data/qdrant_storage/` | ✅ bind-mounted (`docker-compose.yml:9`) |
| Marquez database | `marquez_pgdata` volume | ✅ named volume |

### A.2 Verify the build is still good (10 seconds, safe any time)

Run this if you ever want reassurance. All three are read-only and instant:

```powershell
Invoke-RestMethod "http://localhost:6333/collections/scifact_v1" | Select-Object -ExpandProperty result | Select-Object status, points_count
python -m cli.gvpctl status
```

Expect `status = green` and `points_count = 7620`.

### A.3 ⚠️ Why you must not re-run bootstrap

`python -m ingestion.bootstrap` is **not idempotent in the way you would expect**, despite what older notes claimed:

- The *catalog* half is idempotent — `reconcile_vector_catalog` (`ingestion/bootstrap.py:127`) skips rows that already exist.
- The *embedding* half is **not**. `ingestion/bootstrap.py:211` calls `ingest_chunks` unconditionally, which re-embeds all 7,620 chunks through `bge-small-en-v1.5` on CPU **every single time**. Measured: **~40 minutes**, with no progress output until the very end.
- DuckDB allows exactly one writing process per database file. If a bootstrap is already running, a second one dies instantly with `_duckdb.IOException: ... being used by another process`. That is what happened during this build — it is not a code fault, it is the single-writer rule.

**So: build once, then only read.**

### A.4 Blocked: the migration scenes

Six scenes depend on a second embedding model, `BAAI/bge-large-en-v1.5`, which is **not yet on this machine**:

| Blocked command | Scene | Needs |
|---|---|---|
| `migrate apply` | Shadow writing | bge-large |
| `quality-gate check` | Quality gate | bge-large |
| `cutover execute` | Cutover | bge-large |
| `cutover rollback` | Rollback | bge-large |
| `chaos inject` | Chaos | bge-large |

`model.onnx` for that model is **1,336.9 MB**, and measured throughput on this connection was **0.05 MB/s → ~7.2 hours**. Current cache state: **173 MB / 1,336.9 MB (12.9%)**.

To finish the build when you have time and bandwidth:

```powershell
$env:HF_TOKEN = "hf_..."                                  # avoid the anonymous rate limit
.\.venv\Scripts\python.exe -u tools\fetch_bge_large.py    # resumable; safe to re-run
```

It is safe to interrupt and resume. Once the cache reaches 1,336.9 MB, run Part D to finish the build.

### A.5 Commands that generate a NEW id every time

`migrate plan` does **not** produce a fixed id. Each invocation creates a new one and leaves the old row in the catalog, so repeated runs accumulate duplicate `planned` migrations. The current catalog holds exactly one:

```
mig_77407a443e4d   status=planned   7620 vectors   0 migrated
```

Always read the id off the plan table and pass that exact value to `apply`, `quality-gate`, `cutover`, and `chaos`. Earlier notes that hardcoded `mig_scifact_v1_to_v2` were wrong — that id does not exist.

> 📌 **If you re-run `migrate plan` while rehearsing**, you will end up with two plans and will not know which one the later scenes should use. Either note the newest id, or delete the stale row:
>
> ```powershell
> python -c "from catalog.db import CatalogDB; c=CatalogDB(); c.connection.execute(\"DELETE FROM migrations WHERE migration_id <> 'mig_77407a443e4d' AND status='planned' AND migrated_vectors=0\"); c.close()"
> ```

---

## 🚀 Part B — Daily Startup (2 minutes, before every recording)

Do **not** rebuild anything. Just wake the services up.

**Step 1 — Start Docker Desktop** and wait for the whale to settle.

**Step 2 — In your main terminal:**

```powershell
cd "C:\Users\kheza\Desktop\Data Engineering\Governed Vector Data Platform"
& .venv\Scripts\Activate.ps1
docker compose up -d
docker compose ps
```

Expected — all six services `Up`, with `healthy` on the two that have healthchecks:

| Container | Status | Port |
|---|---|---|
| `gvp_qdrant` | Up (healthy) | 6333 |
| `gvp_postgres` | Up (healthy) | — |
| `gvp_marquez` | Up | 5000 |
| `gvp_marquez_web` | Up | 3000 |
| `gvp_prometheus` | Up | 9090 |
| `gvp_grafana` | Up | 3001 |

**Step 3 — In a second terminal tab, start the control plane:**

```powershell
cd "C:\Users\kheza\Desktop\Data Engineering\Governed Vector Data Platform"
& .venv\Scripts\Activate.ps1
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Leave this tab running for the whole demo.

**Step 4 — Warm the search model (important).** The first search loads the embedding model into memory and takes ~2.5s; later ones take ~0.4s. Fire three throwaway searches now so you never show a stall on camera:

```powershell
1..3 | ForEach-Object { $null = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/v1/search" -ContentType "application/json" -Body '{"query":"warm up","top_k":3}' }
```

**Step 5 — Prime Grafana.** The telemetry panels only plot what has been observed. Fire the real searches from Scene 8 *before* you open Grafana in Scene 10, or the panels will be empty.

**Step 6 — Bump your terminal font to 14pt+ and pre-open every browser tab** (list in Part C, Scene 0).

### ✅ Pre-Recording Checklist

- [ ] Docker Desktop running, all 6 containers `Up`
- [ ] `.venv` activated in both terminals
- [ ] uvicorn running on 8000, `/health` returns `{"status":"ok"}`
- [ ] Three warm-up searches fired
- [ ] `python -m cli.gvpctl status` reports `total_indexed_vectors = 7620`
- [ ] Terminal font ≥ 14pt
- [ ] Browser tabs pre-opened (Part C, Scene 0)
- [ ] **You have NOT run `ingestion.bootstrap`** ✅

---

## 🎬 Part C — The Demo Runbook

Every output block below is **real, captured from this machine**, not illustrative.

### Scene 0 — Browser tabs (pre-open, do not film)

| Tab | URL | Used in |
|---|---|---|
| Qdrant dashboard | http://localhost:6333/dashboard | Scene 3 |
| FastAPI health | http://localhost:8000/health | Scene 1 |
| FastAPI docs | http://localhost:8000/docs | Scene 8 |
| Grafana | http://localhost:3001 (`admin`/`admin`) | Scene 10 |
| Prometheus | http://localhost:9090 | Scene 10 |
| Marquez | http://localhost:3000 | Scene 9 |

---

### Scene 1 — Infrastructure Stack

**Say**: *"The platform runs on six containerised services. One command brings the entire stack up."*

```powershell
docker compose ps
```

```
NAME              STATE     STATUS                  PORTS
gvp_grafana       running   Up 57 minutes           0.0.0.0:3001->3000/tcp
gvp_marquez       running   Up 57 minutes           0.0.0.0:5000-5001->5000-5001/tcp
gvp_marquez_web   running   Up 57 minutes           0.0.0.0:3000->3000/tcp
gvp_postgres      running   Up 57 minutes (healthy) 0.0.0.0:5432->5432/tcp
gvp_prometheus    running   Up 57 minutes           0.0.0.0:9090->9090/tcp
gvp_qdrant        running   Up 57 minutes (healthy) 0.0.0.0:6333-6334->6333-6334/tcp
```

Switch to the browser, show each UI ~10 seconds: Qdrant (6333), Marquez (3000), Grafana (3001), Prometheus (9090).

---

### Scene 2 — Control Plane Health

**Say**: *"The FastAPI control plane routes search traffic, exposes lineage endpoints, and publishes Prometheus telemetry."*

http://localhost:8000/health

```json
{ "status": "ok" }
```

That is the entire response — the endpoint returns only a liveness flag. Then show the generated OpenAPI docs at http://localhost:8000/docs.

---

### Scene 3 — The Governed Baseline

**Say**: *"This is the baseline the whole platform is built on: 5,183 validated documents chunked into 7,620 vectors, every one carrying full provenance."*

Back in the terminal:

```powershell
python -m cli.gvpctl status
```

```
        Governed Vector Platform Status
+----------------------------------------------+
| Metric                   | Value             |
|--------------------------+-------------------+
| active_alias             | vectors_live      |
| active_model             | bge-small-en-v1.5 |
| qdrant_collection_health | healthy           |
| total_indexed_vectors    | 7620              |
| duckdb_sync              | synced            |
+----------------------------------------------+
```

> 📌 **Say "7,620", not "15,240".** The 15,240 figure in earlier drafts of this guide was never real — it appears in no code path. The dataset is 7,620 chunks and the Qdrant collection holds exactly 7,620 points.

Now show the Qdrant dashboard: collection `scifact_v1`, **7,620 points**, 384 dimensions, Cosine distance.

---

### Scene 4 — Staleness Report

**Say**: *"Every vector is tracked as active or stale relative to the current model and strategy, so drift is visible before it bites."*

```powershell
python -m cli.gvpctl staleness
```

```
              Staleness Summary
+-------------------------------------------+
| Model             | Active | Stale | Risk |
|-------------------+--------+-------+------|
| bge-small-en-v1.5 | 7620   | 0     | low  |
| fixed_size_v1     | 7620   | 0     | low  |
+-------------------------------------------+
```

> 🔧 **This table was broken until recently.** `cli/gvpctl.py` queried `vectors.strategy_name`, a column that does not exist. The error was swallowed by a bare `except`, so the command silently printed hardcoded zeros — it *looked* like a working report showing "0 active, 0 stale". The strategy name now comes from a join against `chunks`. The numbers above are the real ones.

Also show the same thing over HTTP:

```
http://localhost:8000/v1/catalog/staleness
```

```json
{ "total_vectors": 7620, "stale_vectors": 0, "breakdown": {} }
```

---

### Scene 5 — Vector Lineage Trace

**Say**: *"Every vector carries a complete backward provenance chain — from the Qdrant point all the way back to the source document, its chunk text, the PII masking state, and the model version."*

```powershell
python -m cli.gvpctl lineage trace "vec_0001"
```

```
Vector: vec_0001
Document -> Chunk -> Model -> Qdrant Index
vector_id: vec_0001
document: 4983 (v1)
chunk: 4983:0000
model: bge-small-en-v1.5:1.0.0
collection: scifact_v1
strategy: fixed_size_v1:1.0.0
source_uri: beir://scifact/4983
```

> ⚠️ **Use `vec_0001`, not `vec_scifact_0001`.** Vector ids are `vec_0001`…`vec_7620` (`embedding/embedder.py:100`). Earlier drafts of this guide used `vec_scifact_0001`, which never existed.

Then show the full provenance including chunk text over HTTP — the strongest single shot in the demo:

```
http://localhost:8000/v1/catalog/lineage/vec_0001
```

```json
{
  "vector_id": "vec_0001",
  "collection_name": "scifact_v1",
  "dimension": 384,
  "model_name": "bge-small-en-v1.5",
  "model_version": "1.0.0",
  "chunk_id": "4983:0000",
  "doc_id": "4983",
  "doc_version": 1,
  "chunk_text": "Alterations of the architecture of cerebral white matter in the developing human brain can affect cortical development and result in functional disabilities. A line scan diffusion-weighted magnetic resonance imaging (MRI) sequence with diffusion tensor analysis was applied to measure the apparent diffusion coefficient...",
  "strategy_name": "fixed_size_v1",
  "source_uri": "beir://scifact/4983",
  "content_hash": "6e1e455b22b6012aec0fbd5b942d77f6b9a9e1020ab3e639f6e6758a93e98ca5",
  "pii_masked_flag": false
}
```

> 🔧 **A second bug, worth knowing about.** This command used to catch *any* failure and print a hardcoded fake trace — `document: doc-1`, `chunk: doc-1:0000`. Asking for a nonexistent id produced a confident, plausible, completely fabricated answer. It now fails loudly with `no cataloged vector '...'; baseline IDs run vec_0001 to vec_7620`. If you ever see invented provenance, that fallback is back.

---

### Scene 6 — Live Vector Search

**Say**: *"Queries are embedded by the active model and served straight from Qdrant, with per-request telemetry."*

Use the terminal form — it prints only id and score:

```powershell
$r = Invoke-RestMethod -Method POST -Uri "http://localhost:8000/v1/search" -ContentType "application/json" -Body '{"query":"CRISPR-Cas9 genome editing mechanisms","top_k":3}'
"collection: $($r.collection)"
$r.results | Format-Table id, score -AutoSize
```

```
collection: vectors_live

id                                        score
--                                       -----
b2e95414-397b-57eb-b975-e107e1aeafb4  0.8404053
8fddfa45-160b-5fab-b719-a331ebc42c2c  0.81772363
d5fee0a9-a82b-59b2-9c92-22d425fa51dc  0.8021588
```

> 📌 Two presentation notes:
> - **Do not paste the raw response into the terminal.** `/v1/search` echoes the full 384-float query vector plus full payloads — roughly 400 lines of noise. Use the projection above, or the Swagger UI at http://localhost:8000/docs.
> - Fire these searches **before** Scene 8, so the Grafana latency panels have data.

---

### Scene 7 — Migration Plan (pre-flight, no embedding)

**Say**: *"Before a migration touches live infrastructure, the operator generates a pre-flight plan — vector counts, token volume, cost, runtime, and predicted drift, all before a single embedding is computed."*

The plan already exists from Part A. Show it without re-running, or re-run to generate a fresh one:

```powershell
python -m cli.gvpctl migrate plan "bge-small-en-v1.5" "bge-large-en-v1.5" --approve
```

```
                Migration Plan mig_77407a443e4d
+--------------------------------------------------------------------+
| Change                    | Value                                  |
|---------------------------+----------------------------------------|
| model                     | bge-small-en-v1.5 -> bge-large-en-v1.5 |
| strategy                  | fixed_size_v1 -> fixed_size_v1         |
| vector count              | 7620                                   |
| token volume              | 1,624,031                              |
| estimated API cost        | $0.000000                              |
| estimated duration        | 12.00s                                 |
| predicted retrieval drift | 0.0000                                 |
| status                    | planned                                |
+--------------------------------------------------------------------+
```

> 📌 **Read the migration id off the table.** It is generated per run — `mig_77407a443e4d` above, a different value each time. Earlier drafts claimed it was always `mig_scifact_v1_to_v2`; that is not true. Also note the real drift estimate is `0.0000` and duration `12.00s`, not the `0.1824` / `45.20s` in earlier drafts.

---

### Scene 8 — Shadow Writing ⛔ *blocked, needs bge-large*

**Say**: *"The migration worker shadow-writes target embeddings into an isolated `scifact_v2_shadow` collection, so live search traffic is completely unaffected."*

```powershell
python -m cli.gvpctl migrate apply "mig_77407a443e4d" --approve
```

**Do not attempt this on filming day.** It requires `bge-large-en-v1.5` (see A.4). Two reasons it is a poor live demo even once available: it re-embeds all 7,620 chunks, and on this CPU `bge-large` is roughly an order of magnitude slower than `bge-small` — expect hours, not the 45 seconds earlier drafts claimed. Complete it as a build step (Part D) and present the finished shadow collection.

---

### Scene 9 — Quality Gate ⛔ *blocked, needs bge-large*

```powershell
python -m cli.gvpctl quality-gate check "mig_77407a443e4d"
```

Also blocked. Note for narration: the gate compares baseline vs target on Recall@10, NDCG@10, MRR, and error rate, and blocks cutover on regression.

---

### Scene 10 — Atomic Cutover ⛔ *blocked, needs bge-large*

```powershell
python -m cli.gvpctl cutover execute "mig_77407a443e4d" --approve
```

Blocked. The real command output is a single line — `Migration <id> cut over to vectors_live` — not the multi-line "✅ Cutover complete / Catalog updated: 15,240 vectors marked active" shown in earlier drafts.

---

### Scene 11 — Grafana Telemetry (✅ works today)

**Say**: *"Every operation is instrumented. Grafana scrapes the control plane every five seconds."*

http://localhost:3001 (`admin` / `admin`) → **Governed Vector Platform — Control Plane Telemetry**

| Panel | Type |
|---|---|
| Stale Vectors Ratio | gauge |
| Total Tokens Embedded | stat |
| Estimated Cost (USD) | stat |
| Total HTTP Requests | stat |
| Search Latency Percentiles (p50/p95/p99) | time series |
| Request Throughput by Endpoint | time series |

> 📌 Run Scene 6's searches **first**. Prometheus scrapes `host.docker.internal:8000/metrics` every 5s, and empty panels are a bad look. Raw metrics: http://localhost:8000/metrics. Metric names are **unprefixed**: `search_latency_seconds`, `embedding_tokens_total`, `estimated_cost_usd_total`, `stale_vector_gauge`, `http_requests_total`.

> 📌 The **Stale Vectors Ratio** gauge only updates when `/v1/catalog/staleness` is called (Scene 4). Hit that endpoint before showing Grafana.

---

### Scene 12 — Marquez Lineage ⛔ *partially blocked*

http://localhost:3000

Marquez is running and reachable, but the OpenLineage events for the baseline are emitted during `migrate apply` / `ingest`, which is blocked. You can show the Marquez UI and the API at http://localhost:5001, but expect an **empty** dataset list until the migration build is completed. Do not narrate a populated lineage graph on screen.

---

### Scene 13 — Chaos Fault Injection ⛔ *blocked, needs bge-large*

```powershell
python -m cli.gvpctl chaos inject --fail-rate 0.4 --migration-id "mig_77407a443e4d"
```

Blocked — the chaos runner drives the same `ShadowMigrationWorker`, so it needs bge-large too.

---

### Scene 14 — Instant Rollback ⛔ *blocked, needs bge-large*

```powershell
python -m cli.gvpctl cutover rollback "mig_77407a443e4d" --approve
```

Blocked. Real output is one line: `Migration <id> rolled back to vectors_live`.

---

### Scene 15 — Test Suite (✅ works today, ~11 seconds)

**Say**: *"Seventy-eight tests cover ingestion, catalog, migration, and telemetry — the whole platform is under test."*

```powershell
python -m pytest tests/ -q
```

```
...............................................................  [100%]
78 passed, 1 warning in 11.16s
```

> 📌 It is **78 passed**, not 73. Requires Qdrant running on 6333 (`test_qdrant_client_matches_configured_server_api` hits the live server).

---

## 📈 What You Can Film Today

| Scene | Status | Time on camera |
|---|---|---|
| 1 Infrastructure | ✅ | 30s |
| 2 Control plane health | ✅ | 15s |
| 3 Baseline + status | ✅ | 30s |
| 4 Staleness | ✅ | 30s |
| 5 Lineage trace | ✅ | 60s |
| 6 Search | ✅ | 45s |
| 7 Migration plan | ✅ | 45s |
| 11 Grafana | ✅ | 45s |
| 15 Test suite | ✅ | 20s |
| 8/9/10/13/14 Migration lifecycle | ⛔ | blocked |
| 12 Marquez | ⚠️ partial | UI only, empty graph |

**That is roughly 5–6 minutes of solid footage** with no build time, no 40-minute stall, and no fabricated output.

---

## 🔨 Part D — Finishing the Blocked Build (do this once, offline)

Run when you have time and bandwidth. Not on filming day.

**D.1 — Download the model (resumable, interruptible):**

```powershell
$env:HF_TOKEN = "hf_..."
.\.venv\Scripts\python.exe -u tools\fetch_bge_large.py
```

Expect `DONE ... size=1336.9MB`. Progress is visible in `%TEMP%\fastembed_cache`.

**D.2 — Shadow-write the target collection (hours on CPU; this is the build, not the demo):**

```powershell
python -m cli.gvpctl migrate apply "mig_77407a443e4d" --approve
```

**D.3 — Quality gate, cutover, verify:**

```powershell
python -m cli.gvpctl quality-gate check "mig_77407a443e4d"
python -m cli.gvpctl cutover execute "mig_77407a443e4d" --approve
python -m cli.gvpctl status
```

After D.3, Scenes 8–14 become filmable and the migration id in this guide should be replaced with yours.

---

## ⚠️ Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'qdrant_client'` | Using global Python 3.11, not the venv | `& .venv\Scripts\Activate.ps1` |
| `_duckdb.IOException: ... being used by another process` | A bootstrap is already running; DuckDB is single-writer | Find it: `Get-CimInstance Win32_Process -Filter "Name like '%python%'" \| Select ProcessId,CommandLine` — or wait it out |
| Bootstrap appears hung | It is embedding 7,620 chunks, ~40 min, silent | Wait. Check progress: see A.2. Never start a second one |
| `Connection refused: 6333` | Qdrant container down | `docker compose up -d`, wait ~20s |
| `no cataloged vector 'vec_scifact_0001'` | That id never existed | Use `vec_0001` … `vec_7620` |
| `unknown migration` | Wrong id, or no plan yet | Copy the id off the `migrate plan` table |
| `Port 8000 already in use` | uvicorn already running | `netstat -ano \| findstr :8000` then `taskkill /PID <PID> /F` |
| Grafana panels empty | No traffic since Prometheus last scraped | Fire 3+ searches, hit `/v1/catalog/staleness`, wait 10s |
| First search takes ~2.5s | Embedding model cold start | Warm up in Part B, Step 4 |
| bge-large download crawls | ~0.05 MB/s unauthenticated | Set `$env:HF_TOKEN` |
| `Required module 'pytz' failed to import` | `pytz` missing from venv; only affects ad-hoc DuckDB timestamp casts | `.\.venv\Scripts\python.exe -m pip install pytz` |

---

## 📋 Quick Command Reference

```powershell
# ── DAILY (Part B) ──────────────────────────────────────────────────────────
docker compose up -d
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000   # separate tab

# ── DEMO (Part C) ───────────────────────────────────────────────────────────
python -m cli.gvpctl status
python -m cli.gvpctl staleness
python -m cli.gvpctl lineage trace "vec_0001"
python -m cli.gvpctl migrate plan "bge-small-en-v1.5" "bge-large-en-v1.5" --approve
python -m pytest tests/ -q

# ── ONE-TIME BUILD (Part A) — ALREADY DONE, DO NOT RE-RUN ──────────────────
# python -m ingestion.download_scifact
# python -m ingestion.bootstrap          # ~40 min, single-writer
```

---

## 📌 Facts to get right on camera

| Claim | Truth |
|---|---|
| Vector count | **7,620** (not 15,240) |
| Documents | 5,183 |
| Vector id format | `vec_0001` … `vec_7620` (not `vec_scifact_0001`) |
| Model | `bge-small-en-v1.5`, 384-dim, Cosine |
| Live alias | `vectors_live` → `scifact_v1` |
| Bootstrap duration | **~40 min** on CPU (not 1–3 min) |
| Bootstrap idempotent? | Catalog yes, embedding **no** |
| `/health` response | `{"status":"ok"}` only |
| Test count | **78 passed** |
| Migration id | Generated per run (e.g. `mig_77407a443e4d`) |
| Drift estimate | `0.0000`; estimated duration `12.00s` |
| Containers | **6**, not 5 |
