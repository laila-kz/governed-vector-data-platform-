# FastAPI Control Plane API Reference

The Governed Vector Data Platform exposes a high-performance, asynchronous REST API for vector query routing, lineage queries, telemetry, and dynamic policy updates.

**Base URL**: http://localhost:8000

---

## 1. System & Health

### GET /health
Returns system health and active routing state.

**Response 200 OK**:
`json
{
  " status\: \healthy\,
 \active_alias\: \vectors_live\,
 \active_model\: \bge-small-en-v1.5\,
 \active_version\: \1.0.0\
}
`

---

## 2. Vector Search & Proxy Routing

### POST /v1/search
Embeds query text and queries the active Qdrant alias (ectors_live). If shadow reading is enabled in configs/routing_policy.yaml, asynchronously executes a background shadow query to scifact_v2_shadow to compute latency deltas.

**Request Body**:
`json
{
 \query\: \What are the molecular mechanisms of CRISPR-Cas9 genome editing?\,
 \top_k\: 5
}
`

**Response 200 OK**:
`json
{
 \collection\: \vectors_live\,
 \limit\: 5,
 \results\: [
 {
 \id\: \sci_chunk_10521\,
 \score\: 0.8942
 },
 {
 \id\: \sci_chunk_08412\,
 \score\: 0.8415
 }
 ]
}
`

---

## 3. Metadata & Lineage Provenance

### GET /v1/catalog/lineage/{vector_id}
Returns complete backward provenance for a vector: source document, chunk index, text snippet, embedding model version, and PII masking status.

**Parameters**:
- ector_id (path, string): Unique vector identifier.

**Response 200 OK**:
`json
{
 \vector_id\: \vec_scifact_0001\,
 \collection_name\: \scifact_v1\,
 \dimension\: 384,
 \model_name\: \BAAI/bge-small-en-v1.5\,
 \model_version\: \1.0.0\,
 \provider\: \fastembed\,
 \chunk_id\: \doc_487:0000\,
 \doc_id\: \doc_487\,
 \doc_version\: 1,
 \chunk_text\: \Alternative splicing generates functional diversity...\,
 \strategy_name\: \fixed_size_v1\,
 \strategy_version\: \1.0.0\,
 \source_uri\: \s3://scifact-corpus/raw/doc_487.json\,
 \content_hash\: \a4f89d12e...\,
 \pii_masked_flag\: true
}
`

### GET /v1/catalog/stale-vectors
Returns all vectors in the catalog that were generated using non-active model or chunking strategy versions.

---

## 4. Telemetry & Routing Control

### GET /v1/telemetry/staleness
Computes stale vector ratios and updates the Prometheus gauge stale_vector_percentage.

### GET /v1/routing/policy
Retrieves current routing and traffic-split configuration.

### POST /v1/routing/policy
Dynamically reloads routing policy from disk or updates active settings without requiring server restarts.

### GET /metrics
Prometheus metrics exposition endpoint in standard OpenMetrics format.
