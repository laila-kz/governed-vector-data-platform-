# Information Retrieval (IR) Evaluation & Quality Gate Guide

The Governed Vector Data Platform incorporates an automated Information Retrieval evaluation harness to mathematically verify search quality before permitting vector index cutovers.

---

## 1. Ground-Truth Benchmark Corpus: BEIR / SciFact

Rather than evaluating on synthetic or toy queries, the platform uses the **BEIR / SciFact** scientific benchmark:
- **Corpus**: 5,183 peer-reviewed scientific paper abstracts.
- **Queries & Annotations**: 1,109 expert-annotated claim queries with human relevance judgments (`qrels/test.tsv`).
- **Materialized Catalog**: 300 qrel-bearing query evaluation pairs in `evaluation/beir_scifact_qrels.json`.

---

## 2. Information Retrieval Metrics

Retrieval effectiveness is evaluated across four core ranking metrics:

### 1. Recall@k ($k=5, 10$)
Measures the proportion of relevant documents retrieved in top-$k$ results:
$$\text{Recall@k} = \frac{|\text{Retrieved}_k \cap \text{Relevant}|}{|\text{Relevant}|}$$

### 2. NDCG@10 (Normalized Discounted Cumulative Gain)
Evaluates ranking position quality with logarithmic position penalties:
$$\text{DCG@10} = \sum_{i=1}^{10} \frac{2^{rel_i} - 1}{\log_2(i + 1)}, \quad \text{NDCG@10} = \frac{\text{DCG@10}}{\text{IDCG@10}}$$

### 3. MRR (Mean Reciprocal Rank)
Evaluates the reciprocal rank of the first relevant document:
$$\text{MRR} = \frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\text{rank}_i}$$

### 4. Latency Benchmark
Measures p50 and p95 retrieval latency per query across embedding models.

---

## 3. High-Dimensional Vector Space Drift Analysis

When migrating across embedding models (e.g. 384-dimensional `bge-small-en-v1.5` to 1024-dimensional `bge-large-en-v1.5`), direct Euclidean distance comparison is mathematically invalid due to mismatched dimensionalities.

The platform's **Drift Analyzer (`evaluation/drift_detector.py`)** computes:
- **Neighborhood Preservation (k-NN Jaccard Similarity)**: Evaluates whether semantic neighbors in source space remain neighbors in target space.
- **Cosine Centroid Drift**: Measures global semantic cluster shifts.
- **Collapse Detection**: Validates that target vectors do not collapse onto a low-rank manifold.

---

## 4. Automated Quality Gate Invariants

When `gvpctl quality-gate check` or `migrate apply` runs, both baseline (`scifact_v1`) and shadow (`scifact_v2_shadow`) collections are queried against the evaluation catalog.

### Invariant Rules:
1. **Recall Regression Bound**:
   $$\text{Recall@10}(v_2) \ge \text{Recall@10}(v_1) - 0.02$$
2. **NDCG Monotonicity**:
   $$\text{NDCG@10}(v_2) \ge \text{NDCG@10}(v_1)$$
3. **Zero Error Rate**:
   $$\text{Error Rate} = 0.0$$

If any rule fails:
- Cutover authorization is blocked.
- Evaluation run status is recorded as `failed` in `retrieval_eval_runs`.
- Live traffic continues querying `vectors_live` without disruption.
