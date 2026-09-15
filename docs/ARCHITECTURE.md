# Architecture

## Control-Plane Boundaries

The platform separates durable content storage, vector serving, and operator metadata:

| Boundary | Technology | Responsibility |
| --- | --- | --- |
| Content | Lance | Versioned sanitized chunks |
| Catalog | DuckDB | Documents, chunks, vectors, migrations, evaluation runs |
| Serving | Qdrant | ANN collections and the `vectors_live` alias |
| Governance | OpenLineage and Marquez | Dataset and embedding lineage |
| Telemetry | Prometheus and Grafana | Search latency, token cost, and staleness |

## Migration Safety

1. `gvpctl migrate plan` estimates scope, tokens, cost, and runtime.
2. `migrate apply` writes inactive target vectors to `scifact_v2_shadow`.
3. The SciFact quality gate evaluates baseline and shadow rankings.
4. `CutoverManager` swaps `vectors_live` atomically only after operator approval.
5. Rollback restores `scifact_v1`, active catalog state, and routing policy.

The quality gate does not compare 384-dimensional and 1024-dimensional vectors directly. Retrieval quality is compared through benchmark rankings; vector geometry is analyzed independently by the drift detector.

## Observability Path

Embedding and search operations emit metadata and Prometheus metrics at the control-plane boundary. Marquez receives OpenLineage run events with the custom vector embedding facet, while Grafana reads Prometheus metrics through the provisioned datasource.