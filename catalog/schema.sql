CREATE TABLE IF NOT EXISTS documents (
    doc_id VARCHAR NOT NULL,
    doc_version INTEGER NOT NULL DEFAULT 1,
    title VARCHAR NOT NULL,
    source_uri VARCHAR NOT NULL,
    author VARCHAR NOT NULL,
    classification_level VARCHAR NOT NULL,
    language VARCHAR NOT NULL,
    content_hash VARCHAR,
    pii_masked_flag BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (doc_id, doc_version)
);

CREATE TABLE IF NOT EXISTS chunk_strategies (
    strategy_name VARCHAR NOT NULL,
    strategy_version VARCHAR NOT NULL,
    splitter_type VARCHAR NOT NULL,
    chunk_size_tokens INTEGER NOT NULL,
    chunk_overlap_tokens INTEGER NOT NULL DEFAULT 0,
    tokenizer_name VARCHAR NOT NULL,
    pii_masked_flag BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (strategy_name, strategy_version),
    CHECK (chunk_size_tokens > 0),
    CHECK (chunk_overlap_tokens >= 0 AND chunk_overlap_tokens < chunk_size_tokens)
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id VARCHAR PRIMARY KEY,
    doc_id VARCHAR NOT NULL,
    doc_version INTEGER NOT NULL DEFAULT 1,
    strategy_name VARCHAR NOT NULL,
    strategy_version VARCHAR NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_text VARCHAR NOT NULL,
    chunk_hash VARCHAR NOT NULL,
    tokenizer_name VARCHAR,
    pii_masked_flag BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    FOREIGN KEY (doc_id, doc_version) REFERENCES documents (doc_id, doc_version),
    FOREIGN KEY (strategy_name, strategy_version)
        REFERENCES chunk_strategies (strategy_name, strategy_version),
    UNIQUE (doc_id, doc_version, chunk_index)
);

CREATE TABLE IF NOT EXISTS embedding_models (
    model_name VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    dimensions INTEGER NOT NULL,
    pricing_usd_per_1k_tokens DOUBLE NOT NULL DEFAULT 0,
    latency_ms_p95 DOUBLE,
    status VARCHAR NOT NULL DEFAULT 'active',
    pii_masked_flag BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    PRIMARY KEY (model_name, model_version),
    CHECK (dimensions > 0),
    CHECK (pricing_usd_per_1k_tokens >= 0)
);

CREATE TABLE IF NOT EXISTS vectors (
    vector_id VARCHAR PRIMARY KEY,
    chunk_id VARCHAR NOT NULL,
    model_name VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    collection_name VARCHAR NOT NULL,
    dimension INTEGER NOT NULL,
    distance_metric VARCHAR NOT NULL DEFAULT 'Cosine',
    active BOOLEAN NOT NULL DEFAULT TRUE,
    pii_masked_flag BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id),
    FOREIGN KEY (model_name, model_version)
        REFERENCES embedding_models (model_name, model_version),
    CHECK (dimension > 0),
    CHECK (distance_metric IN ('Cosine', 'Dot', 'Euclidean'))
);

CREATE TABLE IF NOT EXISTS migrations (
    migration_id VARCHAR PRIMARY KEY,
    source_model_name VARCHAR NOT NULL,
    source_model_version VARCHAR NOT NULL,
    target_model_name VARCHAR NOT NULL,
    target_model_version VARCHAR NOT NULL,
    source_strategy_name VARCHAR,
    source_strategy_version VARCHAR,
    target_strategy_name VARCHAR,
    target_strategy_version VARCHAR,
    status VARCHAR NOT NULL,
    total_vectors BIGINT NOT NULL DEFAULT 0,
    migrated_vectors BIGINT NOT NULL DEFAULT 0,
    total_characters BIGINT NOT NULL DEFAULT 0,
    estimated_tokens BIGINT NOT NULL DEFAULT 0,
    estimated_cost_usd DOUBLE NOT NULL DEFAULT 0,
    estimated_duration_seconds DOUBLE NOT NULL DEFAULT 0,
    error_count INTEGER NOT NULL DEFAULT 0,
    predicted_retrieval_drift DOUBLE NOT NULL DEFAULT 0,
    batch_size INTEGER NOT NULL DEFAULT 64,
    rate_limit_per_second DOUBLE NOT NULL DEFAULT 1,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    pii_masked_flag BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    FOREIGN KEY (source_model_name, source_model_version)
        REFERENCES embedding_models (model_name, model_version),
    FOREIGN KEY (target_model_name, target_model_version)
        REFERENCES embedding_models (model_name, model_version),
    CHECK (total_vectors >= 0 AND migrated_vectors >= 0 AND migrated_vectors <= total_vectors)
);

CREATE TABLE IF NOT EXISTS migration_events (
    event_id VARCHAR PRIMARY KEY,
    migration_id VARCHAR NOT NULL,
    event_type VARCHAR NOT NULL,
    details_json JSON,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    FOREIGN KEY (migration_id) REFERENCES migrations (migration_id)
);

CREATE TABLE IF NOT EXISTS retrieval_eval_runs (
    run_id VARCHAR PRIMARY KEY,
    migration_id VARCHAR,
    model_name VARCHAR NOT NULL,
    model_version VARCHAR NOT NULL,
    strategy_name VARCHAR,
    strategy_version VARCHAR,
    dataset_name VARCHAR NOT NULL,
    recall_at_5 DOUBLE,
    recall_at_10 DOUBLE,
    ndcg_at_10 DOUBLE,
    mrr DOUBLE,
    cosine_drift DOUBLE,
    error_rate DOUBLE NOT NULL DEFAULT 0,
    status VARCHAR NOT NULL DEFAULT 'completed',
    metrics_json JSON,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    pii_masked_flag BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT current_timestamp,
    FOREIGN KEY (migration_id) REFERENCES migrations (migration_id),
    FOREIGN KEY (model_name, model_version)
        REFERENCES embedding_models (model_name, model_version)
);