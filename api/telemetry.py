"""Prometheus metrics for the search proxy and FastAPI control plane."""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

SEARCH_LATENCY_SECONDS = Histogram(
    "search_latency_seconds",
    "Latency of vector search requests by route and model.",
    labelnames=("route", "model", "status"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

EMBEDDING_TOKENS_TOTAL = Counter(
    "embedding_tokens_total",
    "Total number of tokens embedded by model version.",
    labelnames=("model_name", "model_version"),
)

ESTIMATED_COST_USD_TOTAL = Counter(
    "estimated_cost_usd_total",
    "Estimated cumulative embedding spend in USD.",
    labelnames=("model_name", "model_version"),
)

STALE_VECTOR_GAUGE = Gauge(
    "stale_vector_gauge",
    "Percentage of vectors that are stale relative to the active model or strategy.",
    labelnames=("model_name",),
)

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests handled by the control plane.",
    labelnames=("method", "route", "status_code"),
)

HTTP_REQUEST_LATENCY_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds.",
    labelnames=("method", "route", "status_code"),
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)


def metrics_payload() -> bytes:
    """Return the current Prometheus payload."""
    return generate_latest()


def observe_http_request(method: str, route: str, status_code: int, duration_seconds: float) -> None:
    """Record HTTP request telemetry."""
    HTTP_REQUESTS_TOTAL.labels(method=method, route=route, status_code=str(status_code)).inc()
    HTTP_REQUEST_LATENCY_SECONDS.labels(method=method, route=route, status_code=str(status_code)).observe(duration_seconds)


def observe_search_latency(route: str, model: str, status: str, duration_seconds: float) -> None:
    """Record search latency."""
    SEARCH_LATENCY_SECONDS.labels(route=route, model=model, status=status).observe(duration_seconds)


def observe_token_usage(model_name: str, model_version: str, tokens: int) -> None:
    """Track embedding token consumption."""
    EMBEDDING_TOKENS_TOTAL.labels(model_name=model_name, model_version=model_version).inc(tokens)


def observe_cost(model_name: str, model_version: str, cost_usd: float) -> None:
    """Track estimated spend."""
    ESTIMATED_COST_USD_TOTAL.labels(model_name=model_name, model_version=model_version).inc(cost_usd)


def set_stale_vector_percentage(model_name: str, percentage: float) -> None:
    """Publish the percentage of stale vectors."""
    STALE_VECTOR_GAUGE.labels(model_name=model_name).set(percentage)


def metrics_content_type() -> str:
    return CONTENT_TYPE_LATEST
