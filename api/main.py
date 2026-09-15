"""FastAPI control plane for the governed vector platform."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from qdrant_client import QdrantClient

from api.search_proxy import SearchProxy, _load_policy
from api.telemetry import (
    metrics_content_type,
    metrics_payload,
    observe_http_request,
    observe_search_latency,
    observe_shadow_latency_delta,
    set_stale_vector_percentage,
)
from catalog.db import DEFAULT_DATABASE_PATH, CatalogDB
from catalog.lineage_service import LineageService

APP_ROOT = Path(__file__).resolve().parent.parent
POLICY_PATH = APP_ROOT / "configs" / "routing_policy.yaml"


class FakeQdrantClient:
    """Fallback client used when Qdrant is not available at runtime."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def search(self, collection_name: str, query_vector: list[float], limit: int, **_: Any) -> dict[str, Any]:
        payload = {
            "collection": collection_name,
            "limit": limit,
            "query_vector": query_vector,
            "points": [{"id": "fake-1", "score": 0.98}],
        }
        self.calls.append(payload)
        return payload

    def search_shadow(self, collection_name: str, query_vector: list[float], limit: int, **_: Any) -> dict[str, Any]:
        payload = {
            "collection": collection_name,
            "limit": limit,
            "query_vector": query_vector,
            "points": [{"id": "fake-shadow-1", "score": 0.94}],
        }
        self.calls.append(payload)
        return payload


def _build_proxy(config: dict[str, Any] | None = None) -> SearchProxy:
    proxy_config = _load_policy(POLICY_PATH) if config is None else config
    client: Any = FakeQdrantClient()
    use_real_qdrant = os.getenv("USE_REAL_QDRANT", "true").lower() == "true"
    if use_real_qdrant:
        try:
            client = QdrantClient(url=os.getenv("QDRANT_URL", "http://localhost:6333"))
            client.get_collections()
            if not client.collection_exists(proxy_config.get("active_alias", "vectors_live")):
                raise RuntimeError("active Qdrant alias is not available")
        except Exception:
            client = FakeQdrantClient()

    def embedder(texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]

    return SearchProxy(
        client=client,
        config=proxy_config,
        embedder=embedder,
        shadow_latency_observer=lambda delta: observe_shadow_latency_delta("/v1/search", delta),
    )


app = FastAPI(title="Governed Vector Data Platform Control Plane")
app.state.proxy_config = _load_policy(POLICY_PATH)
app.state.search_proxy = _build_proxy(app.state.proxy_config)


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    route = request.scope.get("route")
    route_path = getattr(route, "path", request.url.path)
    observe_http_request(request.method, str(route_path), response.status_code, duration)
    return response


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/v1/search")
async def search(request: Request) -> dict[str, Any]:
    payload = await request.json()
    query = str(payload.get("query", ""))
    top_k = int(payload.get("top_k", 5))
    started = time.perf_counter()
    try:
        result = app.state.search_proxy.search(query, top_k)
        status = "success"
    except Exception as exc:  # pragma: no cover - safety fallback
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    observe_search_latency(
        "/v1/search",
        app.state.proxy_config.get("active_model", "unknown"),
        status,
        time.perf_counter() - started,
    )
    return result


@app.get("/v1/catalog/lineage/{vector_id}")
async def get_lineage(vector_id: str) -> dict[str, Any]:
    with CatalogDB(DEFAULT_DATABASE_PATH) as catalog:
        service = LineageService(catalog)
        try:
            return service.get_vector_lineage(vector_id).model_dump(mode="json")
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/v1/catalog/staleness")
async def get_staleness() -> dict[str, Any]:
    with CatalogDB(DEFAULT_DATABASE_PATH) as catalog:
        total_vectors = catalog.query("SELECT COUNT(*) FROM vectors")[0][0]
        if total_vectors == 0:
            set_stale_vector_percentage(app.state.proxy_config.get("active_model", "unknown"), 0.0)
            return {"total_vectors": 0, "stale_vectors": 0, "breakdown": {}}
        strategy_rows = catalog.query(
            "SELECT strategy_version FROM chunk_strategies ORDER BY created_at DESC, strategy_version DESC LIMIT 1"
        )
        current_strategy = str(strategy_rows[0][0]) if strategy_rows else None
        stale_condition = "v.model_name <> ? OR v.model_version <> ?"
        parameters: list[Any] = [
            app.state.proxy_config.get("active_model", ""),
            app.state.proxy_config.get("active_version", ""),
        ]
        if current_strategy is not None:
            stale_condition += " OR c.strategy_version <> ?"
            parameters.append(current_strategy)
        stale_rows = catalog.query(
            f"""
            SELECT v.model_version, COUNT(*)
            FROM vectors AS v
            JOIN chunks AS c ON c.chunk_id = v.chunk_id
            WHERE {stale_condition}
            GROUP BY v.model_version
            ORDER BY v.model_version
            """,
            parameters,
        )
        breakdown = {str(model_version): int(count) for model_version, count in stale_rows}
        stale_vectors = sum(breakdown.values())
        set_stale_vector_percentage(
            app.state.proxy_config.get("active_model", "unknown"),
            stale_vectors / int(total_vectors) * 100,
        )
        return {
            "total_vectors": int(total_vectors),
            "stale_vectors": stale_vectors,
            "breakdown": breakdown,
        }


@app.post("/v1/admin/routing")
async def update_routing(request: Request) -> dict[str, Any]:
    new_policy = await request.json()
    config = _load_policy(POLICY_PATH)
    config.update({k: v for k, v in new_policy.items() if v is not None})
    with POLICY_PATH.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    app.state.proxy_config = config
    app.state.search_proxy = _build_proxy(config)
    return {"status": "updated", "config": config}


@app.get("/metrics")
async def metrics() -> Response:
    return Response(content=metrics_payload(), media_type=metrics_content_type())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=False)
