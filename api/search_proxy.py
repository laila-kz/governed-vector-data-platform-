"""Qdrant search proxy with routing policy and optional shadow reads."""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml
from qdrant_client import QdrantClient

from api.telemetry import observe_embedding_usage

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "configs" / "routing_policy.yaml"


class QdrantCompatibilityClient:
    """Normalize search behavior across Qdrant client versions."""

    def __init__(self, client: Any) -> None:
        self._client = client

    def _call_search(self, method_name: str, collection_name: str, query_vector: list[float], limit: int, **kwargs: Any) -> dict[str, Any]:
        method = getattr(self._client, method_name, None)
        if callable(method):
            return method(collection_name=collection_name, query_vector=query_vector, limit=limit, **kwargs)

        query_method = getattr(self._client, "query_points", None)
        if callable(query_method):
            result = query_method(
                collection_name=collection_name,
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
                **kwargs,
            )
            points: list[dict[str, Any]] = []
            for point in getattr(result, "points", []) or []:
                payload = getattr(point, "payload", None)
                if payload is None and isinstance(point, dict):
                    payload = point.get("payload", {})
                point_id = getattr(point, "id", None)
                if point_id is None and isinstance(point, dict):
                    point_id = point.get("id")
                score = getattr(point, "score", None)
                if score is None and isinstance(point, dict):
                    score = point.get("score")
                points.append({"id": point_id, "score": score, "payload": payload})
            return {"collection": collection_name, "limit": limit, "query_vector": query_vector, "points": points}

        raise AttributeError(f"Qdrant client does not support search or query_points: {type(self._client)!r}")

    def search(self, *, collection_name: str, query_vector: list[float], limit: int, **kwargs: Any) -> dict[str, Any]:
        return self._call_search("search", collection_name, query_vector, limit, **kwargs)

    def search_shadow(self, *, collection_name: str, query_vector: list[float], limit: int, **kwargs: Any) -> dict[str, Any]:
        return self._call_search("search_shadow", collection_name, query_vector, limit, **kwargs)


def _load_policy(path: str | Path = DEFAULT_POLICY_PATH) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return {
        "active_alias": config.get("active_alias", "vectors_live"),
        "active_model": config.get("active_model", "bge-small-en-v1.5"),
        "active_version": config.get("active_version", "1.0.0"),
        "shadow_collection": config.get("shadow_collection"),
        "shadow_model": config.get("shadow_model"),
        "shadow_version": config.get("shadow_version"),
        "traffic_split_ratio": float(config.get("traffic_split_ratio", 1.0)),
        "shadow_read_enabled": bool(config.get("shadow_read_enabled", False)),
    }


class SearchProxy:
    """Route query requests to live or shadow Qdrant collections."""

    def __init__(
        self,
        client: Any,
        config: Mapping[str, Any] | None = None,
        embedder: Callable[[list[str]], list[list[float]]] | None = None,
        policy_path: str | Path = DEFAULT_POLICY_PATH,
        shadow_latency_observer: Callable[[float], None] | None = None,
    ) -> None:
        self.client = QdrantCompatibilityClient(client)
        self.policy_path = Path(policy_path)
        self.config = dict(_load_policy(self.policy_path) if config is None else config)
        self.embedder = embedder or (lambda texts: [[0.1, 0.2, 0.3] for _ in texts])
        self.shadow_latency_observer = shadow_latency_observer

    def update_policy(self, updates: Mapping[str, Any]) -> dict[str, Any]:
        """Apply a routing-policy delta and persist it to the YAML config file."""
        self.config.update({key: value for key, value in updates.items() if value is not None})
        self.policy_path.parent.mkdir(parents=True, exist_ok=True)
        with self.policy_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(self.config, handle, sort_keys=False)
        return dict(self.config)

    def _route_collection(self, query: str) -> str:
        ratio = float(self.config.get("traffic_split_ratio", 1.0))
        active_alias = str(self.config.get("active_alias", "vectors_live"))
        shadow_collection = self.config.get("shadow_collection")
        if shadow_collection is None or ratio >= 1.0:
            return active_alias
        if ratio <= 0.0:
            return str(shadow_collection)
        digest = int(hashlib.sha256(query.encode("utf-8")).hexdigest(), 16)
        bucket = digest % 1000 / 1000.0
        if bucket < ratio:
            return active_alias
        return str(shadow_collection)

    def _embed_query(self, query: str) -> list[float]:
        embeddings = self.embedder([query])
        if not embeddings:
            raise ValueError("embedding model returned no result for the query")
        return list(embeddings[0])

    def _shadow_read(
        self,
        query: str,
        query_vector: list[float],
        limit: int,
        primary_duration_seconds: float,
    ) -> None:
        shadow_collection = self.config.get("shadow_collection")
        if not self.config.get("shadow_read_enabled", False) or not shadow_collection:
            return
        def _task() -> None:
            try:
                started = time.perf_counter()
                self.client.search_shadow(
                    collection_name=str(shadow_collection),
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=0.0,
                )
                if self.shadow_latency_observer is not None:
                    self.shadow_latency_observer(time.perf_counter() - started - primary_duration_seconds)
            except Exception:
                pass

        thread = threading.Thread(target=_task, daemon=True)
        thread.start()

    def search(self, query: str, top_k: int) -> dict[str, Any]:
        """Embed and query Qdrant using the configured active/shadow collection policy."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self._embed_query(query)
        observe_embedding_usage(
            str(self.config.get("active_model", "unknown")),
            str(self.config.get("active_version", "unknown")),
            query,
            float(self.config.get("pricing_usd_per_1k_tokens", 0.0)),
        )
        active_alias = str(self.config.get("active_alias", "vectors_live"))
        collection_name = self._route_collection(query)
        started = time.perf_counter()
        primary_result = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            score_threshold=0.0,
        )
        primary_duration_seconds = time.perf_counter() - started

        self._shadow_read(query, query_vector, top_k, primary_duration_seconds)
        payload = primary_result.copy()
        payload["collection"] = collection_name
        payload["results"] = [
            {"id": point.get("id"), "score": point.get("score")}
            for point in payload.get("points", [])
        ]
        return payload


def _build_client_from_url(url: str) -> QdrantClient:
    return QdrantClient(url=url)


if __name__ == "__main__":
    proxy = SearchProxy(client=_build_client_from_url("http://localhost:6333"))
    print(proxy.search("example query", 5))
