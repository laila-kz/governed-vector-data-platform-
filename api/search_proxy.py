"""Qdrant search proxy with routing policy and optional shadow reads."""

from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml
from qdrant_client import QdrantClient

DEFAULT_POLICY_PATH = Path(__file__).resolve().parents[1] / "configs" / "routing_policy.yaml"


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
    ) -> None:
        self.client = client
        self.config = dict(_load_policy() if config is None else config)
        self.embedder = embedder or (lambda texts: [[0.1, 0.2, 0.3] for _ in texts])

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

    def _shadow_read(self, query: str, query_vector: list[float], limit: int) -> None:
        shadow_collection = self.config.get("shadow_collection")
        if not self.config.get("shadow_read_enabled", False) or not shadow_collection:
            return
        def _task() -> None:
            try:
                self.client.search_shadow(
                    collection_name=str(shadow_collection),
                    query_vector=query_vector,
                    limit=limit,
                    score_threshold=0.0,
                )
            except Exception:
                pass

        thread = threading.Thread(target=_task, daemon=True)
        thread.start()

    def search(self, query: str, top_k: int) -> dict[str, Any]:
        """Embed and query Qdrant using the configured active/shadow collection policy."""
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self._embed_query(query)
        active_alias = str(self.config.get("active_alias", "vectors_live"))
        collection_name = self._route_collection(query)
        if collection_name == active_alias:
            primary_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=0.0,
            )
        else:
            primary_result = self.client.search(
                collection_name=collection_name,
                query_vector=query_vector,
                limit=top_k,
                score_threshold=0.0,
            )

        self._shadow_read(query, query_vector, top_k)
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
