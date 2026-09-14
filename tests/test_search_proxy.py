import time

from api.search_proxy import SearchProxy


class FakeEmbedder:
    def __call__(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeQdrant:
    def __init__(self):
        self.calls = []
        self.shadow_calls = []

    def search(self, collection_name, query_vector, limit, **kwargs):
        payload = {
            "collection": collection_name,
            "limit": limit,
            "query_vector": query_vector,
            "points": [{"id": "p1", "score": 0.92}],
        }
        self.calls.append(payload)
        return payload

    def search_shadow(self, collection_name, query_vector, limit, **kwargs):
        payload = {
            "collection": collection_name,
            "limit": limit,
            "query_vector": query_vector,
            "points": [{"id": "shadow-1", "score": 0.88}],
        }
        self.shadow_calls.append(payload)
        return payload


def test_search_uses_live_alias_and_returns_results() -> None:
    client = FakeQdrant()
    proxy = SearchProxy(
        client,
        config={
            "active_alias": "vectors_live",
            "active_model": "bge-small-en-v1.5",
            "active_version": "1.0.0",
            "shadow_collection": "vectors_shadow",
            "shadow_model": "bge-small-en-v1.5",
            "shadow_version": "2.0.0",
            "traffic_split_ratio": 1.0,
            "shadow_read_enabled": False,
        },
        embedder=FakeEmbedder(),
    )

    result = proxy.search("test query", 3)

    assert result["collection"] == "vectors_live"
    assert result["results"][0]["id"] == "p1"
    assert client.calls[0]["collection"] == "vectors_live"


def test_shadow_read_runs_in_background_without_blocking() -> None:
    client = FakeQdrant()
    proxy = SearchProxy(
        client,
        config={
            "active_alias": "vectors_live",
            "active_model": "bge-small-en-v1.5",
            "active_version": "1.0.0",
            "shadow_collection": "vectors_shadow",
            "shadow_model": "bge-small-en-v1.5",
            "shadow_version": "2.0.0",
            "traffic_split_ratio": 1.0,
            "shadow_read_enabled": True,
        },
        embedder=FakeEmbedder(),
    )

    result = proxy.search("shadow test", 3)
    deadline = time.time() + 0.5
    while time.time() < deadline and len(client.shadow_calls) == 0:
        time.sleep(0.01)

    assert result["collection"] == "vectors_live"
    assert len(client.shadow_calls) == 1
    assert client.shadow_calls[0]["collection"] == "vectors_shadow"


def test_traffic_split_is_deterministic() -> None:
    client = FakeQdrant()
    proxy = SearchProxy(
        client,
        config={
            "active_alias": "vectors_live",
            "active_model": "bge-small-en-v1.5",
            "active_version": "1.0.0",
            "shadow_collection": "vectors_shadow",
            "shadow_model": "bge-small-en-v1.5",
            "shadow_version": "2.0.0",
            "traffic_split_ratio": 0.5,
            "shadow_read_enabled": False,
        },
        embedder=FakeEmbedder(),
    )

    first = proxy._route_collection("query-alpha")
    second = proxy._route_collection("query-alpha")
    assert first == second
    assert first in {"vectors_live", "vectors_shadow"}
