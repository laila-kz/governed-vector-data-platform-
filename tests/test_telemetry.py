from fastapi.testclient import TestClient

from api.main import app


def test_metrics_endpoint_exposes_prometheus_metrics() -> None:
    client = TestClient(app)

    response = client.get("/metrics")

    assert response.status_code == 200
    assert "search_latency_seconds" in response.text
    assert "http_requests_total" in response.text


def test_search_route_records_metrics_and_returns_payload() -> None:
    client = TestClient(app)

    response = client.post("/v1/search", json={"query": "what is lineage", "top_k": 3})

    assert response.status_code == 200
    payload = response.json()
    assert payload["collection"] == "vectors_live"
    assert payload["results"]

    metrics_res = client.get("/metrics")
    assert "embedding_tokens_total" in metrics_res.text
    assert "search_latency_seconds_bucket" in metrics_res.text


def test_dynamic_routing_policy_update_without_restart() -> None:
    client = TestClient(app)

    # Initial search goes to vectors_live
    res1 = client.post("/v1/search", json={"query": "alpha query", "top_k": 2})
    assert res1.status_code == 200
    assert res1.json()["collection"] == "vectors_live"

    # Update routing policy dynamically
    update_res = client.post(
        "/v1/admin/routing",
        json={
            "active_alias": "vectors_live_v2",
            "active_model": "bge-base-en-v1.5",
            "traffic_split_ratio": 1.0,
        },
    )
    assert update_res.status_code == 200
    assert update_res.json()["status"] == "updated"

    # Immediate query reflects updated active alias without restarting
    res2 = client.post("/v1/search", json={"query": "alpha query", "top_k": 2})
    assert res2.status_code == 200
    assert res2.json()["collection"] == "vectors_live_v2"

    # Revert back
    client.post(
        "/v1/admin/routing",
        json={
            "active_alias": "vectors_live",
            "active_model": "bge-small-en-v1.5",
            "traffic_split_ratio": 1.0,
        },
    )


def test_staleness_endpoint_records_stale_vector_gauge() -> None:
    client = TestClient(app)

    res = client.get("/v1/catalog/staleness")
    assert res.status_code == 200
    data = res.json()
    assert "total_vectors" in data
    assert "stale_vectors" in data

    metrics_res = client.get("/metrics")
    assert "stale_vector_gauge" in metrics_res.text

