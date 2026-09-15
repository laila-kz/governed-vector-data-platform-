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
