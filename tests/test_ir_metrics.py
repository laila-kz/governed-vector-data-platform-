import pytest

from evaluation.metrics import benchmark_latency, evaluate_rankings, percentile


def test_evaluate_rankings_computes_recall_ndcg_and_mrr() -> None:
    metrics = evaluate_rankings(
        "test-model",
        {"q1": ["a", "b"], "q2": ["c"]},
        {"q1": ["x", "a", "b"], "q2": ["c", "y"]},
    )

    assert metrics.query_count == 2
    assert metrics.recall_at_5 == 1.0
    assert metrics.recall_at_10 == 1.0
    expected_q1 = (1 / 2.0 + 1 / 1.5849625007) / (1 + 1 / 1.5849625007)
    assert metrics.ndcg_at_10 == pytest.approx((expected_q1 + 1.0) / 2.0)
    assert metrics.mrr == (0.5 + 1.0) / 2


def test_missing_rankings_count_as_empty_results() -> None:
    metrics = evaluate_rankings("test-model", {"q1": ["a"]}, {})

    assert metrics.recall_at_5 == 0.0
    assert metrics.recall_at_10 == 0.0
    assert metrics.ndcg_at_10 == 0.0
    assert metrics.mrr == 0.0


def test_rankings_accept_qdrant_style_result_mappings() -> None:
    metrics = evaluate_rankings(
        "test-model",
        {"q1": ["doc-1"]},
        {"q1": [{"id": "doc-1", "score": 0.9}]},
    )

    assert metrics.recall_at_10 == 1.0
    assert metrics.mrr == 1.0


def test_percentile_interpolates_and_latency_benchmark_records_summary() -> None:
    assert percentile([1, 2, 3, 4], 50) == 2.5
    calls: list[str] = []
    result = benchmark_latency("test-model", ["one", "two"], calls.append)

    assert result.model == "test-model"
    assert result.sample_count == 2
    assert result.average_ms >= 0.0
    assert result.p50_ms >= 0.0
    assert result.p95_ms >= result.p50_ms
    assert calls == ["one", "two"]