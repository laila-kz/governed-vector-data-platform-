import numpy as np
import pytest

from evaluation.drift_detector import analyze_drift


def test_drift_analyzer_reports_dimensions_and_preserved_neighbors() -> None:
    ids = [f"chunk-{index}" for index in range(6)]
    model_v1 = {chunk_id: [float(index), 1.0, 0.0] for index, chunk_id in enumerate(ids)}
    model_v2 = {
        chunk_id: [float(index), 1.0, 0.0, 0.0]
        for index, chunk_id in enumerate(ids)
    }

    report = analyze_drift(model_v1, model_v2, neighborhood_k=2)

    assert report.shared_chunk_count == 6
    assert report.sampled_chunk_count == 6
    assert report.model_v1_dimension == 3
    assert report.model_v2_dimension == 4
    assert report.neighborhood_jaccard_mean == pytest.approx(1.0)
    assert report.collapse_detected is False
    assert report.semantic_shift_detected is False


def test_drift_analyzer_samples_deterministically_and_detects_collapse() -> None:
    model_v1 = {
        f"chunk-{index}": [float(index), 1.0]
        for index in range(8)
    }
    model_v2 = {
        f"chunk-{index}": [1.0, 1.0, 1.0]
        for index in range(8)
    }

    first = analyze_drift(model_v1, model_v2, sample_size=5, seed=7)
    second = analyze_drift(model_v1, model_v2, sample_size=5, seed=7)

    assert first == second
    assert first.sampled_chunk_count == 5
    assert first.collapse_detected is True
    assert first.semantic_shift_detected is True


def test_drift_analyzer_rejects_missing_shared_chunks_and_zero_vectors() -> None:
    with pytest.raises(ValueError, match="no shared"):
        analyze_drift({"a": [1.0]}, {"b": [1.0, 0.0]})
    with pytest.raises(ValueError, match="zero vectors"):
        analyze_drift({"a": [0.0, 0.0]}, {"a": [1.0, 0.0]})


def test_drift_analyzer_accepts_numpy_input() -> None:
    report = analyze_drift(
        {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])},
        {"a": np.array([1.0, 0.0, 0.0]), "b": np.array([0.0, 1.0, 0.0])},
        neighborhood_k=1,
    )

    assert report.neighborhood_jaccard_mean == pytest.approx(1.0)