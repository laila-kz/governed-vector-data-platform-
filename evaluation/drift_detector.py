"""Analyze representation drift between two embedding spaces."""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class CosineDistribution:
    """Summary of off-diagonal pairwise cosine similarities."""

    mean: float
    median: float
    p05: float
    p95: float
    standard_deviation: float


@dataclass(frozen=True)
class DriftReport:
    """Vector-space drift measurements for shared chunks."""

    shared_chunk_count: int
    sampled_chunk_count: int
    model_v1_dimension: int
    model_v2_dimension: int
    cosine_distribution_v1: CosineDistribution
    cosine_distribution_v2: CosineDistribution
    cosine_mean_shift: float
    centroid_norm_v1: float
    centroid_norm_v2: float
    centroid_norm_shift: float
    neighborhood_k: int
    neighborhood_jaccard_mean: float
    neighborhood_jaccard_median: float
    neighborhood_jaccard_p05: float
    neighborhood_jaccard_p95: float
    collapse_detected: bool
    semantic_shift_detected: bool


def _as_matrix(vectors: Mapping[str, Sequence[float]], chunk_ids: Sequence[str]) -> np.ndarray:
    try:
        matrix = np.asarray([vectors[chunk_id] for chunk_id in chunk_ids], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("vectors must map every shared chunk to a numeric sequence") from error
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("vectors must form a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("vectors must contain only finite values")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms == 0):
        raise ValueError("zero vectors cannot be used for cosine drift analysis")
    return matrix / norms[:, None]


def _distribution(values: np.ndarray) -> CosineDistribution:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1)
    return CosineDistribution(
        mean=float(np.mean(flattened)),
        median=float(np.percentile(flattened, 50)),
        p05=float(np.percentile(flattened, 5)),
        p95=float(np.percentile(flattened, 95)),
        standard_deviation=float(np.std(flattened)),
    )


def _pairwise_cosine_distribution(normalized_vectors: np.ndarray) -> CosineDistribution:
    similarities = normalized_vectors @ normalized_vectors.T
    upper_triangle = similarities[np.triu_indices(similarities.shape[0], k=1)]
    if upper_triangle.size == 0:
        upper_triangle = np.array([1.0])
    return _distribution(upper_triangle)


def _nearest_neighbors(normalized_vectors: np.ndarray, k: int) -> list[set[int]]:
    similarities = normalized_vectors @ normalized_vectors.T
    np.fill_diagonal(similarities, -np.inf)
    neighbor_count = min(k, max(0, normalized_vectors.shape[0] - 1))
    if neighbor_count == 0:
        return [set() for _ in range(normalized_vectors.shape[0])]
    indices = np.argpartition(-similarities, neighbor_count - 1, axis=1)[:, :neighbor_count]
    return [set(row.tolist()) for row in indices]


def _summary(values: Sequence[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    return (
        float(np.mean(array)),
        float(np.percentile(array, 50)),
        float(np.percentile(array, 5)),
        float(np.percentile(array, 95)),
    )


def analyze_drift(
    model_v1_vectors: Mapping[str, Sequence[float]],
    model_v2_vectors: Mapping[str, Sequence[float]],
    *,
    sample_size: int = 1000,
    neighborhood_k: int = 10,
    seed: int = 0,
    collapse_mean_threshold: float = 0.95,
    collapse_std_threshold: float = 0.05,
    semantic_shift_threshold: float = 0.10,
) -> DriftReport:
    """Measure geometry and neighborhood drift over shared chunk IDs.

    Since model dimensions differ, cosine similarities are calculated within
    each model space. Cross-model alignment is measured through shared-chunk
    k-NN Jaccard overlap and centroid concentration shifts.
    """
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")
    if neighborhood_k <= 0:
        raise ValueError("neighborhood_k must be greater than zero")
    shared_ids = sorted(set(model_v1_vectors) & set(model_v2_vectors))
    if not shared_ids:
        raise ValueError("model vector maps have no shared chunk IDs")
    if sample_size < len(shared_ids):
        shared_ids = random.Random(seed).sample(shared_ids, sample_size)
    sampled_ids = sorted(shared_ids)

    normalized_v1 = _as_matrix(model_v1_vectors, sampled_ids)
    normalized_v2 = _as_matrix(model_v2_vectors, sampled_ids)
    cosine_v1 = _pairwise_cosine_distribution(normalized_v1)
    cosine_v2 = _pairwise_cosine_distribution(normalized_v2)

    neighbors_v1 = _nearest_neighbors(normalized_v1, neighborhood_k)
    neighbors_v2 = _nearest_neighbors(normalized_v2, neighborhood_k)
    jaccards = [
        len(left & right) / len(left | right) if left | right else 1.0
        for left, right in zip(neighbors_v1, neighbors_v2)
    ]
    jaccard_mean, jaccard_median, jaccard_p05, jaccard_p95 = _summary(jaccards)

    raw_v1 = np.asarray([model_v1_vectors[chunk_id] for chunk_id in sampled_ids], dtype=np.float64)
    raw_v2 = np.asarray([model_v2_vectors[chunk_id] for chunk_id in sampled_ids], dtype=np.float64)
    centroid_norm_v1 = float(np.linalg.norm(np.mean(raw_v1, axis=0)))
    centroid_norm_v2 = float(np.linalg.norm(np.mean(raw_v2, axis=0)))
    centroid_norm_shift = abs(centroid_norm_v2 - centroid_norm_v1)
    collapse_detected = (
        cosine_v2.mean >= collapse_mean_threshold
        and cosine_v2.standard_deviation <= collapse_std_threshold
    )
    semantic_shift_detected = (
        abs(cosine_v2.mean - cosine_v1.mean) >= semantic_shift_threshold
        or centroid_norm_shift >= semantic_shift_threshold
        or jaccard_mean < 1.0 - semantic_shift_threshold
    )
    return DriftReport(
        shared_chunk_count=len(set(model_v1_vectors) & set(model_v2_vectors)),
        sampled_chunk_count=len(sampled_ids),
        model_v1_dimension=int(normalized_v1.shape[1]),
        model_v2_dimension=int(normalized_v2.shape[1]),
        cosine_distribution_v1=cosine_v1,
        cosine_distribution_v2=cosine_v2,
        cosine_mean_shift=float(cosine_v2.mean - cosine_v1.mean),
        centroid_norm_v1=centroid_norm_v1,
        centroid_norm_v2=centroid_norm_v2,
        centroid_norm_shift=centroid_norm_shift,
        neighborhood_k=min(neighborhood_k, max(0, len(sampled_ids) - 1)),
        neighborhood_jaccard_mean=jaccard_mean,
        neighborhood_jaccard_median=jaccard_median,
        neighborhood_jaccard_p05=jaccard_p05,
        neighborhood_jaccard_p95=jaccard_p95,
        collapse_detected=collapse_detected,
        semantic_shift_detected=semantic_shift_detected,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-v1-vectors", type=Path, required=True)
    parser.add_argument("--model-v2-vectors", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()
    model_v1 = json.loads(args.model_v1_vectors.read_text(encoding="utf-8-sig"))
    model_v2 = json.loads(args.model_v2_vectors.read_text(encoding="utf-8-sig"))
    report = analyze_drift(model_v1, model_v2, sample_size=args.sample_size, neighborhood_k=args.k)
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()