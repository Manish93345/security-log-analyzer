"""Reciprocal Rank Fusion.

RRF is the reason this project does not need to *tune* a dense-vs-lexical weight: it combines
rankings using only their **ranks**, not their scores. BM25 scores are unbounded and cosine
similarities live in [-1, 1], so any score-level blend would need per-corpus calibration that
breaks the moment the corpus changes. Ranks are comparable by construction.

For each document::

    score(d) = sum over retrievers r that returned d of  weight_r / (k + rank_r(d))

``k`` (default 60, from the original Cormack et al. paper) damps the influence of the very top
ranks so a single retriever cannot unilaterally dictate the order. A document returned by both
retrievers therefore outranks a document returned by only one — which is exactly the behaviour
we want from a hybrid.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

#: The RRF damping constant. 60 is the value used in the original paper and the industry default.
DEFAULT_RRF_K = 60


@dataclass(frozen=True, slots=True)
class FusionHit:
    """A fused document: its combined score and the rank each retriever gave it."""

    chunk_id: str
    score: float
    ranks: dict[str, int]

    @property
    def sources(self) -> tuple[str, ...]:
        """Names of the retrievers that found this document (sorted, for stable output)."""
        return tuple(sorted(self.ranks))


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    *,
    k: int = DEFAULT_RRF_K,
    weights: Mapping[str, float] | None = None,
    limit: int | None = None,
) -> list[FusionHit]:
    """Fuse ``{retriever_name: [chunk_id, ...]}`` into one ranking.

    Args:
        rankings: ordered candidate ids per retriever, best first.
        k: RRF damping constant (must be positive). Larger ``k`` flattens the curve.
        weights: optional per-retriever multiplier; a weight of ``0`` drops that retriever.
        limit: keep only the first ``limit`` fused hits.

    Returns:
        :class:`FusionHit` objects sorted by descending score, ties broken by ``chunk_id``
        so the result is fully deterministic.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    scores: dict[str, float] = {}
    ranks: dict[str, dict[str, int]] = {}

    for source, ids in rankings.items():
        weight = 1.0 if weights is None else float(weights.get(source, 1.0))
        if weight == 0.0:
            continue
        seen: set[str] = set()
        for position, chunk_id in enumerate(ids, start=1):
            if chunk_id in seen:
                # A retriever that returns the same id twice must not be rewarded twice;
                # the best (first) rank is the one that counts.
                continue
            seen.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + weight / (k + position)
            ranks.setdefault(chunk_id, {})[source] = position

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if limit is not None:
        ordered = ordered[: max(0, limit)]
    return [
        FusionHit(chunk_id=chunk_id, score=score, ranks=ranks[chunk_id])
        for chunk_id, score in ordered
    ]


__all__ = ["DEFAULT_RRF_K", "FusionHit", "reciprocal_rank_fusion"]
