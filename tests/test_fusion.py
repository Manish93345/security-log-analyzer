"""Unit tests for Reciprocal Rank Fusion — pure maths, no dependencies."""

from __future__ import annotations

import pytest

from slrag.retrieval.fusion import DEFAULT_RRF_K, reciprocal_rank_fusion


def test_document_found_by_both_retrievers_wins():
    """The core promise of a hybrid: agreement between retrievers beats a lone first place."""
    fused = reciprocal_rank_fusion({"dense": ["a", "b"], "bm25": ["a", "c"]})

    assert fused[0].chunk_id == "a"
    assert fused[0].ranks == {"dense": 1, "bm25": 1}
    assert fused[0].score == pytest.approx(2 / (DEFAULT_RRF_K + 1))
    assert {hit.chunk_id for hit in fused} == {"a", "b", "c"}


def test_score_follows_the_rrf_formula():
    fused = reciprocal_rank_fusion({"dense": ["x", "y", "z"]}, k=10)

    assert [hit.chunk_id for hit in fused] == ["x", "y", "z"]
    assert fused[0].score == pytest.approx(1 / 11)
    assert fused[1].score == pytest.approx(1 / 12)
    assert fused[2].score == pytest.approx(1 / 13)


def test_two_weak_agreements_beat_one_strong_single_hit():
    """b is 2nd on both lists, c is 1st on one — b should still win."""
    fused = reciprocal_rank_fusion({"dense": ["c", "b"], "bm25": ["b"]}, k=60)
    assert fused[0].chunk_id == "b"
    assert fused[0].score == pytest.approx(1 / 62 + 1 / 61)


def test_k_damps_the_top_rank_advantage():
    tight = reciprocal_rank_fusion({"dense": ["a"], "bm25": ["b"]}, k=1)
    loose = reciprocal_rank_fusion({"dense": ["a"], "bm25": ["b"]}, k=1000)
    # Both are single hits at rank 1, so k only changes the magnitude, never the order.
    assert tight[0].score > loose[0].score
    assert [hit.chunk_id for hit in tight] == [hit.chunk_id for hit in loose]


def test_weights_shift_the_order():
    rankings = {"dense": ["a"], "bm25": ["b"]}
    fused = reciprocal_rank_fusion(rankings, weights={"dense": 1.0, "bm25": 3.0})
    assert fused[0].chunk_id == "b"
    assert fused[0].score == pytest.approx(3 / 61)


def test_zero_weight_drops_a_retriever():
    fused = reciprocal_rank_fusion({"dense": ["a"], "bm25": ["b"]}, weights={"bm25": 0.0})
    assert [hit.chunk_id for hit in fused] == ["a"]


def test_duplicate_ids_in_one_ranking_are_counted_once():
    fused = reciprocal_rank_fusion({"dense": ["a", "a", "b"]})
    assert fused[0].chunk_id == "a"
    assert fused[0].score == pytest.approx(1 / 61)
    assert fused[0].ranks == {"dense": 1}


def test_ties_break_deterministically_by_id():
    """Equal scores must not depend on dict iteration order."""
    first = reciprocal_rank_fusion({"dense": ["b"], "bm25": ["a"]})
    second = reciprocal_rank_fusion({"bm25": ["a"], "dense": ["b"]})

    assert [hit.chunk_id for hit in first] == ["a", "b"]
    assert [hit.chunk_id for hit in first] == [hit.chunk_id for hit in second]


def test_limit_truncates_after_fusion():
    fused = reciprocal_rank_fusion({"dense": ["a", "b", "c", "d"]}, limit=2)
    assert [hit.chunk_id for hit in fused] == ["a", "b"]


def test_sources_property_is_sorted():
    fused = reciprocal_rank_fusion({"bm25": ["a"], "dense": ["a"]})
    assert fused[0].sources == ("bm25", "dense")


def test_empty_input_returns_nothing():
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"dense": [], "bm25": []}) == []


def test_non_positive_k_is_rejected():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"dense": ["a"]}, k=0)
