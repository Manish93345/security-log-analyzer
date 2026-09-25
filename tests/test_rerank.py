"""Unit tests for the reranker — no model download, no network.

The real cross-encoder is exercised by ``test_fastembed_reranker_orders_by_relevance``, which
is marked ``integration`` (it downloads ~23 MB on first run) and therefore excluded from the
default ``python scripts/tasks.py test``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from slrag.config import Settings
from slrag.retrieval.rerank import (
    DEFAULT_FASTEMBED_MODEL,
    FLASHRANK_DEFAULT_MODEL,
    FastEmbedReranker,
    NullReranker,
    RerankedHit,
    get_reranker,
    resolve_fastembed_model,
)


def settings_for(tmp_path: Path, **overrides) -> Settings:
    """Settings pointed at a temporary directory, so no real data/ is touched."""
    base = {
        "data_dir": tmp_path,
        "raw_dir": tmp_path / "raw",
        "processed_dir": tmp_path / "processed",
        "chroma_dir": tmp_path / "chroma",
        "sqlite_path": tmp_path / "events.sqlite",
        "cache_dir": tmp_path / "cache",
        "reports_dir": tmp_path / "reports",
    }
    base.update(overrides)
    return Settings(**base)


# --------------------------------------------------------------------------- #
# model-name resolution
# --------------------------------------------------------------------------- #


def test_short_name_resolves_to_the_fastembed_repo_id():
    assert resolve_fastembed_model("ms-marco-MiniLM-L-6-v2") == DEFAULT_FASTEMBED_MODEL


def test_resolution_is_case_insensitive():
    assert resolve_fastembed_model("MS-MARCO-MINILM-L-6-V2") == DEFAULT_FASTEMBED_MODEL


def test_full_repo_id_passes_through_unchanged():
    """Idempotent: resolving twice must not corrupt an already-resolved name."""
    repo_id = "Xenova/ms-marco-MiniLM-L-6-v2"
    assert resolve_fastembed_model(repo_id) == repo_id
    assert resolve_fastembed_model(resolve_fastembed_model(repo_id)) == repo_id


def test_unknown_short_name_is_left_alone():
    assert resolve_fastembed_model("my-own-reranker") == "my-own-reranker"


def test_empty_name_falls_back_to_the_default():
    assert resolve_fastembed_model("") == DEFAULT_FASTEMBED_MODEL
    assert resolve_fastembed_model(None) == DEFAULT_FASTEMBED_MODEL
    assert resolve_fastembed_model("   ") == DEFAULT_FASTEMBED_MODEL


def test_other_known_aliases_resolve():
    assert resolve_fastembed_model("bge-reranker-base") == "BAAI/bge-reranker-base"
    assert resolve_fastembed_model("jina-reranker-v1-tiny-en") == "jinaai/jina-reranker-v1-tiny-en"


def test_flashrank_default_is_not_the_dead_model():
    """ms-marco-MiniLM-L-6-v2.zip 404s upstream — the default must not be it."""
    assert FLASHRANK_DEFAULT_MODEL != "ms-marco-MiniLM-L-6-v2"
    assert FLASHRANK_DEFAULT_MODEL == "ms-marco-TinyBERT-L-2-v2"


# --------------------------------------------------------------------------- #
# NullReranker
# --------------------------------------------------------------------------- #


CANDIDATES = [("w-1", "sudo nginx"), ("w-2", "accepted password"), ("w-3", "create access key")]


def test_null_reranker_keeps_the_incoming_order():
    hits = NullReranker().rerank("anything", CANDIDATES, top_n=3)
    assert [hit.chunk_id for hit in hits] == ["w-1", "w-2", "w-3"]
    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert all(hit.score == 0.0 for hit in hits)


def test_null_reranker_respects_top_n():
    hits = NullReranker().rerank("anything", CANDIDATES, top_n=2)
    assert len(hits) == 2


def test_null_reranker_handles_empty_input():
    assert NullReranker().rerank("anything", [], top_n=5) == []


def test_null_reranker_is_marked_unavailable():
    assert NullReranker().available is False


# --------------------------------------------------------------------------- #
# backend selection
# --------------------------------------------------------------------------- #


def test_disabled_by_setting_returns_the_null_reranker(tmp_path):
    reranker = get_reranker(settings_for(tmp_path, rerank_enabled=False))
    assert isinstance(reranker, NullReranker)


def test_explicit_enabled_false_returns_the_null_reranker(tmp_path):
    reranker = get_reranker(settings_for(tmp_path), enabled=False)
    assert isinstance(reranker, NullReranker)


def test_backend_none_returns_the_null_reranker(tmp_path):
    reranker = get_reranker(settings_for(tmp_path, rerank_backend="none"))
    assert isinstance(reranker, NullReranker)


def test_unknown_backend_degrades_instead_of_raising(tmp_path):
    reranker = get_reranker(settings_for(tmp_path, rerank_backend="does-not-exist"))
    assert isinstance(reranker, NullReranker)


def test_default_backend_is_fastembed():
    """The shipped default must be the backend whose model download actually works."""
    assert Settings().rerank_backend == "fastembed"


def test_fastembed_reranker_resolves_its_model_without_loading_it():
    """Constructor arguments are validated before any download is attempted."""
    assert FastEmbedReranker.name == "fastembed"
    assert resolve_fastembed_model(Settings().rerank_model) == DEFAULT_FASTEMBED_MODEL


def test_reranked_hit_is_comparable_by_fields():
    hit = RerankedHit(chunk_id="w-1", score=0.5, rank=1)
    assert hit.chunk_id == "w-1"
    assert hit.score == 0.5
    assert hit.rank == 1


# --------------------------------------------------------------------------- #
# real cross-encoder (downloads ~23 MB on first run)
# --------------------------------------------------------------------------- #


@pytest.mark.integration
def test_fastembed_reranker_orders_by_relevance():
    reranker = FastEmbedReranker()
    hits = reranker.rerank(
        "failed sudo commands on the web server",
        [
            ("w-sudo", "sudo COMMAND=/usr/bin/systemctl restart nginx"),
            ("w-login", "Accepted password for alice from 10.0.1.23"),
            ("w-key", "CreateAccessKey arn:aws:iam::123456789012:user/bob"),
        ],
        top_n=3,
    )

    assert [hit.rank for hit in hits] == [1, 2, 3]
    assert hits[0].chunk_id == "w-sudo", "the sudo window must outrank the unrelated ones"
    scores = [hit.score for hit in hits]
    assert scores == sorted(scores, reverse=True)
