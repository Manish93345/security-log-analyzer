"""Unit tests for the BM25 lexical retriever and its log-aware tokeniser."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from slrag.retrieval.bm25 import BM25Index, tokenize

# --------------------------------------------------------------------------- #
# tokenisation
# --------------------------------------------------------------------------- #


def test_tokenize_is_case_folded():
    assert tokenize("ConsoleLogin")[0] == "consolelogin"


def test_tokenize_splits_camel_case_so_english_questions_match_actions():
    tokens = tokenize("CreateAccessKey")
    assert "create" in tokens
    assert "access" in tokens
    assert "key" in tokens


def test_tokenize_keeps_ipv4_addresses_whole():
    """The one thing BM25 must never do here: make 203.0.113.44 match 10.0.1.23."""
    tokens = tokenize("Failed password from 203.0.113.44 port 51234")
    assert "203.0.113.44" in tokens
    assert "203" not in tokens
    assert "113" not in tokens


def test_tokenize_splits_paths_so_the_binary_name_is_searchable():
    tokens = tokenize("COMMAND=/usr/bin/systemctl restart nginx")
    assert "systemctl" in tokens
    assert "nginx" in tokens


def test_tokenize_keeps_arns_as_one_token_but_also_splits_them():
    tokens = tokenize("arn:aws:iam::123456789012:user/bob")
    assert "arn:aws:iam::123456789012:user/bob" in tokens
    assert "aws" in tokens
    assert "bob" in tokens


def test_tokenize_keeps_short_meaningful_tokens():
    assert "s3" in tokenize("GetBucketAcl s3")
    assert "ec2" in tokenize("ec2:DescribeInstances")


def test_tokenize_handles_empty_and_punctuation_only_input():
    assert tokenize("") == []
    assert tokenize("   ") == []
    assert tokenize("!!! ??? ---") == []


# --------------------------------------------------------------------------- #
# index + search
# --------------------------------------------------------------------------- #

DOCS = [
    (
        "w-1",
        "auth.log Failed password for invalid user admin from 203.0.113.44 | "
        "sudo COMMAND=/usr/bin/systemctl restart nginx | failures=12",
    ),
    ("w-2", "auth.log Accepted password for alice from 10.0.1.23 | ConsoleLogin MFAUsed=No"),
    ("w-3", "cloudtrail CreateAccessKey arn:aws:iam::123456789012:user/bob"),
]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index.from_pairs(DOCS)


def test_relevant_window_ranks_first(index):
    hits = index.search("failed sudo commands on the web server", k=5)
    assert hits, "the query shares terms with w-1 and must return something"
    assert hits[0].chunk_id == "w-1"
    assert hits[0].rank == 1


def test_rare_literal_ip_is_retrieved_exactly(index):
    hits = index.search("203.0.113.44", k=5)
    assert [hit.chunk_id for hit in hits] == ["w-1"]


def test_camel_case_action_name_query_matches(index):
    hits = index.search("who created access keys", k=5)
    assert hits[0].chunk_id == "w-3"


def test_unmatched_query_returns_nothing(index):
    assert index.search("kubernetes helm chart rollout", k=5) == []


def test_empty_query_returns_nothing(index):
    assert index.search("", k=5) == []
    assert index.search("   ", k=5) == []


def test_empty_index_is_safe():
    assert BM25Index().search("anything", k=5) == []
    assert len(BM25Index()) == 0


def test_k_limits_the_result_count(index):
    hits = index.search("password", k=2)
    assert len(hits) <= 2


def test_ranks_are_contiguous_from_one(index):
    hits = index.search("password from", k=5)
    assert [hit.rank for hit in hits] == list(range(1, len(hits) + 1))


def test_scores_are_descending(index):
    scores = [hit.score for hit in index.search("password alice bob", k=5)]
    assert scores == sorted(scores, reverse=True)


def test_max_tokens_bounds_the_document_length():
    text = " ".join(["alpha"] * 50) + " omega"
    truncated = BM25Index.from_pairs([("w-1", text)], max_tokens=5)
    assert truncated.search("omega", k=1) == []
    assert truncated.search("alpha", k=1)[0].chunk_id == "w-1"


def test_from_chunks_prefers_search_text():
    chunks = [
        SimpleNamespace(chunk_id="w-1", search_text="sudo nginx", text="unrelated prose"),
        SimpleNamespace(chunk_id="w-2", search_text="", text="lambda invocation"),
    ]
    index = BM25Index.from_chunks(chunks)
    assert index.search("nginx", k=1)[0].chunk_id == "w-1"
    # empty search_text falls back to text
    assert index.search("lambda", k=1)[0].chunk_id == "w-2"


def test_len_and_vocabulary(index):
    assert len(index) == 3
    assert index.vocabulary_size > 0
