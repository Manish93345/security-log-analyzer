"""Tests for incident-window chunking (Phase 2 acceptance criteria)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from slrag.ingest.chunker import (
    SENSITIVE_ACTIONS,
    build_windows,
    compute_risk_score,
    is_root_principal,
    summarize_chunks,
)
from slrag.ingest.schema import Event

BASE = datetime(2026, 8, 14, 2, 0, 0, tzinfo=timezone.utc)


def make_event(
    minutes: float = 0,
    *,
    action: str = "GetObject",
    status: str = "success",
    principal: str = "arn:aws:iam::123456789012:user/alice",
    src_ip: str = "10.0.1.23",
    source: str = "cloudtrail",
    event_id: str | None = None,
) -> Event:
    ts = BASE + timedelta(minutes=minutes)
    return Event(
        event_id=event_id or f"ct-{int(minutes * 1000):012d}",
        ts=ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
        source=source,  # type: ignore[arg-type]
        action=action,
        status=status,  # type: ignore[arg-type]
        principal=principal,
        src_ip=src_ip,
    )


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #


def test_empty_input_returns_no_chunks():
    assert build_windows([]) == []


def test_events_within_gap_merge_into_one_window():
    events = [make_event(0), make_event(5), make_event(14.9)]
    chunks = build_windows(events, gap_minutes=15)
    assert len(chunks) == 1
    chunk = chunks[0]
    assert chunk.n_events == 3
    assert chunk.chunk_id == "w-00001"
    assert chunk.ts_start == "2026-08-14T02:00:00Z"
    assert chunk.ts_end == "2026-08-14T02:14:54Z"


def test_gap_larger_than_threshold_splits_windows():
    events = [make_event(0), make_event(30)]
    chunks = build_windows(events, gap_minutes=15)
    assert len(chunks) == 2
    assert chunks[0].n_events == 1 and chunks[1].n_events == 1


def test_gap_is_measured_between_consecutive_events_not_from_start():
    """0 -> 14 -> 28 minutes must stay one window (each hop is <= 15 min)."""
    chunks = build_windows([make_event(0), make_event(14), make_event(28)], gap_minutes=15)
    assert len(chunks) == 1 and chunks[0].n_events == 3


def test_interleaved_streams_do_not_fragment_windows():
    """Regression: another principal's event landing mid-session must not split it.

    Merging over a single globally-sorted timeline produced one-event windows on real
    interleaved traffic, which silently destroyed retrieval quality.
    """
    alice = [
        make_event(i * 0.5, principal="arn:aws:iam::1:user/alice", src_ip="10.0.0.1")
        for i in range(10)
    ]
    bob = [
        make_event(i * 0.5 + 0.1, principal="arn:aws:iam::1:user/bob", src_ip="10.0.0.2")
        for i in range(10)
    ]
    interleaved = [event for pair in zip(alice, bob) for event in pair]

    chunks = build_windows(interleaved, gap_minutes=15)
    assert len(chunks) == 2, "interleaving must not fragment a session"
    assert sorted(chunk.n_events for chunk in chunks) == [10, 10]
    assert [chunk.principal for chunk in chunks] == [
        "arn:aws:iam::1:user/alice",
        "arn:aws:iam::1:user/bob",
    ]


def test_chunk_ids_are_chronological_across_interleaved_streams():
    alice = [make_event(30, principal="arn:aws:iam::1:user/alice", src_ip="10.0.0.1")]
    bob = [make_event(0, principal="arn:aws:iam::1:user/bob", src_ip="10.0.0.2")]
    chunks = build_windows(bob + alice)
    assert chunks[0].ts_start < chunks[1].ts_start
    assert [chunk.chunk_id for chunk in chunks] == ["w-00001", "w-00002"]


def test_different_principals_never_share_a_window():
    events = [
        make_event(0, principal="arn:aws:iam::1:user/alice"),
        make_event(1, principal="arn:aws:iam::1:user/bob"),
    ]
    chunks = build_windows(events, gap_minutes=15)
    assert len(chunks) == 2
    assert {chunk.principal for chunk in chunks} == {
        "arn:aws:iam::1:user/alice",
        "arn:aws:iam::1:user/bob",
    }


def test_different_source_ips_split_windows():
    events = [
        make_event(0, src_ip="10.0.0.1"),
        make_event(1, src_ip="203.0.113.44"),
    ]
    assert len(build_windows(events, gap_minutes=15)) == 2


def test_different_sources_split_windows():
    events = [make_event(0, source="cloudtrail"), make_event(1, source="auth")]
    assert len(build_windows(events, gap_minutes=15)) == 2


def test_events_are_sorted_before_grouping():
    events = [make_event(20), make_event(0), make_event(5)]
    chunks = build_windows(events, gap_minutes=15)
    assert chunks[0].ts_start == "2026-08-14T02:00:00Z"
    assert chunks[0].n_events == 3


def test_max_events_cap_splits_a_long_window():
    events = [make_event(i * 0.1) for i in range(25)]
    chunks = build_windows(events, gap_minutes=15, max_events=10)
    assert [chunk.n_events for chunk in chunks] == [10, 10, 5]


def test_max_chars_cap_splits_a_long_window():
    events = [make_event(i * 0.1, action="DescribeInstances") for i in range(40)]
    chunks = build_windows(events, gap_minutes=15, max_chars=900, max_events=1000)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 1200, "char cap should keep chunks near the limit"


def test_chunk_ids_are_sequential_and_unique():
    events = [make_event(i * 60) for i in range(5)]
    chunks = build_windows(events, gap_minutes=15)
    ids = [chunk.chunk_id for chunk in chunks]
    assert ids == ["w-00001", "w-00002", "w-00003", "w-00004", "w-00005"]


def test_start_index_offsets_chunk_ids():
    chunks = build_windows([make_event(0)], start_index=42)
    assert chunks[0].chunk_id == "w-00042"


# --------------------------------------------------------------------------- #
# rendering / metadata
# --------------------------------------------------------------------------- #


def test_text_contains_header_and_one_line_per_event():
    events = [
        make_event(0, action="ConsoleLogin", status="failure"),
        make_event(1, action="ConsoleLogin", status="failure"),
    ]
    chunk = build_windows(events)[0]
    lines = chunk.text.splitlines()
    assert lines[0].startswith(f"[WINDOW {chunk.chunk_id}]")
    assert "source=cloudtrail" in lines[0]
    assert "failures=2" in lines[0]
    assert len(lines) == 3
    assert all(line.startswith("- ") for line in lines[1:])
    assert "FAIL" in chunk.text


def test_event_ids_are_traceable_from_the_chunk():
    events = [make_event(0, event_id="ct-aaa"), make_event(1, event_id="ct-bbb")]
    chunk = build_windows(events)[0]
    assert chunk.event_ids == ["ct-aaa", "ct-bbb"]
    for event_id in chunk.event_ids:
        assert event_id in chunk.text, "citation ids must appear in the rendered text"


def test_event_names_and_accounts_are_aggregated():
    events = [
        make_event(0, action="CreateUser"),
        make_event(1, action="CreateAccessKey"),
        make_event(2, action="CreateUser"),
    ]
    chunk = build_windows(events)[0]
    assert chunk.event_names == ["CreateAccessKey", "CreateUser"]


def test_search_text_carries_lexical_signal():
    events = [make_event(0, action="StopLogging", status="failure", src_ip="203.0.113.44")]
    chunk = build_windows(events)[0]
    for token in ("stoplogging", "203.0.113.44", "failure"):
        assert token in chunk.search_text
    assert chunk.search_text == chunk.search_text.lower()


def test_chunk_round_trips_through_dict():
    chunk = build_windows([make_event(0), make_event(1)])[0]
    restored = type(chunk).from_dict(chunk.to_dict())
    assert restored.to_dict() == chunk.to_dict()


# --------------------------------------------------------------------------- #
# risk scoring
# --------------------------------------------------------------------------- #


def test_risk_score_is_zero_for_benign_reads():
    events = [make_event(i, action="GetObject") for i in range(5)]
    assert compute_risk_score(events) == 0


def test_risk_score_rewards_sensitive_actions():
    events = [make_event(0, action="AttachUserPolicy")]
    score = compute_risk_score(events)
    assert score == SENSITIVE_ACTIONS["AttachUserPolicy"]


def test_risk_score_does_not_double_count_repeated_actions():
    one = compute_risk_score([make_event(0, action="StopLogging")])
    many = compute_risk_score([make_event(i, action="StopLogging") for i in range(20)])
    assert one == SENSITIVE_ACTIONS["StopLogging"]
    assert many == one, "the same sensitive action must be counted once, not per event"

    failed = compute_risk_score(
        [make_event(i, action="StopLogging", status="failure") for i in range(20)]
    )
    assert failed > one, "repeated failures should still add failure weight"
    assert failed <= 100


def test_risk_score_is_capped_at_100():
    events = [
        make_event(i, action=action, status="failure")
        for i, action in enumerate(SENSITIVE_ACTIONS)
    ]
    assert compute_risk_score(events) == 100


def test_root_principal_adds_risk():
    events = [make_event(0, principal="arn:aws:iam::123456789012:root")]
    assert is_root_principal(events[0].principal)
    assert compute_risk_score(events) == 25


def test_is_root_principal_negative_case():
    assert not is_root_principal("arn:aws:iam::123456789012:user/alice")


def test_summarize_chunks():
    events = [make_event(0, action="StopLogging"), make_event(60), make_event(61)]
    chunks = build_windows(events, gap_minutes=15)
    summary = summarize_chunks(chunks)
    assert summary["chunks"] == 2
    assert summary["events"] == 3
    assert summary["failures"] == 0
    assert summary["by_source"] == {"cloudtrail": 2}
    assert summary["high_risk"] == 0, "risk 40 is below the high-risk threshold of 50"
    assert summarize_chunks([])["chunks"] == 0


def test_summarize_counts_high_risk_windows():
    root = "arn:aws:iam::123456789012:root"
    events = [
        make_event(0, action="StopLogging", principal=root, status="failure"),
        make_event(1, action="DeleteTrail", principal=root, status="failure"),
    ]
    summary = summarize_chunks(build_windows(events))
    assert summary["chunks"] == 1
    assert summary["high_risk"] == 1
    assert summary["failures"] == 2
