"""Unit tests for the SQLite store (no network, no models)."""

from __future__ import annotations

import json

import pytest

from slrag.ingest.chunker import Chunk
from slrag.ingest.schema import Event
from slrag.ingest.store import (
    EventStore,
    _percentile,
    iter_chunks_jsonl,
    load_chunks_jsonl,
    write_chunks_jsonl,
)


def make_event(**overrides) -> Event:
    base = dict(
        event_id="ct-1",
        ts="2026-08-14T02:11:03Z",
        source="cloudtrail",
        action="ConsoleLogin",
        status="failure",
        principal="arn:aws:iam::123456789012:user/alice",
        src_ip="203.0.113.44",
        region="us-east-1",
        account_id="123456789012",
        resource="",
        user_agent="Mozilla/5.0",
        raw='{"eventName":"ConsoleLogin"}',
        raw_ref="cloudtrail-2026-08-14.json:1",
        extra={"identity_type": "IAMUser"},
    )
    base.update(overrides)
    return Event(**base)


def make_chunk(chunk_id: str = "w-00001", **overrides) -> Chunk:
    base = dict(
        chunk_id=chunk_id,
        ts_start="2026-08-14T02:11:03Z",
        ts_end="2026-08-14T02:25:00Z",
        source="cloudtrail",
        principal="arn:aws:iam::123456789012:user/alice",
        src_ip="203.0.113.44",
        n_events=2,
        n_failures=1,
        event_names=["ConsoleLogin", "AssumeRole"],
        event_ids=["ct-1", "ct-2"],
        risk_score=60,
        text="[WINDOW w-00001] ...",
        search_text="consolelogin assumerole alice",
        accounts=["123456789012"],
    )
    base.update(overrides)
    return Chunk(**base)


@pytest.fixture
def store(tmp_path) -> EventStore:
    with EventStore(tmp_path / "events.sqlite") as opened:
        opened.create_schema()
        yield opened


def test_schema_creates_all_tables(store):
    rows = store.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {row["name"] for row in rows}
    assert {"events", "chunks", "chunk_events", "meta"} <= names
    assert store.get_meta("schema_version") == "1"


def test_insert_events_roundtrip(store):
    written = store.insert_events([make_event()])
    assert written == 1
    fetched = store.get_event("ct-1")
    assert fetched is not None
    assert fetched.action == "ConsoleLogin"
    assert fetched.extra == {"identity_type": "IAMUser"}
    assert store.counts()["events"] == 1


def test_insert_events_is_idempotent(store):
    events = [make_event(event_id=f"ct-{i}", ts=f"2026-08-14T02:{i:02d}:00Z") for i in range(5)]
    store.insert_events(events)
    store.insert_events(events)  # re-ingest must not duplicate
    assert store.counts()["events"] == 5


def test_insert_chunks_and_membership_order(store):
    store.insert_events(
        [
            make_event(event_id="ct-1"),
            make_event(event_id="ct-2", ts="2026-08-14T02:12:00Z"),
        ]
    )
    store.insert_chunks([make_chunk()])
    counts = store.counts()
    assert counts["chunks"] == 1
    assert counts["chunk_events"] == 2

    members = store.events_for_chunk("w-00001")
    assert [event.event_id for event in members] == ["ct-1", "ct-2"]
    assert store.chunk("w-00001").risk_score == 60
    assert store.chunk("w-99999") is None


def test_aggregate_helpers(store):
    store.insert_events(
        [
            make_event(event_id="a", action="ConsoleLogin", src_ip="1.1.1.1", status="failure"),
            make_event(event_id="b", action="ConsoleLogin", src_ip="1.1.1.1", status="failure"),
            make_event(event_id="c", action="AssumeRole", src_ip="2.2.2.2", status="success"),
            make_event(
                event_id="d",
                action="CreateAccessKey",
                src_ip="2.2.2.2",
                status="success",
                principal="arn:aws:iam::123456789012:user/bob",
            ),
        ]
    )
    actions = {row["action"]: row["n"] for row in store.top_actions()}
    assert actions["ConsoleLogin"] == 2

    ips = {row["src_ip"]: row["failures"] for row in store.top_src_ips()}
    assert ips["1.1.1.1"] == 2
    assert ips["2.2.2.2"] == 0

    principals = store.top_principals()
    assert principals[0]["principal"].endswith("alice")
    assert principals[0]["n"] == 3

    only_failures = store.top_actions(status="failure")
    assert {row["action"] for row in only_failures} == {"ConsoleLogin"}


def test_events_between_filters(store):
    store.insert_events(
        [
            make_event(event_id="a", ts="2026-08-14T01:00:00Z"),
            make_event(event_id="b", ts="2026-08-14T02:00:00Z"),
            make_event(event_id="c", ts="2026-08-14T03:00:00Z", source="auth"),
        ]
    )
    window = store.events_between("2026-08-14T01:30:00Z", "2026-08-14T02:30:00Z")
    assert [event.event_id for event in window] == ["b"]

    auth_only = store.events_between(
        "2026-08-14T00:00:00Z", "2026-08-14T23:59:59Z", source="auth"
    )
    assert [event.event_id for event in auth_only] == ["c"]


def test_top_risk_chunks_ordering(store):
    store.insert_chunks(
        [
            make_chunk("w-00001", risk_score=10),
            make_chunk("w-00002", risk_score=95),
            make_chunk("w-00003", risk_score=55),
        ]
    )
    assert [c.chunk_id for c in store.top_risk_chunks(2)] == ["w-00002", "w-00003"]


def test_stats_shape(store):
    store.insert_events([make_event(event_id="a"), make_event(event_id="b", status="success")])
    store.insert_chunks([make_chunk()])
    stats = store.stats(top_n=3)

    assert stats["events"]["total"] == 2
    assert stats["events"]["by_source"] == {"cloudtrail": 2}
    assert stats["events"]["failures"] == 1
    assert stats["events"]["first_ts"] == "2026-08-14T02:11:03Z"
    assert stats["chunks"]["total"] == 1
    assert stats["chunks"]["high_risk"] == 1
    assert stats["chunks"]["median_events"] == 2.0
    assert stats["db_size_mb"] >= 0


def test_drop_schema_clears_rows(store):
    store.insert_events([make_event()])
    store.insert_chunks([make_chunk()])
    store.drop_schema()
    assert store.counts() == {"events": 0, "chunks": 0, "chunk_events": 0}


def test_read_only_store_rejects_missing_file(tmp_path):
    with EventStore(tmp_path / "nope.sqlite", read_only=True) as store:
        with pytest.raises(FileNotFoundError):
            store.counts()


def test_chunks_jsonl_roundtrip(tmp_path):
    target = tmp_path / "nested" / "chunks.jsonl"
    chunks = [make_chunk("w-00001"), make_chunk("w-00002", risk_score=90)]
    assert write_chunks_jsonl(chunks, target) == 2

    first_line = json.loads(target.read_text(encoding="utf-8").splitlines()[0])
    assert first_line["chunk_id"] == "w-00001"

    streamed = list(iter_chunks_jsonl(target))
    assert [c.chunk_id for c in streamed] == ["w-00001", "w-00002"]
    assert streamed[1].risk_score == 90
    assert load_chunks_jsonl(target)[0].event_ids == ["ct-1", "ct-2"]


def test_percentile_helper():
    assert _percentile([], 0.5) == 0.0
    assert _percentile([7], 0.9) == 7.0
    assert _percentile([1, 2, 3, 4], 0.5) == 2.5
    assert _percentile([1, 2, 3, 4], 1.0) == 4.0
