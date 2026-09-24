"""Tests for the CloudTrail and syslog loaders (Phase 2 acceptance criteria)."""

from __future__ import annotations

import json

from slrag.ingest.loaders import (
    LoadStats,
    cloudtrail_record_to_event,
    detect_source,
    discover_files,
    iter_cloudtrail_file,
    iter_syslog_file,
    load_events,
    syslog_line_to_event,
)


# --------------------------------------------------------------------------- #
# CloudTrail
# --------------------------------------------------------------------------- #


def test_cloudtrail_record_maps_every_field(cloudtrail_record):
    event = cloudtrail_record_to_event(cloudtrail_record)
    assert event is not None
    assert event.event_id == "ct-22222222-2222-2222-2222-222222222222"
    assert event.ts == "2026-08-14T02:11:03Z"
    assert event.source == "cloudtrail"
    assert event.action == "ConsoleLogin"
    assert event.status == "failure"
    assert event.principal == "arn:aws:iam::123456789012:user/alice"
    assert event.src_ip == "203.0.113.44"
    assert event.region == "us-east-1"
    assert event.account_id == "123456789012"
    assert event.extra["error_code"] == "FailedAuthentication"
    event.validate()


def test_cloudtrail_success_has_success_status(cloudtrail_record):
    record = dict(cloudtrail_record)
    record.pop("errorCode")
    event = cloudtrail_record_to_event(record)
    assert event is not None and event.status == "success"


def test_cloudtrail_missing_required_fields_returns_none(cloudtrail_record):
    for key in ("eventTime", "eventName"):
        broken = dict(cloudtrail_record)
        broken.pop(key)
        assert cloudtrail_record_to_event(broken) is None
    assert cloudtrail_record_to_event("not a dict") is None  # type: ignore[arg-type]


def test_cloudtrail_generates_id_when_event_id_absent(cloudtrail_record):
    record = dict(cloudtrail_record)
    record.pop("eventID")
    event = cloudtrail_record_to_event(record)
    assert event is not None
    assert event.event_id.startswith("ct-")
    # Deterministic: same input -> same synthetic id
    assert cloudtrail_record_to_event(record).event_id == event.event_id  # type: ignore[union-attr]


def test_cloudtrail_resource_extraction(cloudtrail_record):
    record = dict(cloudtrail_record)
    record["eventName"] = "PutBucketPolicy"
    record["requestParameters"] = {"bucketName": "customer-exports"}
    event = cloudtrail_record_to_event(record)
    assert event is not None and event.resource == "bucketName=customer-exports"


def test_iter_cloudtrail_file_reads_json_array(tmp_path, cloudtrail_record):
    path = tmp_path / "cloudtrail-2026-08-14.json"
    path.write_text(json.dumps([cloudtrail_record, cloudtrail_record]), encoding="utf-8")
    stats = LoadStats()
    events = list(iter_cloudtrail_file(path, stats))
    assert len(events) == 2
    assert stats.files == 1 and stats.records == 2 and stats.skipped == 0


def test_iter_cloudtrail_file_reads_json_lines(tmp_path, cloudtrail_record):
    path = tmp_path / "cloudtrail.jsonl"
    path.write_text(
        "\n".join(json.dumps(cloudtrail_record) for _ in range(3)), encoding="utf-8"
    )
    assert len(list(iter_cloudtrail_file(path))) == 3


def test_iter_cloudtrail_file_survives_malformed_lines(tmp_path, cloudtrail_record):
    path = tmp_path / "cloudtrail.jsonl"
    path.write_text(
        json.dumps(cloudtrail_record)
        + "\n{ not json at all\n"
        + json.dumps({"eventName": "NoTimestamp"})
        + "\n"
        + json.dumps(cloudtrail_record)
        + "\n",
        encoding="utf-8",
    )
    stats = LoadStats()
    events = list(iter_cloudtrail_file(path, stats))
    assert len(events) == 2, "valid records must survive malformed neighbours"
    assert stats.skipped == 2


def test_iter_cloudtrail_file_handles_broken_array(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("[{ truncated", encoding="utf-8")
    stats = LoadStats()
    assert list(iter_cloudtrail_file(path, stats)) == []
    assert stats.skipped == 1


def test_empty_file_is_not_an_error(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text("", encoding="utf-8")
    stats = LoadStats()
    assert list(iter_cloudtrail_file(path, stats)) == []
    assert stats.records == 0


# --------------------------------------------------------------------------- #
# syslog
# --------------------------------------------------------------------------- #


def test_syslog_parses_every_branch(auth_lines):
    events = [
        syslog_line_to_event(line, assumed_year=2026, raw_ref=f"auth.log:{i}")
        for i, line in enumerate(auth_lines, start=1)
    ]
    assert events[-1] is None, "unprefixed line must be skipped"

    parsed = events[:-1]
    actions = [event.action for event in parsed if event]
    assert actions == [
        "sshd.failed_password",
        "sshd.accepted_password",
        "sshd.session_opened",
        "sudo.command",
        "cron.job",
        "kernel.message",
        "systemd.unit",
    ]

    failed = parsed[0]
    assert failed is not None
    assert failed.status == "failure"
    assert failed.principal == "admin"
    assert failed.src_ip == "203.0.113.44"
    assert failed.ts == "2026-08-14T02:11:03Z"
    assert failed.source == "auth"
    failed.validate()

    accepted = parsed[1]
    assert accepted is not None and accepted.status == "success"
    assert accepted.principal == "alice" and accepted.src_ip == "10.0.1.23"

    sudo = parsed[3]
    assert sudo is not None and "systemctl restart nginx" in sudo.resource
    assert sudo.principal == "alice"

    kernel = parsed[5]
    assert kernel is not None and kernel.source == "syslog"
    assert kernel.status == "failure", "OOM kill should read as a failure"


def test_syslog_ids_are_stable_and_unique(auth_lines):
    first = syslog_line_to_event(auth_lines[0], assumed_year=2026, raw_ref="auth.log:1")
    second = syslog_line_to_event(auth_lines[0], assumed_year=2026, raw_ref="auth.log:1")
    other = syslog_line_to_event(auth_lines[1], assumed_year=2026, raw_ref="auth.log:2")
    assert first and second and other
    assert first.event_id == second.event_id
    assert first.event_id != other.event_id
    assert first.event_id.startswith("sl-")


def test_syslog_rejects_garbage():
    assert syslog_line_to_event("", assumed_year=2026) is None
    assert syslog_line_to_event("random text", assumed_year=2026) is None


def test_iter_syslog_file_counts_skips(tmp_path, auth_lines):
    path = tmp_path / "auth.log"
    path.write_text("\n".join(auth_lines) + "\n", encoding="utf-8")
    stats = LoadStats()
    events = list(iter_syslog_file(path, assumed_year=2026, stats=stats))
    assert len(events) == len(auth_lines) - 1
    assert stats.skipped == 1 and stats.records == len(auth_lines)


def test_syslog_handles_space_padded_day(tmp_path):
    path = tmp_path / "auth.log"
    path.write_text(
        "Aug  4 02:11:03 ip-10-0-1-23 sshd[1]: Accepted password for bob from 10.0.1.9 "
        "port 2222 ssh2\n",
        encoding="utf-8",
    )
    events = list(iter_syslog_file(path, assumed_year=2026))
    assert events and events[0].ts == "2026-08-04T02:11:03Z"


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #


def test_detect_source():
    assert detect_source(__import__("pathlib").Path("cloudtrail-2026-08-14.json")) == "cloudtrail"
    assert detect_source(__import__("pathlib").Path("auth.log")) == "syslog"
    assert detect_source(__import__("pathlib").Path("syslog")) == "syslog"
    assert detect_source(__import__("pathlib").Path("anything.txt")) == "syslog"


def test_discover_files_walks_directories(tmp_path, cloudtrail_record, auth_lines):
    (tmp_path / "cloudtrail").mkdir()
    (tmp_path / "syslog").mkdir()
    (tmp_path / "cloudtrail" / "cloudtrail-2026-08-14.json").write_text(
        json.dumps([cloudtrail_record]), encoding="utf-8"
    )
    (tmp_path / "syslog" / "auth.log").write_text(auth_lines[0] + "\n", encoding="utf-8")
    (tmp_path / "syslog" / "notes.md").write_text("ignored", encoding="utf-8")

    found = discover_files([tmp_path])
    names = sorted(path.name for path in found)
    assert names == ["auth.log", "cloudtrail-2026-08-14.json"]


def test_discover_files_keeps_extensionless_syslog(tmp_path):
    """``/var/log/syslog`` has no extension and must not be skipped."""
    (tmp_path / "syslog").write_text("Aug 14 02:11:03 host kernel: [1] boot\n", encoding="utf-8")
    (tmp_path / "auth.log").write_text("", encoding="utf-8")
    names = sorted(path.name for path in discover_files([tmp_path]))
    assert names == ["auth.log", "syslog"]


def test_discover_files_ignores_pipeline_artefacts(tmp_path, cloudtrail_record):
    """manifest.json / labels.json / .gitkeep must never be parsed as logs."""
    (tmp_path / "cloudtrail").mkdir()
    (tmp_path / "cloudtrail" / "cloudtrail-2026-08-14.json").write_text(
        json.dumps([cloudtrail_record]), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"counts": {}}), encoding="utf-8")
    (tmp_path / "labels.json").write_text(json.dumps({"scenarios": []}), encoding="utf-8")
    (tmp_path / ".gitkeep").write_text("", encoding="utf-8")

    names = sorted(path.name for path in discover_files([tmp_path]))
    assert names == ["cloudtrail-2026-08-14.json"]


def test_load_events_skips_nothing_on_the_real_corpus_layout(tmp_path, cloudtrail_record):
    """The generator writes manifest.json next to the logs — it must not inflate skips."""
    (tmp_path / "cloudtrail").mkdir()
    (tmp_path / "cloudtrail" / "cloudtrail-2026-08-14.json").write_text(
        json.dumps([cloudtrail_record]), encoding="utf-8"
    )
    (tmp_path / "manifest.json").write_text(json.dumps({"counts": {"total": 1}}), encoding="utf-8")
    stats = LoadStats()
    events = list(load_events([tmp_path], assumed_year=2026, stats=stats))
    assert len(events) == 1
    assert stats.skipped == 0


def test_load_events_mixes_both_families(tmp_path, cloudtrail_record, auth_lines):
    (tmp_path / "cloudtrail").mkdir()
    (tmp_path / "syslog").mkdir()
    (tmp_path / "cloudtrail" / "cloudtrail-2026-08-14.json").write_text(
        json.dumps([cloudtrail_record] * 2), encoding="utf-8"
    )
    (tmp_path / "syslog" / "auth.log").write_text("\n".join(auth_lines) + "\n", encoding="utf-8")

    stats = LoadStats()
    events = list(load_events([tmp_path], assumed_year=2026, stats=stats))
    sources = {event.source for event in events}
    assert sources == {"cloudtrail", "auth"}
    assert stats.files == 2
    assert stats.records == 2 + len(auth_lines)


def test_load_events_on_generated_corpus(small_corpus):
    """Integration-lite: the loader must consume the generated corpus end to end."""
    stats = LoadStats()
    events = list(load_events([small_corpus["dir"]], assumed_year=2026, stats=stats))
    assert len(events) == small_corpus["manifest"]["counts"]["total_records"]
    assert stats.skipped == 0, "the generator must emit only parseable records"
    for event in events[:200]:
        event.validate()


def test_syslog_ground_truth_ids_resolve_through_the_loader(small_corpus):
    """The S2 (ssh brute force) labels must match the ids the loader actually produces."""
    import json
    from datetime import datetime, timezone

    labels = json.loads((small_corpus["dir"] / "labels.json").read_text(encoding="utf-8"))
    s2 = next(scenario for scenario in labels["scenarios"] if scenario["id"] == "S2")
    assert s2["event_ids"], "S2 has no ground-truth ids"

    loaded = {
        event.event_id
        for event in load_events(
            [small_corpus["dir"]], assumed_year=datetime.now(timezone.utc).year
        )
        if event.source == "auth"
    }
    missing = [event_id for event_id in s2["event_ids"] if event_id not in loaded]
    assert not missing, f"S2 ground-truth ids are not reproducible through the loader: {missing[:3]}"


def test_syslog_source_comes_from_the_file_family(tmp_path, auth_lines):
    """A kernel line inside auth.log belongs to the auth stream, not to syslog."""
    path = tmp_path / "auth.log"
    path.write_text("\n".join(auth_lines) + "\n", encoding="utf-8")
    events = list(iter_syslog_file(path, assumed_year=2026))
    assert {event.source for event in events} == {"auth"}

    other = tmp_path / "syslog"
    other.write_text(auth_lines[5] + "\n", encoding="utf-8")
    assert {event.source for event in iter_syslog_file(other, assumed_year=2026)} == {"syslog"}
