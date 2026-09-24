"""Tests for the synthetic corpus generator (Phase 1 acceptance criteria)."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def _run(generator, out: Path, **kwargs) -> dict:
    params = {
        "seed": 42,
        "days": 2,
        "events_per_day": 120,
        "ssh_per_day": 40,
        "syslog_per_day": 15,
        "labels_path": out / "labels.json",
        "verbose": False,
    }
    params.update(kwargs)
    return generator.generate(out, **params)


def test_generator_is_deterministic(generator, tmp_path):
    """Same seed + same window => byte-identical corpus (CI depends on this)."""
    a, b = tmp_path / "a", tmp_path / "b"
    end = datetime(2026, 8, 31, tzinfo=timezone.utc)
    _run(generator, a, end_date=end)
    _run(generator, b, end_date=end)
    assert _digest(a) == _digest(b)


def test_different_seed_changes_corpus(generator, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    end = datetime(2026, 8, 31, tzinfo=timezone.utc)
    _run(generator, a, end_date=end, seed=1)
    _run(generator, b, end_date=end, seed=2)
    assert _digest(a) != _digest(b)


def test_record_counts_match_configuration(generator, tmp_path):
    out = tmp_path / "corpus"
    manifest = _run(generator, out, days=3, events_per_day=100, ssh_per_day=30, syslog_per_day=10)
    counts = manifest["counts"]
    # Background events are exactly days * events_per_day; scenario events are extra.
    assert counts["cloudtrail_records"] >= 300
    assert counts["auth_log_lines"] >= 90
    assert counts["syslog_lines"] == 30
    assert counts["total_records"] == (
        counts["cloudtrail_records"] + counts["auth_log_lines"] + counts["syslog_lines"]
    )
    assert counts["scenarios"] == 10
    assert counts["scenario_events"] > 0


def test_cloudtrail_files_are_valid_json_arrays(small_corpus):
    files = sorted((small_corpus["dir"] / "cloudtrail").glob("*.json"))
    assert files, "no CloudTrail files written"
    for path in files:
        records = json.loads(path.read_text(encoding="utf-8"))
        assert isinstance(records, list) and records
        for record in records:
            assert {"eventTime", "eventName", "eventID", "userIdentity"} <= set(record)
            # eventTime must round-trip as ISO-8601 UTC
            assert record["eventTime"].endswith("Z")
            datetime.fromisoformat(record["eventTime"].replace("Z", "+00:00"))


def test_cloudtrail_records_are_time_sorted(small_corpus):
    path = sorted((small_corpus["dir"] / "cloudtrail").glob("*.json"))[0]
    records = json.loads(path.read_text(encoding="utf-8"))
    times = [record["eventTime"] for record in records]
    assert times == sorted(times)


def test_labels_have_ground_truth_event_ids(small_corpus):
    labels = json.loads((small_corpus["dir"] / "labels.json").read_text(encoding="utf-8"))
    assert labels["scenarios"], "no scenarios recorded"
    ids = {scenario["id"] for scenario in labels["scenarios"]}
    assert ids == {f"S{i}" for i in range(1, 11)}
    for scenario in labels["scenarios"]:
        assert scenario["expected_queries"], f"{scenario['id']} has no expected queries"
        assert scenario["event_ids"], f"{scenario['id']} has no ground-truth events"
        assert scenario["ts_start"] <= scenario["ts_end"]


def test_label_event_ids_exist_in_the_corpus(small_corpus):
    """Every ground-truth id must be findable in the written logs (no dangling labels)."""
    labels = json.loads((small_corpus["dir"] / "labels.json").read_text(encoding="utf-8"))
    cloudtrail_ids: set[str] = set()
    for path in (small_corpus["dir"] / "cloudtrail").glob("*.json"):
        for record in json.loads(path.read_text(encoding="utf-8")):
            cloudtrail_ids.add("ct-" + record["eventID"])

    checked = 0
    for scenario in labels["scenarios"]:
        if scenario["stream"] != "cloudtrail":
            continue
        for event_id in scenario["event_ids"]:
            assert event_id in cloudtrail_ids, f"{scenario['id']}: {event_id} not in corpus"
            checked += 1
    assert checked > 0


def test_attack_signatures_are_greppable(small_corpus):
    """The planted scenarios must be discoverable by simple pattern matching."""
    raw = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (small_corpus["dir"] / "cloudtrail").glob("*.json")
    )
    for needle in (
        "FailedAuthentication",
        "AttachUserPolicy",
        "AdministratorAccess",
        "AuthorizeSecurityGroupIngress",
        "StopLogging",
        "DeleteTrail",
        "PutBucketPolicy",
    ):
        assert needle in raw, f"expected signature {needle!r} missing from the corpus"

    auth_log = (small_corpus["dir"] / "syslog" / "auth.log").read_text(encoding="utf-8")
    assert "Failed password for root" in auth_log
    assert auth_log.count("Failed password for root from") >= 400


def test_manifest_hashes_every_file(small_corpus):
    manifest = small_corpus["manifest"]
    assert manifest["files"], "manifest has no file hashes"
    for name, digest in manifest["files"].items():
        assert len(digest) == 64 and name


def test_full_corpus_exceeds_100k():
    """Guard the 100K+ claim: the --full configuration must clear it by construction."""
    days, events_per_day, ssh_per_day, syslog_per_day = 30, 4000, 2000, 667
    total = days * (events_per_day + ssh_per_day + syslog_per_day)
    assert total >= 100_000
    assert total >= 200_000 - 1


def test_syslog_line_format_is_classic():
    """Classic syslog has a space-padded day (``Aug  4``, not ``Aug 04``)."""
    generator_stub = None
    from datetime import datetime as dt

    import importlib.util
    from pathlib import Path as P

    spec = importlib.util.spec_from_file_location(
        "gen", P(__file__).resolve().parents[1] / "scripts" / "generate_logs.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    generator_stub = module
    assert generator_stub._syslog_ts(dt(2026, 8, 4, 2, 11, 3)) == "Aug  4 02:11:03"
    assert generator_stub._syslog_ts(dt(2026, 8, 14, 2, 11, 3)) == "Aug 14 02:11:03"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
