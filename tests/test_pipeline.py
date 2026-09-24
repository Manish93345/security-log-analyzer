"""End-to-end tests for the ingest pipeline and the CLI (no network, no models)."""

from __future__ import annotations

import json

import pytest

from slrag import cli
from slrag.ingest.pipeline import resolve_assumed_year, run_ingest
from slrag.ingest.store import EventStore, load_chunks_jsonl


@pytest.fixture
def paths(tmp_path):
    return {"db": tmp_path / "events.sqlite", "chunks": tmp_path / "chunks.jsonl"}


def test_run_ingest_end_to_end(small_corpus, paths):
    result = run_ingest(
        [small_corpus["dir"]], db_path=paths["db"], chunks_path=paths["chunks"], verbose=False
    )
    assert result.events > 0
    assert result.chunks > 0
    assert result.files >= 2
    assert result.records >= result.events
    assert paths["chunks"].exists()

    with EventStore(paths["db"], read_only=True) as store:
        counts = store.counts()
    assert counts["events"] == result.events
    assert counts["chunks"] == result.chunks
    assert counts["chunk_events"] == sum(c.n_events for c in load_chunks_jsonl(paths["chunks"]))


def test_run_ingest_is_idempotent(small_corpus, paths):
    first = run_ingest(
        [small_corpus["dir"]], db_path=paths["db"], chunks_path=paths["chunks"], verbose=False
    )
    second = run_ingest(
        [small_corpus["dir"]], db_path=paths["db"], chunks_path=paths["chunks"], verbose=False
    )
    assert first.events == second.events
    with EventStore(paths["db"], read_only=True) as store:
        assert store.counts()["events"] == first.events


def test_run_ingest_limit_truncates(small_corpus, paths):
    result = run_ingest(
        [small_corpus["dir"]],
        db_path=paths["db"],
        chunks_path=paths["chunks"],
        limit=25,
        verbose=False,
    )
    assert result.events == 25
    assert result.chunks >= 1


def test_run_ingest_rebuild_resets(small_corpus, paths):
    run_ingest(
        [small_corpus["dir"]], db_path=paths["db"], chunks_path=paths["chunks"], verbose=False
    )
    rebuilt = run_ingest(
        [small_corpus["dir"]],
        db_path=paths["db"],
        chunks_path=paths["chunks"],
        rebuild=True,
        verbose=False,
    )
    with EventStore(paths["db"], read_only=True) as store:
        assert store.counts()["events"] == rebuilt.events


def test_run_ingest_missing_path_raises(tmp_path, paths):
    with pytest.raises(FileNotFoundError):
        run_ingest(
            [tmp_path / "does-not-exist"],
            db_path=paths["db"],
            chunks_path=paths["chunks"],
            verbose=False,
        )


def test_resolve_assumed_year_from_manifest(small_corpus):
    assert resolve_assumed_year([small_corpus["dir"]]) == 2026


def test_resolve_assumed_year_fallback(tmp_path):
    assert resolve_assumed_year([tmp_path]) == 2026 or resolve_assumed_year([tmp_path]) >= 2025


def test_summary_lines_are_renderable(small_corpus, paths):
    result = run_ingest(
        [small_corpus["dir"]], db_path=paths["db"], chunks_path=paths["chunks"], verbose=False
    )
    text = "\n".join(result.summary_lines())
    assert "incident windows" in text
    assert str(result.chunks) in text


def test_cli_ingest_then_stats_json(small_corpus, paths, capsys):
    exit_code = cli.main(
        [
            "ingest",
            "--raw",
            str(small_corpus["dir"]),
            "--db",
            str(paths["db"]),
            "--chunks",
            str(paths["chunks"]),
            "--json",
        ]
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["events"] > 0

    assert cli.main(["stats", "--db", str(paths["db"]), "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["events"]["total"] == payload["events"]
    assert stats["chunks"]["total"] == payload["chunks"]


def test_cli_stats_human_output(small_corpus, paths, capsys):
    cli.main(
        [
            "ingest",
            "--raw",
            str(small_corpus["dir"]),
            "--db",
            str(paths["db"]),
            "--chunks",
            str(paths["chunks"]),
        ]
    )
    capsys.readouterr()
    assert cli.main(["stats", "--db", str(paths["db"]), "--top", "3"]) == 0
    out = capsys.readouterr().out
    assert "slrag stats" in out
    assert "top actions" in out


def test_cli_missing_db_returns_2(tmp_path, capsys):
    assert cli.main(["stats", "--db", str(tmp_path / "nope.sqlite")]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_missing_raw_returns_2(tmp_path, capsys):
    assert cli.main(["ingest", "--raw", str(tmp_path / "nope")]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_version(capsys):
    assert cli.main(["version"]) == 0
    assert "slrag" in capsys.readouterr().out
