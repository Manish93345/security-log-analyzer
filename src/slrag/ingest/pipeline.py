"""End-to-end ingest orchestration: raw log files -> events -> windows -> stores.

One pass, one source of truth. The same normalized events feed:

* ``data/processed/events.sqlite`` — exact/aggregate queries (:mod:`slrag.ingest.store`)
* ``data/processed/chunks.jsonl``  — the hand-off artefact the Phase 3 vector index reads

Keeping the JSONL on disk means the embedding pass can be re-run (or a different embedding
model tried) without re-parsing 224,000 raw records every time.

Memory note: windowing needs the whole event list in memory (windows are defined by gaps
*between* events), which measures ~450 MB peak for the full 224k-record corpus. That is fine
on a laptop; ``--limit`` exists for quick smoke runs.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import get_settings
from .chunker import build_windows, summarize_chunks
from .loaders import LoadStats, discover_files, load_events
from .schema import Event
from .store import EventStore, write_chunks_jsonl


@dataclass
class IngestResult:
    """Everything the CLI prints after an ingest (and the handoff doc records)."""

    files: int = 0
    records: int = 0
    skipped: int = 0
    events: int = 0
    chunks: int = 0
    high_risk: int = 0
    by_source: dict[str, int] = field(default_factory=dict)
    db_path: str = ""
    chunks_path: str = ""
    assumed_year: int = 0
    duration_s: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "files": self.files,
            "records": self.records,
            "skipped": self.skipped,
            "events": self.events,
            "chunks": self.chunks,
            "high_risk": self.high_risk,
            "by_source": self.by_source,
            "db_path": self.db_path,
            "chunks_path": self.chunks_path,
            "assumed_year": self.assumed_year,
            "duration_s": round(self.duration_s, 2),
        }

    def summary_lines(self) -> list[str]:
        by_source = " ".join(f"{k}={v:,}" for k, v in sorted(self.by_source.items()))
        return [
            f"files parsed      : {self.files:,}",
            f"records read      : {self.records:,}",
            f"events kept       : {self.events:,}   ({by_source})",
            f"records skipped   : {self.skipped:,}",
            f"incident windows  : {self.chunks:,}   (high risk: {self.high_risk:,})",
            f"sqlite            : {self.db_path}",
            f"chunks jsonl      : {self.chunks_path}",
            f"elapsed           : {self.duration_s:.1f}s",
        ]


def resolve_assumed_year(paths: list[Path], fallback: int | None = None) -> int:
    """syslog lines carry no year — read it from the generator's manifest when available.

    Falls back to the current UTC year so a hand-made log file still ingests.
    """
    for path in paths:
        candidates = [path / "manifest.json"] if path.is_dir() else []
        candidates.extend([path.parent / "manifest.json", path.parent.parent / "manifest.json"])
        for candidate in candidates:
            if not candidate.is_file():
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            for key in ("corpus_now", "end_date", "end", "generated_at"):
                value = payload.get(key)
                if isinstance(value, str) and len(value) >= 4 and value[:4].isdigit():
                    return int(value[:4])
    return fallback or datetime.now(timezone.utc).year


def run_ingest(
    raw_paths: list[str | Path],
    *,
    db_path: str | Path | None = None,
    chunks_path: str | Path | None = None,
    assumed_year: int | None = None,
    gap_minutes: int | None = None,
    max_chars: int | None = None,
    max_events: int | None = None,
    rebuild: bool = False,
    limit: int | None = None,
    verbose: bool = True,
) -> IngestResult:
    """Parse, normalize, window and persist. Returns a populated :class:`IngestResult`."""
    started = time.perf_counter()
    settings = get_settings()
    settings.ensure_dirs()

    db_target = Path(db_path) if db_path else settings.sqlite_path
    chunks_target = Path(chunks_path) if chunks_path else settings.processed_dir / "chunks.jsonl"
    paths = [Path(p) for p in raw_paths]

    files = discover_files(paths)
    if not files:
        raise FileNotFoundError(
            "no log files found under: " + ", ".join(str(p) for p in paths)
        )
    if verbose:
        print(f"[ingest] {len(files)} log file(s) discovered")

    year = assumed_year or resolve_assumed_year(paths)
    if verbose:
        print(f"[ingest] assuming year {year} for year-less syslog timestamps")

    stats = LoadStats()
    events: list[Event] = []
    for event in load_events(files, assumed_year=year, stats=stats):
        events.append(event)
        if limit is not None and len(events) >= limit:
            break

    if verbose:
        print(f"[ingest] parsed {stats.records:,} records, kept {len(events):,} events")

    store = EventStore(db_target)
    try:
        if rebuild:
            store.drop_schema()
        else:
            store.create_schema()
        store.insert_events(events)
        store.set_meta("assumed_year", str(year))
        store.set_meta("source_files", str(stats.files))
        store.set_meta("records_read", str(stats.records))
        store.set_meta("records_skipped", str(stats.skipped))

        chunks = build_windows(
            events,
            gap_minutes=gap_minutes if gap_minutes is not None else settings.gap_minutes,
            max_chars=max_chars if max_chars is not None else settings.max_chunk_chars,
            max_events=max_events if max_events is not None else settings.max_events_per_chunk,
        )
        store.insert_chunks(chunks)
    finally:
        store.close()

    write_chunks_jsonl(chunks, chunks_target)

    summary = summarize_chunks(chunks)
    result = IngestResult(
        files=stats.files,
        records=stats.records,
        skipped=stats.skipped,
        events=len(events),
        chunks=len(chunks),
        high_risk=int(summary.get("high_risk", 0)),
        by_source={k: int(v) for k, v in Counter(e.source for e in events).items()},
        db_path=str(db_target),
        chunks_path=str(chunks_target),
        assumed_year=year,
        duration_s=time.perf_counter() - started,
    )

    if verbose:
        print("[ingest] done")
        for line in result.summary_lines():
            print(f"[ingest]   {line}")
    return result


__all__ = ["IngestResult", "resolve_assumed_year", "run_ingest"]
