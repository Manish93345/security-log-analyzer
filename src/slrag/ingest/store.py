"""SQLite structured store — the exact/aggregate half of the dual retrieval path.

Why SQLite (and not only vectors)?
----------------------------------
A large share of real analyst questions are *exact* questions: "how many failed logins came
from 203.0.113.7 on Tuesday", "which principal created the most access keys", "show me the
highest-risk windows". Similarity search is the wrong tool for those — it returns *plausible
neighbours*, not an exact count, and it will happily answer "about 40" when the true answer
is 7. So the pipeline keeps two stores and routes each question to the right one:

    exact / aggregate questions   ->  SQLite  (this module)
    fuzzy / semantic questions    ->  Chroma  (Phase 3)

Both are written in a single pass by ``slrag ingest``, so the two views can never drift apart.

Everything here is stdlib ``sqlite3``: no extra dependency, no server, and the entire corpus
lives in one portable file (``data/processed/events.sqlite``).

Idempotency
-----------
Every write is ``INSERT OR REPLACE`` keyed on a deterministic id, so re-running ``ingest``
over the same logs is safe and leaves the row counts unchanged.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from .chunker import Chunk
from .schema import Event

SCHEMA_VERSION = 1

# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS events (
    event_id    TEXT PRIMARY KEY,
    ts          TEXT NOT NULL,
    source      TEXT NOT NULL,
    action      TEXT NOT NULL,
    status      TEXT NOT NULL,
    principal   TEXT NOT NULL DEFAULT '',
    src_ip      TEXT NOT NULL DEFAULT '',
    region      TEXT NOT NULL DEFAULT '',
    account_id  TEXT NOT NULL DEFAULT '',
    resource    TEXT NOT NULL DEFAULT '',
    user_agent  TEXT NOT NULL DEFAULT '',
    raw_ref     TEXT NOT NULL DEFAULT '',
    raw         TEXT NOT NULL DEFAULT '',
    extra       TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_events_ts        ON events(ts);
CREATE INDEX IF NOT EXISTS idx_events_source    ON events(source);
CREATE INDEX IF NOT EXISTS idx_events_action    ON events(action);
CREATE INDEX IF NOT EXISTS idx_events_status    ON events(status);
CREATE INDEX IF NOT EXISTS idx_events_principal ON events(principal);
CREATE INDEX IF NOT EXISTS idx_events_src_ip    ON events(src_ip);
CREATE INDEX IF NOT EXISTS idx_events_ts_source ON events(source, ts);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT PRIMARY KEY,
    ts_start    TEXT NOT NULL,
    ts_end      TEXT NOT NULL,
    source      TEXT NOT NULL,
    principal   TEXT NOT NULL DEFAULT '',
    src_ip      TEXT NOT NULL DEFAULT '',
    n_events    INTEGER NOT NULL,
    n_failures  INTEGER NOT NULL,
    risk_score  INTEGER NOT NULL,
    event_names TEXT NOT NULL DEFAULT '[]',
    accounts    TEXT NOT NULL DEFAULT '[]',
    text        TEXT NOT NULL,
    search_text TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_chunks_ts_start ON chunks(ts_start);
CREATE INDEX IF NOT EXISTS idx_chunks_risk     ON chunks(risk_score);
CREATE INDEX IF NOT EXISTS idx_chunks_source   ON chunks(source);

-- membership map: which raw events make up each window (drives citation rendering)
CREATE TABLE IF NOT EXISTS chunk_events (
    chunk_id TEXT NOT NULL REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    event_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    PRIMARY KEY (chunk_id, event_id)
);

CREATE INDEX IF NOT EXISTS idx_chunk_events_event ON chunk_events(event_id);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

EVENT_COLUMNS: tuple[str, ...] = (
    "event_id",
    "ts",
    "source",
    "action",
    "status",
    "principal",
    "src_ip",
    "region",
    "account_id",
    "resource",
    "user_agent",
    "raw_ref",
    "raw",
    "extra",
)

CHUNK_COLUMNS: tuple[str, ...] = (
    "chunk_id",
    "ts_start",
    "ts_end",
    "source",
    "principal",
    "src_ip",
    "n_events",
    "n_failures",
    "risk_score",
    "event_names",
    "accounts",
    "text",
    "search_text",
)


def _percentile(values: Sequence[int], q: float) -> float:
    """Linear-interpolation percentile (q in 0..1) — no numpy dependency."""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    frac = position - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


# --------------------------------------------------------------------------- #
# store
# --------------------------------------------------------------------------- #


class EventStore:
    """Thin, dependency-free wrapper around the SQLite file.

    Usage::

        with EventStore("data/processed/events.sqlite") as store:
            store.create_schema()
            store.insert_events(events)
    """

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------ connection
    @property
    def conn(self) -> sqlite3.Connection:
        """Lazily-opened connection (``sqlite3.Row`` rows, WAL when writable)."""
        if self._conn is None:
            if self.read_only:
                if not self.path.exists():
                    raise FileNotFoundError(
                        f"{self.path} does not exist — run `slrag ingest` first"
                    )
                self._conn = sqlite3.connect(
                    f"file:{self.path.as_posix()}?mode=ro", uri=True, timeout=30.0
                )
            else:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(str(self.path), timeout=30.0)
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA foreign_keys=ON")
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.commit()
            self._conn.close()
            self._conn = None

    def __enter__(self) -> EventStore:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # ---------------------------------------------------------------- schema
    def create_schema(self) -> None:
        """Create every table/index (idempotent)."""
        self.conn.executescript(SCHEMA_SQL)
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )
        self.conn.commit()

    def drop_schema(self) -> None:
        """Delete all rows and recreate — used by ``ingest --rebuild`` and tests."""
        for table in ("chunk_events", "chunks", "events", "meta"):
            self.conn.execute(f"DROP TABLE IF EXISTS {table}")
        self.conn.commit()
        self.create_schema()

    def set_meta(self, key: str, value: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, str(value))
        )
        self.conn.commit()

    def get_meta(self, key: str, default: str = "") -> str:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return str(row["value"]) if row else default

    # ----------------------------------------------------------------- write
    def insert_events(
        self, events: Iterable[Event], *, batch_size: int = 5_000
    ) -> int:
        """Insert/replace events in batches. Returns the number of rows written."""
        placeholders = ", ".join("?" for _ in EVENT_COLUMNS)
        sql = (
            f"INSERT OR REPLACE INTO events ({', '.join(EVENT_COLUMNS)}) "
            f"VALUES ({placeholders})"
        )
        written = 0
        batch: list[tuple[Any, ...]] = []
        for event in events:
            batch.append(self._event_row(event))
            if len(batch) >= batch_size:
                self.conn.executemany(sql, batch)
                written += len(batch)
                batch.clear()
        if batch:
            self.conn.executemany(sql, batch)
            written += len(batch)
        self.conn.commit()
        return written

    def insert_chunks(
        self, chunks: Iterable[Chunk], *, batch_size: int = 1_000
    ) -> int:
        """Insert/replace windows plus their event membership rows."""
        placeholders = ", ".join("?" for _ in CHUNK_COLUMNS)
        chunk_sql = (
            f"INSERT OR REPLACE INTO chunks ({', '.join(CHUNK_COLUMNS)}) "
            f"VALUES ({placeholders})"
        )
        link_sql = (
            "INSERT OR REPLACE INTO chunk_events (chunk_id, event_id, position) "
            "VALUES (?, ?, ?)"
        )
        written = 0
        chunk_batch: list[tuple[Any, ...]] = []
        link_batch: list[tuple[str, str, int]] = []

        for chunk in chunks:
            chunk_batch.append(self._chunk_row(chunk))
            link_batch.extend(
                (chunk.chunk_id, event_id, position)
                for position, event_id in enumerate(chunk.event_ids)
            )
            if len(chunk_batch) >= batch_size:
                self.conn.executemany(chunk_sql, chunk_batch)
                self.conn.executemany(link_sql, link_batch)
                written += len(chunk_batch)
                chunk_batch.clear()
                link_batch.clear()

        if chunk_batch:
            self.conn.executemany(chunk_sql, chunk_batch)
            self.conn.executemany(link_sql, link_batch)
            written += len(chunk_batch)
        self.conn.commit()
        return written

    @staticmethod
    def _event_row(event: Event) -> tuple[Any, ...]:
        return (
            event.event_id,
            event.ts,
            event.source,
            event.action,
            event.status,
            event.principal,
            event.src_ip,
            event.region,
            event.account_id,
            event.resource,
            event.user_agent,
            event.raw_ref,
            event.raw,
            json.dumps(event.extra, separators=(",", ":"), default=str),
        )

    @staticmethod
    def _chunk_row(chunk: Chunk) -> tuple[Any, ...]:
        return (
            chunk.chunk_id,
            chunk.ts_start,
            chunk.ts_end,
            chunk.source,
            chunk.principal,
            chunk.src_ip,
            chunk.n_events,
            chunk.n_failures,
            chunk.risk_score,
            json.dumps(chunk.event_names, separators=(",", ":")),
            json.dumps(chunk.accounts, separators=(",", ":")),
            chunk.text,
            chunk.search_text,
        )

    # ------------------------------------------------------------------ read
    def counts(self) -> dict[str, int]:
        """Row counts — the cheap health check after an ingest."""
        return {
            "events": int(self.conn.execute("SELECT COUNT(*) c FROM events").fetchone()["c"]),
            "chunks": int(self.conn.execute("SELECT COUNT(*) c FROM chunks").fetchone()["c"]),
            "chunk_events": int(
                self.conn.execute("SELECT COUNT(*) c FROM chunk_events").fetchone()["c"]
            ),
        }

    def time_range(self) -> tuple[str, str]:
        row = self.conn.execute("SELECT MIN(ts) a, MAX(ts) b FROM events").fetchone()
        return (row["a"] or "", row["b"] or "")

    def chunk(self, chunk_id: str) -> Chunk | None:
        row = self.conn.execute(
            "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
        ).fetchone()
        return self._row_to_chunk(row) if row else None

    def events_for_chunk(self, chunk_id: str) -> list[Event]:
        """The raw records behind a window, in original order (for citations)."""
        rows = self.conn.execute(
            "SELECT e.* FROM chunk_events ce JOIN events e ON e.event_id = ce.event_id "
            "WHERE ce.chunk_id = ? ORDER BY ce.position",
            (chunk_id,),
        ).fetchall()
        return [self._row_to_event(row) for row in rows]

    def get_event(self, event_id: str) -> Event | None:
        row = self.conn.execute(
            "SELECT * FROM events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return self._row_to_event(row) if row else None

    def top_actions(self, limit: int = 10, *, status: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT action, COUNT(*) n FROM events"
        params: list[Any] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " GROUP BY action ORDER BY n DESC, action LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self.conn.execute(sql, params)]

    def top_principals(self, limit: int = 10, *, status: str | None = None) -> list[dict[str, Any]]:
        sql = (
            "SELECT principal, COUNT(*) n, SUM(status = 'failure') failures FROM events "
            "WHERE principal != ''"
        )
        params: list[Any] = []
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " GROUP BY principal ORDER BY n DESC, principal LIMIT ?"
        params.append(limit)
        return [dict(row) for row in self.conn.execute(sql, params)]

    def top_src_ips(self, limit: int = 10) -> list[dict[str, Any]]:
        return [
            dict(row)
            for row in self.conn.execute(
                "SELECT src_ip, COUNT(*) n, SUM(status = 'failure') failures FROM events "
                "WHERE src_ip != '' GROUP BY src_ip ORDER BY n DESC, src_ip LIMIT ?",
                (limit,),
            )
        ]

    def events_between(
        self,
        start: str,
        end: str,
        *,
        source: str | None = None,
        status: str | None = None,
        principal: str | None = None,
        limit: int = 500,
    ) -> list[Event]:
        """Exact time-window query — the shape an analyst uses most often."""
        sql = "SELECT * FROM events WHERE ts >= ? AND ts <= ?"
        params: list[Any] = [start, end]
        for column, value in (
            ("source", source),
            ("status", status),
            ("principal", principal),
        ):
            if value:
                sql += f" AND {column} = ?"
                params.append(value)
        sql += " ORDER BY ts, event_id LIMIT ?"
        params.append(limit)
        return [self._row_to_event(row) for row in self.conn.execute(sql, params)]

    def top_risk_chunks(self, limit: int = 10) -> list[Chunk]:
        rows = self.conn.execute(
            "SELECT * FROM chunks ORDER BY risk_score DESC, n_events DESC, chunk_id LIMIT ?",
            (limit,),
        ).fetchall()
        return [self._row_to_chunk(row) for row in rows]

    def iter_chunks(self, *, load_members: bool = False) -> Iterator[Chunk]:
        """Stream every window.

        Membership is skipped by default: loading ``event_ids`` for each of 38k windows is
        an N+1 query pattern, and the Phase 3 index only needs ``text``/``search_text``.
        Pass ``load_members=True`` when citations are required.
        """
        for row in self.conn.execute("SELECT * FROM chunks ORDER BY chunk_id"):
            yield self._row_to_chunk(row, load_members=load_members)

    # ----------------------------------------------------------------- stats
    def stats(self, *, top_n: int = 10) -> dict[str, Any]:
        """Everything the ``slrag stats`` command and the UI dashboard need."""
        counts = self.counts()
        first_ts, last_ts = self.time_range()

        by_source = {
            row["source"]: row["n"]
            for row in self.conn.execute(
                "SELECT source, COUNT(*) n FROM events GROUP BY source ORDER BY n DESC"
            )
        }
        by_status = {
            row["status"]: row["n"]
            for row in self.conn.execute(
                "SELECT status, COUNT(*) n FROM events GROUP BY status ORDER BY n DESC"
            )
        }
        chunk_source = {
            row["source"]: row["n"]
            for row in self.conn.execute(
                "SELECT source, COUNT(*) n FROM chunks GROUP BY source ORDER BY n DESC"
            )
        }
        event_sizes = [
            int(row["n_events"])
            for row in self.conn.execute("SELECT n_events FROM chunks")
        ]
        risk_row = self.conn.execute(
            "SELECT MAX(risk_score) m FROM chunks"
        ).fetchone()

        size_bytes = self.path.stat().st_size if self.path.exists() else 0
        return {
            "db_path": str(self.path),
            "db_size_mb": round(size_bytes / (1024 * 1024), 2),
            "schema_version": self.get_meta("schema_version", str(SCHEMA_VERSION)),
            "events": {
                "total": counts["events"],
                "by_source": by_source,
                "by_status": by_status,
                "failures": by_status.get("failure", 0),
                "first_ts": first_ts,
                "last_ts": last_ts,
                "top_actions": self.top_actions(top_n),
                "top_principals": self.top_principals(top_n),
                "top_src_ips": self.top_src_ips(top_n),
            },
            "chunks": {
                "total": counts["chunks"],
                "chunk_events": counts["chunk_events"],
                "by_source": chunk_source,
                "high_risk": int(
                    self.conn.execute(
                        "SELECT COUNT(*) c FROM chunks WHERE risk_score >= 50"
                    ).fetchone()["c"]
                ),
                "max_risk": int(risk_row["m"] or 0),
                "median_events": round(_percentile(event_sizes, 0.5), 1),
                "p90_events": round(_percentile(event_sizes, 0.9), 1),
                "max_events": max(event_sizes) if event_sizes else 0,
            },
        }

    # ------------------------------------------------------------- row mappers
    @staticmethod
    def _row_to_event(row: sqlite3.Row) -> Event:
        payload = {key: row[key] for key in row.keys() if key in Event.__dataclass_fields__}
        try:
            payload["extra"] = json.loads(payload.get("extra") or "{}")
        except json.JSONDecodeError:
            payload["extra"] = {}
        return Event.from_dict(payload)

    def _row_to_chunk(self, row: sqlite3.Row, *, load_members: bool = True) -> Chunk:
        """Rebuild a :class:`Chunk` from a row.

        ``event_ids`` is not a column on purpose — membership lives in ``chunk_events``
        (normalized, so a window and its events cannot drift apart) — so it is rehydrated
        here. ``load_members=False`` skips that lookup for bulk streaming.
        """
        payload = {key: row[key] for key in row.keys() if key in Chunk.__dataclass_fields__}
        for key in ("event_names", "accounts"):
            try:
                payload[key] = json.loads(payload.get(key) or "[]")
            except json.JSONDecodeError:
                payload[key] = []
        if load_members:
            payload["event_ids"] = [
                member["event_id"]
                for member in self.conn.execute(
                    "SELECT event_id FROM chunk_events WHERE chunk_id = ? ORDER BY position",
                    (payload["chunk_id"],),
                )
            ]
        else:
            payload["event_ids"] = []
        return Chunk.from_dict(payload)


# --------------------------------------------------------------------------- #
# chunks.jsonl — the hand-off artefact to the Phase 3 vector index
# --------------------------------------------------------------------------- #


def write_chunks_jsonl(chunks: Iterable[Chunk], path: str | Path) -> int:
    """Stream windows to newline-delimited JSON. Returns the number written."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with target.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False, default=str))
            handle.write("\n")
            written += 1
    return written


def iter_chunks_jsonl(path: str | Path) -> Iterator[Chunk]:
    """Read windows back one at a time (constant memory, even for 40k windows)."""
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield Chunk.from_dict(json.loads(line))


def load_chunks_jsonl(path: str | Path) -> list[Chunk]:
    return list(iter_chunks_jsonl(path))


__all__ = [
    "CHUNK_COLUMNS",
    "EVENT_COLUMNS",
    "SCHEMA_SQL",
    "SCHEMA_VERSION",
    "EventStore",
    "iter_chunks_jsonl",
    "load_chunks_jsonl",
    "write_chunks_jsonl",
]
