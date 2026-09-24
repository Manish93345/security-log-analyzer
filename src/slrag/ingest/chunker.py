"""Incident-window chunking.

Why not a generic text splitter? Because a 512-token slice of a log file can cut an attack
chain in half, and similarity search over half a chain retrieves nothing useful. Instead we
group events that *belong together* — same source, same principal, same source IP, no gap
longer than ``gap_minutes`` — into a **window**, then render that window as a compact,
self-describing text block. This is the single biggest retrieval-quality decision in the
project.

The chunk keeps its member ``event_ids`` so every answer can cite the exact raw records.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from .schema import Event, iso_utc, parse_iso

#: Actions that matter for a security review, with an additive risk weight.
SENSITIVE_ACTIONS: dict[str, int] = {
    "StopLogging": 40,
    "DeleteTrail": 40,
    "UpdateTrail": 30,
    "PutEventSelectors": 30,
    "CreateUser": 20,
    "DeleteUser": 25,
    "CreateLoginProfile": 30,
    "UpdateLoginProfile": 25,
    "AttachUserPolicy": 35,
    "AttachRolePolicy": 35,
    "PutUserPolicy": 30,
    "PutRolePolicy": 30,
    "AddUserToGroup": 25,
    "CreateAccessKey": 30,
    "CreatePolicyVersion": 30,
    "AssumeRole": 10,
    "PutBucketPolicy": 30,
    "PutBucketAcl": 25,
    "DeleteBucketPolicy": 20,
    "PutBucketPublicAccessBlock": 15,
    "AuthorizeSecurityGroupIngress": 30,
    "AuthorizeSecurityGroupEgress": 25,
    "ModifySecurityGroupRules": 25,
    "GetSecretValue": 25,
    "DeleteDBInstance": 30,
    "CreateFunction": 20,
    "UpdateFunctionCode": 25,
}

#: A principal rendered like this is the AWS account root user.
ROOT_MARKERS = (":root", "/root")


def is_root_principal(principal: str) -> bool:
    return any(marker in principal for marker in ROOT_MARKERS)


@dataclass(slots=True)
class Chunk:
    """A coherent group of related events — the unit of retrieval."""

    chunk_id: str
    ts_start: str
    ts_end: str
    source: str
    principal: str
    src_ip: str
    n_events: int
    n_failures: int
    event_names: list[str]
    event_ids: list[str]
    risk_score: int
    text: str
    search_text: str
    accounts: list[str] = field(default_factory=list)

    @property
    def ts_start_dt(self):
        return parse_iso(self.ts_start)

    @property
    def ts_end_dt(self):
        return parse_iso(self.ts_end)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "ts_start": self.ts_start,
            "ts_end": self.ts_end,
            "source": self.source,
            "principal": self.principal,
            "src_ip": self.src_ip,
            "n_events": self.n_events,
            "n_failures": self.n_failures,
            "event_names": self.event_names,
            "event_ids": self.event_ids,
            "risk_score": self.risk_score,
            "text": self.text,
            "search_text": self.search_text,
            "accounts": self.accounts,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Chunk:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})

    # ---------------------------------------------------------------- helpers
    @property
    def header(self) -> str:
        """First line of the rendered text (also used as the citation label)."""
        return (
            f"[WINDOW {self.chunk_id}] {self.ts_start} -> {self.ts_end} | "
            f"source={self.source} | principal={self.principal or '-'} | "
            f"src_ip={self.src_ip or '-'} | events={self.n_events} | "
            f"failures={self.n_failures} | risk={self.risk_score}"
        )


def compute_risk_score(events: list[Event]) -> int:
    """Heuristic 0-100 risk score for a window (drives the UI triage view)."""
    score = 0
    seen_sensitive: set[str] = set()
    for event in events:
        if event.action in SENSITIVE_ACTIONS and event.action not in seen_sensitive:
            seen_sensitive.add(event.action)
            score += SENSITIVE_ACTIONS[event.action]
    failures = sum(1 for e in events if e.status == "failure")
    score += min(30, failures * 2)
    if any(is_root_principal(e.principal) for e in events):
        score += 25
    return min(100, score)


def _build_search_text(events: list[Event], extra_tokens: str = "") -> str:
    """Lexical bag-of-words used by BM25 — ids and names kept verbatim on purpose."""
    tokens: list[str] = [extra_tokens]
    for event in events:
        tokens.append(event.action)
        tokens.append(event.principal)
        tokens.append(event.src_ip)
        tokens.append(event.resource)
        tokens.append(event.region)
        tokens.append(event.event_id)
        if event.status == "failure":
            tokens.append("failure failed error denied")
    return " ".join(token for token in tokens if token).lower()


def _render(events: list[Event]) -> str:
    """Render a list of events as a window block (header + one line per event)."""
    header = (
        f"[WINDOW {events[0].event_id}] "  # replaced by caller with the real chunk id
    )
    del header
    return "\n".join(f"- {event.render_line()}" for event in events)


def _split_oversized(events: list[Event], max_chars: int, max_events: int) -> list[list[Event]]:
    """Greedily split a group into sub-groups respecting the char/event caps."""
    groups: list[list[Event]] = []
    current: list[Event] = []
    current_chars = 0

    for event in events:
        line_len = len(event.render_line()) + 3  # "- " + newline
        would_overflow = current and (
            current_chars + line_len > max_chars or len(current) >= max_events
        )
        if would_overflow:
            groups.append(current)
            current, current_chars = [], 0
        current.append(event)
        current_chars += line_len

    if current:
        groups.append(current)
    return groups


def _group_key(event: Event) -> tuple[str, str, str]:
    return (event.source, event.principal, event.src_ip)


def build_windows(
    events: list[Event],
    *,
    gap_minutes: int = 15,
    max_chars: int = 4800,
    max_events: int = 120,
    start_index: int = 1,
) -> list[Chunk]:
    """Group events into incident windows and render them.

    Events are grouped by ``(source, principal, src_ip)`` and merged while the gap between
    consecutive events stays within ``gap_minutes``. Groups are then split to respect the
    character and event-count caps.
    """
    if not events:
        return []

    # Group by stream identity FIRST, then merge consecutive events inside each stream.
    # Merging over a single globally-sorted timeline fragments a session whenever an
    # unrelated principal's event lands between two of its own events — with interleaved
    # traffic that collapses every window to a single event and destroys retrieval
    # quality. Bucketing first makes a window boundary depend only on that stream's own
    # timeline.
    buckets: dict[tuple[str, str, str], list[Event]] = {}
    for event in events:
        buckets.setdefault(_group_key(event), []).append(event)

    groups: list[list[Event]] = []
    for key in sorted(buckets):
        stream = sorted(buckets[key], key=lambda e: (e.ts, e.event_id))
        current: list[Event] = []
        last_ts = None
        for event in stream:
            ts = event.ts_dt
            if current and (ts - last_ts).total_seconds() <= gap_minutes * 60:
                current.append(event)
            else:
                if current:
                    groups.append(current)
                current = [event]
            last_ts = ts
        if current:
            groups.append(current)

    # Chronological numbering keeps chunk ids stable and readable.
    groups.sort(key=lambda group: (group[0].ts, _group_key(group[0])))

    chunks: list[Chunk] = []
    index = start_index
    for group in groups:
        for piece in _split_oversized(group, max_chars, max_events):
            chunk_id = f"w-{index:05d}"
            index += 1
            failures = sum(1 for e in piece if e.status == "failure")
            names = sorted({e.action for e in piece})
            accounts = sorted({e.account_id for e in piece if e.account_id})
            chunk = Chunk(
                chunk_id=chunk_id,
                ts_start=iso_utc(piece[0].ts_dt),
                ts_end=iso_utc(piece[-1].ts_dt),
                source=piece[0].source,
                principal=piece[0].principal,
                src_ip=piece[0].src_ip,
                n_events=len(piece),
                n_failures=failures,
                event_names=names,
                event_ids=[e.event_id for e in piece],
                risk_score=compute_risk_score(piece),
                text="",
                search_text="",
            )
            body = "\n".join(f"- {e.render_line()}" for e in piece)
            chunk.text = f"{chunk.header}\n{body}"
            chunk.search_text = _build_search_text(
                piece,
                extra_tokens=f"{chunk.source} {chunk.principal} {chunk.src_ip} "
                f"{' '.join(names)} {' '.join(accounts)}",
            )
            chunks.append(chunk)

    return chunks


def summarize_chunks(chunks: list[Chunk]) -> dict[str, Any]:
    """Aggregate stats used by the ingest CLI and the UI dashboard."""
    if not chunks:
        return {"chunks": 0, "events": 0, "failures": 0, "by_source": {}, "high_risk": 0}
    return {
        "chunks": len(chunks),
        "events": sum(c.n_events for c in chunks),
        "failures": sum(c.n_failures for c in chunks),
        "by_source": dict(Counter(c.source for c in chunks)),
        "high_risk": sum(1 for c in chunks if c.risk_score >= 50),
    }


__all__ = [
    "Chunk",
    "ROOT_MARKERS",
    "SENSITIVE_ACTIONS",
    "build_windows",
    "compute_risk_score",
    "is_root_principal",
    "summarize_chunks",
]
