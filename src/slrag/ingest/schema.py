"""Normalized event schema — every log source funnels into :class:`Event`.

The schema is deliberately flat: a security analyst's questions are almost always
filter/aggregate questions ("failed logins from this IP in the last 24h"), so the fields
that get filtered on are first-class columns rather than JSON blobs. The original record is
kept in ``raw`` for auditability and for citation back to the source line.
"""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

Source = Literal["cloudtrail", "auth", "syslog"]
Status = Literal["success", "failure", "info"]

SOURCES: tuple[str, ...] = ("cloudtrail", "auth", "syslog")
STATUSES: tuple[str, ...] = ("success", "failure", "info")

#: Human-readable status markers used when rendering an event into chunk text.
_STATUS_MARK = {"success": "OK", "failure": "FAIL", "info": "INFO"}

_FAILURE_HINTS = (
    "fail",
    "denied",
    "deny",
    "error",
    "invalid",
    "unauthorized",
    "refused",
    "timeout",
    "locked",
    "not authorized",
    "no such",
    "killed",
    "out of memory",
    "oom-kill",
    "segfault",
    "panic",
    "critical",
    "unable to",
)


def normalize_status(value: str | None) -> Status:
    """Map a provider-specific status/error string onto the three-value enum."""
    if not value:
        return "info"
    lowered = value.strip().lower()
    if lowered in {"success", "succeeded", "ok", "accepted"}:
        return "success"
    if lowered in {"failure", "failed", "error"}:
        return "failure"
    if any(hint in lowered for hint in _FAILURE_HINTS):
        return "failure"
    return "info"


def iso_utc(value: datetime) -> str:
    """Format a datetime as ``YYYY-MM-DDTHH:MM:SSZ`` (second precision, always UTC)."""
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp produced by :func:`iso_utc` (tolerant of offsets)."""
    text = value.strip().replace("Z", "+00:00")
    dt = datetime.fromisoformat(text)
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def stable_id(prefix: str, *parts: str) -> str:
    """Deterministic short id — makes re-ingesting the same logs idempotent."""
    digest = hashlib.sha1("|".join(parts).encode("utf-8", "replace")).hexdigest()
    return f"{prefix}-{digest[:12]}"


@dataclass(slots=True)
class Event:
    """One normalized log record."""

    event_id: str
    ts: str  # ISO-8601 UTC, second precision
    source: Source
    action: str
    status: Status = "info"
    principal: str = ""
    src_ip: str = ""
    region: str = ""
    account_id: str = ""
    resource: str = ""
    user_agent: str = ""
    raw: str = ""
    raw_ref: str = ""  # "<file>:<line>"
    extra: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------ time
    @property
    def ts_dt(self) -> datetime:
        return parse_iso(self.ts)

    # ------------------------------------------------------------- rendering
    def render_line(self) -> str:
        """Compact one-line rendering used inside chunk text.

        Kept stable — the eval harness and the citation renderer both depend on it.
        """
        mark = _STATUS_MARK.get(self.status, "INFO")
        parts = [
            self.ts,
            self.action or "-",
            mark,
            self.principal or "-",
            self.src_ip or "-",
            self.region or "-",
            self.event_id,
        ]
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Event:
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in payload.items() if k in known})

    def validate(self) -> None:
        """Cheap structural validation — used by the loaders' self-check."""
        if not self.event_id:
            raise ValueError("event_id is required")
        if self.source not in SOURCES:
            raise ValueError(f"unknown source: {self.source!r}")
        if self.status not in STATUSES:
            raise ValueError(f"unknown status: {self.status!r}")
        parse_iso(self.ts)  # raises ValueError on malformed timestamps


__all__ = [
    "Event",
    "SOURCES",
    "STATUSES",
    "Source",
    "Status",
    "iso_utc",
    "normalize_status",
    "parse_iso",
    "stable_id",
]
