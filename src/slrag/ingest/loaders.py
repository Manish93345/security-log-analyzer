"""Log loaders: raw CloudTrail / syslog text -> normalized :class:`Event` objects.

Design rules
------------
* **Tolerant, never fatal.** A single malformed line must not abort a 200,000-record
  ingest. Every loader reports how many records it skipped.
* **Deterministic ids.** ``eventID`` is used when the source provides one (CloudTrail);
  otherwise the id is a hash of ``raw_ref + raw`` so re-ingesting is idempotent.
* **No LangChain.** Loaders are plain Python — fast to test and easy to reason about.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .schema import Event, normalize_status, stable_id

# --------------------------------------------------------------------------- #
# syslog patterns
# --------------------------------------------------------------------------- #

#: ``Aug 14 02:11:03 ip-10-0-1-23 sshd[2451]: Failed password for ...``
SYSLOG_PREFIX = re.compile(
    r"^(?P<mon>[A-Z][a-z]{2})\s+(?P<day>\d{1,2})\s+(?P<time>\d{2}:\d{2}:\d{2})\s+"
    r"(?P<host>\S+)\s+(?P<proc>[A-Za-z0-9_.\-]+)(?:\[(?P<pid>\d+)\])?:\s*(?P<msg>.*)$"
)

RE_FAILED_PASSWORD = re.compile(
    r"Failed password for (?:invalid user )?(?P<user>\S+) from (?P<ip>\S+) port (?P<port>\d+)"
)
RE_ACCEPTED = re.compile(
    r"Accepted (?P<method>password|publickey) for (?P<user>\S+) from (?P<ip>\S+) port (?P<port>\d+)"
)
RE_INVALID_USER = re.compile(r"Invalid user (?P<user>\S+) from (?P<ip>\S+)")
RE_DISCONNECT = re.compile(r"Disconnected from (?:invalid user (?P<user>\S+) )?(?P<ip>\S+)")
RE_SUDO = re.compile(r"(?P<user>\S+)\s*:\s*.*?COMMAND=(?P<cmd>.*)$")
RE_CRON = re.compile(r"\((?P<user>\S+)\) CMD \((?P<cmd>.*)\)")

_MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


def _syslog_timestamp(mon: str, day: str, time_str: str, assumed_year: int) -> str:
    from datetime import datetime, timezone

    hour, minute, second = (int(part) for part in time_str.split(":"))
    dt = datetime(
        assumed_year, _MONTHS.get(mon, 1), int(day), hour, minute, second, tzinfo=timezone.utc
    )
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass
class LoadStats:
    """Counters returned alongside the events (surfaced in the ingest CLI output)."""

    files: int = 0
    records: int = 0
    skipped: int = 0

    def merge(self, other: LoadStats) -> LoadStats:
        self.files += other.files
        self.records += other.records
        self.skipped += other.skipped
        return self

    def as_dict(self) -> dict[str, int]:
        return {"files": self.files, "records": self.records, "skipped": self.skipped}


# --------------------------------------------------------------------------- #
# CloudTrail
# --------------------------------------------------------------------------- #

_IDENTITY_KEYS = ("arn", "userName", "principalId", "accountId", "type")


def _principal_from_identity(identity: dict[str, Any] | None) -> str:
    if not identity:
        return ""
    arn = identity.get("arn")
    if arn:
        return str(arn)
    user = identity.get("userName")
    account = identity.get("accountId", "")
    if user:
        return f"arn:aws:iam::{account}:user/{user}"
    return str(identity.get("principalId", ""))


def _resource_from_request(event_name: str, request: Any, response: Any) -> str:
    """Best-effort resource extraction from ``requestParameters``/``responseElements``."""
    for payload in (request, response):
        if not isinstance(payload, dict):
            continue
        for key in (
            "bucketName",
            "bucket",
            "userName",
            "groupName",
            "roleName",
            "policyName",
            "trailName",
            "groupName",
            "instanceId",
            "functionName",
            "queueUrl",
            "topicArn",
            "secretId",
            "keyId",
            "dbInstanceIdentifier",
        ):
            value = payload.get(key)
            if value:
                return f"{key}={value}"
        security_group = payload.get("groupId") or payload.get("groupName")
        if security_group and event_name.startswith(("Authorize", "Revoke", "Modify")):
            return f"securityGroup={security_group}"
    return ""


def cloudtrail_record_to_event(
    record: dict[str, Any], *, raw_ref: str = "", raw: str = ""
) -> Event | None:
    """Convert one CloudTrail JSON record into an :class:`Event` (``None`` if unusable)."""
    if not isinstance(record, dict):
        return None

    event_time = record.get("eventTime")
    event_name = record.get("eventName")
    if not event_time or not event_name:
        return None

    ts = str(event_time)
    if ts.endswith("+00:00"):
        ts = ts[:-6] + "Z"

    identity = record.get("userIdentity") if isinstance(record.get("userIdentity"), dict) else {}
    error_code = record.get("errorCode") or record.get("errorMessage")
    status = normalize_status("failure" if record.get("errorCode") else "success")

    event_id = str(record.get("eventID") or "").strip()
    if not event_id:
        event_id = stable_id(
            "ct",
            ts,
            str(event_name),
            _principal_from_identity(identity),
            str(record.get("sourceIPAddress", "")),
        )
    else:
        event_id = f"ct-{event_id}"

    return Event(
        event_id=event_id,
        ts=ts,
        source="cloudtrail",
        action=str(event_name),
        status=status,
        principal=_principal_from_identity(identity),
        src_ip=str(record.get("sourceIPAddress") or ""),
        region=str(record.get("awsRegion") or ""),
        account_id=str(
            identity.get("accountId") or record.get("recipientAccountId") or ""
        ),
        resource=_resource_from_request(
            str(event_name), record.get("requestParameters"), record.get("responseElements")
        ),
        user_agent=str(record.get("userAgent") or ""),
        raw=raw or json.dumps(record, separators=(",", ":")),
        raw_ref=raw_ref,
        extra={
            "event_source": record.get("eventSource", ""),
            "error_code": error_code or "",
            "identity_type": identity.get("type", ""),
            "read_only": bool(record.get("readOnly", False)),
        },
    )


def iter_cloudtrail_file(path: Path, stats: LoadStats | None = None) -> Iterator[Event]:
    """Yield events from a CloudTrail file (JSON array **or** JSON-lines)."""
    stats = stats if stats is not None else LoadStats()
    stats.files += 1

    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return

    records: Iterable[Any]
    if text.startswith("["):
        try:
            records = json.loads(text)
        except json.JSONDecodeError:
            stats.skipped += 1
            return
    else:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                stats.skipped += 1

    for index, record in enumerate(records, start=1):
        stats.records += 1
        raw_ref = f"{path.name}:{index}"
        try:
            event = cloudtrail_record_to_event(record, raw_ref=raw_ref)
        except (TypeError, ValueError, AttributeError):
            stats.skipped += 1
            continue
        if event is None:
            stats.skipped += 1
            continue
        yield event


# --------------------------------------------------------------------------- #
# syslog / auth.log
# --------------------------------------------------------------------------- #


_AUTH_PROCESSES = {"sshd", "sudo", "su", "cron", "systemd-logind"}

#: File-name hints that identify an authentication stream rather than a general syslog.
_AUTH_FILE_HINTS = ("auth", "ssh", "secure")


def syslog_line_to_event(
    line: str, *, assumed_year: int, raw_ref: str = "", source: str | None = None
) -> Event | None:
    """Convert one syslog line into an :class:`Event` (``None`` if unparseable).

    ``source`` lets the caller declare which stream the line came from (``iter_syslog_file``
    derives it from the file name). When omitted, it is inferred from the emitting process.
    """
    match = SYSLOG_PREFIX.match(line.strip())
    if not match:
        return None

    parts = match.groupdict()
    ts = _syslog_timestamp(parts["mon"], parts["day"], parts["time"], assumed_year)
    message = parts["msg"] or ""
    process = parts["proc"] or "unknown"

    action = f"{process}.message"
    status = "info"
    principal = ""
    src_ip = ""
    resource = ""

    failed = RE_FAILED_PASSWORD.search(message)
    accepted = RE_ACCEPTED.search(message)
    invalid = RE_INVALID_USER.search(message)
    sudo = RE_SUDO.search(message)
    cron = RE_CRON.search(message)

    if failed:
        action, status = "sshd.failed_password", "failure"
        principal, src_ip = failed.group("user"), failed.group("ip")
    elif accepted:
        action, status = "sshd.accepted_password", "success"
        principal, src_ip = accepted.group("user"), accepted.group("ip")
    elif invalid:
        action, status = "sshd.invalid_user", "failure"
        principal, src_ip = invalid.group("user"), invalid.group("ip")
    elif "session opened" in message:
        action, status = "sshd.session_opened", "success"
    elif "session closed" in message:
        action, status = "sshd.session_closed", "info"
    elif sudo:
        action, status = "sudo.command", "success"
        principal, resource = sudo.group("user"), sudo.group("cmd")[:160]
    elif cron:
        action, status = "cron.job", "info"
        principal, resource = cron.group("user"), cron.group("cmd")[:160]
    elif process == "kernel":
        action = "kernel.message"
        status = normalize_status(message)
    elif "systemd" in process:
        action = "systemd.unit"
        status = normalize_status(message)

    event_id = stable_id("sl", raw_ref or "", line.strip())

    return Event(
        event_id=event_id,
        ts=ts,
        source=source or ("auth" if process in _AUTH_PROCESSES else "syslog"),
        action=action,
        status=status,
        principal=principal,
        src_ip=src_ip,
        region="",
        account_id="",
        resource=resource,
        raw=line.strip(),
        raw_ref=raw_ref,
        extra={"host": parts["host"] or "", "pid": parts["pid"] or "", "process": process},
    )


def iter_syslog_file(
    path: Path, *, assumed_year: int = 2026, stats: LoadStats | None = None
) -> Iterator[Event]:
    """Yield events from a syslog-style text file, skipping unparseable lines."""
    stats = stats if stats is not None else LoadStats()
    stats.files += 1

    file_source = (
        "auth" if any(hint in path.name.lower() for hint in _AUTH_FILE_HINTS) else "syslog"
    )

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            stats.records += 1
            event = syslog_line_to_event(
                line,
                assumed_year=assumed_year,
                raw_ref=f"{path.name}:{line_no}",
                source=file_source,
            )
            if event is None:
                stats.skipped += 1
                continue
            yield event


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #

SYSLOG_SUFFIXES = {".log", ".txt", ".out"}
JSON_SUFFIXES = {".json", ".jsonl", ".ndjson"}

#: Extension-less log file names that must still be picked up (``/var/log/syslog``).
KNOWN_SYSLOG_NAMES = {"syslog", "auth", "messages", "secure", "sshd", "daemon", "kern", "user"}

#: Files that look like JSON but are pipeline artefacts, not logs.
_EXCLUDED_STEMS = {"manifest", "labels", "eval_set", "eval-set", "index", "stats", "config"}


def is_log_file(path: Path) -> bool:
    """True when ``path`` should be treated as a log file (not a pipeline artefact)."""
    name = path.name.lower()
    if name.startswith("."):
        return False
    suffix = path.suffix.lower()
    if suffix in JSON_SUFFIXES:
        return path.stem.lower() not in _EXCLUDED_STEMS
    if suffix in SYSLOG_SUFFIXES:
        return True
    return name in KNOWN_SYSLOG_NAMES


def detect_source(path: Path) -> str:
    """Guess the log family from the file name/extension."""
    name = path.name.lower()
    if path.suffix.lower() in JSON_SUFFIXES or "cloudtrail" in name:
        return "cloudtrail"
    if path.suffix.lower() in SYSLOG_SUFFIXES or name.startswith(("auth", "syslog", "ssh")):
        return "syslog"
    return "cloudtrail"


def discover_files(paths: Iterable[Path]) -> list[Path]:
    """Expand directories into a sorted list of log files.

    Explicitly-passed files are always kept (the caller asked for them); directory walks
    filter out pipeline artefacts such as ``manifest.json`` / ``labels.json`` and keep
    extension-less logs such as ``syslog``.
    """
    found: list[Path] = []
    for path in paths:
        if path.is_dir():
            for candidate in sorted(path.rglob("*")):
                if candidate.is_file() and is_log_file(candidate):
                    found.append(candidate)
        elif path.is_file():
            found.append(path)
    return found


def load_events(
    paths: Iterable[Path],
    *,
    assumed_year: int = 2026,
    stats: LoadStats | None = None,
) -> Iterator[Event]:
    """Load events from any mix of CloudTrail and syslog files."""
    stats = stats if stats is not None else LoadStats()
    for path in discover_files(paths):
        kind = detect_source(path)
        if kind == "cloudtrail":
            yield from iter_cloudtrail_file(path, stats)
        else:
            yield from iter_syslog_file(path, assumed_year=assumed_year, stats=stats)


__all__ = [
    "LoadStats",
    "cloudtrail_record_to_event",
    "detect_source",
    "discover_files",
    "is_log_file",
    "iter_cloudtrail_file",
    "iter_syslog_file",
    "load_events",
    "syslog_line_to_event",
]
