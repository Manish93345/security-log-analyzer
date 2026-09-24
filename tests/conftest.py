"""Shared pytest fixtures.

Unit tests must run with **no network, no API keys and no downloaded models** — anything
that needs those is marked ``integration`` or ``llm`` and excluded from the default run.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

for candidate in (str(SCRIPTS_DIR), str(SRC_DIR)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)


def _load_generator():
    """Import scripts/generate_logs.py by path (it is a script, not a package)."""
    spec = importlib.util.spec_from_file_location(
        "generate_logs", SCRIPTS_DIR / "generate_logs.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def generator():
    return _load_generator()


@pytest.fixture(scope="session")
def small_corpus(tmp_path_factory, generator) -> dict:
    """A tiny deterministic corpus (2 days) generated once per session."""
    out = tmp_path_factory.mktemp("corpus")
    manifest = generator.generate(
        out,
        seed=7,
        days=2,
        events_per_day=150,
        ssh_per_day=60,
        syslog_per_day=20,
        labels_path=out / "labels.json",
        verbose=False,
    )
    return {"dir": out, "manifest": manifest}


@pytest.fixture
def cloudtrail_record() -> dict:
    """One hand-written CloudTrail record (fixture, not generated)."""
    return {
        "eventVersion": "1.08",
        "userIdentity": {
            "type": "IAMUser",
            "principalId": "AIDAEXAMPLE",
            "arn": "arn:aws:iam::123456789012:user/alice",
            "accountId": "123456789012",
            "userName": "alice",
        },
        "eventTime": "2026-08-14T02:11:03Z",
        "eventSource": "signin.amazonaws.com",
        "eventName": "ConsoleLogin",
        "awsRegion": "us-east-1",
        "sourceIPAddress": "203.0.113.44",
        "userAgent": "Mozilla/5.0",
        "errorCode": "FailedAuthentication",
        "requestParameters": {"additionalEventData": {"MFAUsed": "No"}},
        "responseElements": None,
        "requestID": "11111111-1111-1111-1111-111111111111",
        "eventID": "22222222-2222-2222-2222-222222222222",
        "readOnly": False,
        "eventType": "AwsApiCall",
        "recipientAccountId": "123456789012",
    }


@pytest.fixture
def auth_lines() -> list[str]:
    """Hand-written syslog lines covering every parser branch."""
    return [
        "Aug 14 02:11:03 ip-10-0-1-23 sshd[2451]: Failed password for invalid user admin "
        "from 203.0.113.44 port 51234 ssh2",
        "Aug 14 02:11:09 ip-10-0-1-23 sshd[2452]: Accepted password for alice from 10.0.1.23 "
        "port 40122 ssh2",
        "Aug 14 02:11:10 ip-10-0-1-23 sshd[2452]: pam_unix(sshd:session): session opened for "
        "user alice by (uid=0)",
        "Aug 14 02:12:00 ip-10-0-1-23 sudo: alice : TTY=pts/0 ; PWD=/home/alice ; USER=root ; "
        "COMMAND=/usr/bin/systemctl restart nginx",
        "Aug 14 02:13:00 ip-10-0-1-23 CRON[2600]: (root) CMD (/usr/local/bin/rotate-logs.sh)",
        "Aug 14 02:14:00 ip-10-0-1-23 kernel: [1234] Out of memory: Killed process 999 (python3)",
        "Aug 14 02:15:00 ip-10-0-1-23 systemd[1]: Failed to start Docker Application Container Engine.",
        "this line has no syslog prefix and must be skipped",
    ]


@pytest.fixture
def write_jsonl(tmp_path):
    def _write(name: str, records: list[dict]) -> Path:
        path = tmp_path / name
        path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
        return path

    return _write
