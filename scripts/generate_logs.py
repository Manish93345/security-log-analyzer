#!/usr/bin/env python3
"""Generate a labelled synthetic security-log corpus.

Produces ~200,000 records by default (``--full``):

* ``cloudtrail/cloudtrail-<date>.json``  ~120,000 CloudTrail records (JSON array per day)
* ``syslog/auth.log``                    ~60,000 sshd / sudo / cron lines
* ``syslog/syslog``                      ~20,000 kernel / systemd lines
* ``../eval/labels.json``                ground truth for 10 injected attack scenarios

**Determinism.** Same ``--seed`` and same date window => byte-identical output. Wall-clock
timestamps are deliberately absent from the output files (CI asserts determinism by hashing
two runs).

**Time anchoring.** ``--end-date`` (default: today, UTC) is the corpus "now". Time-scoped
questions in the eval set are anchored to it, so "last 24h" has a stable meaning.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

# --------------------------------------------------------------------------- #
# corpus constants
# --------------------------------------------------------------------------- #

USERS = ["alice", "bob", "carol", "dave", "erin", "frank", "grace", "heidi", "ivan", "judy"]
ROLES = ["AdminRole", "ReadOnlyRole", "DevOpsRole", "AuditRole", "DataEngineerRole"]
BUCKETS = [
    "prod-app-logs",
    "customer-exports",
    "backup-archive",
    "ml-training-data",
    "static-assets",
]
INSTANCES = [
    "i-0a1b2c3d4e5f60001",
    "i-0a1b2c3d4e5f60002",
    "i-0a1b2c3d4e5f60003",
    "i-0a1b2c3d4e5f60004",
    "i-0a1b2c3d4e5f60005",
]
SECURITY_GROUPS = ["sg-0a11bb22cc33dd440", "sg-0e55ff66aa77bb880"]
TRAIL_NAME = "org-audit-trail"
REGIONS = [
    "us-east-1",
    "us-west-2",
    "eu-west-1",
    "ap-southeast-1",
    "ap-northeast-1",
    "sa-east-1",
]
ACCOUNTS = ["123456789012", "210987654321", "345678901234"]

BENIGN_IPS = [
    "10.0.1.23",
    "10.0.2.44",
    "10.0.3.17",
    "172.16.5.8",
    "192.168.10.5",
    "198.51.100.10",
    "198.51.100.11",
    "203.0.113.5",
]
ATTACKER_IPS = [
    "203.0.113.44",
    "198.51.100.23",
    "192.0.2.77",
    "203.0.113.201",
    "198.51.100.9",
]
UNUSUAL_IPS = ["41.203.77.12", "185.220.101.34", "103.75.190.8"]

USER_AGENTS = [
    "aws-cli/2.15.17 Python/3.12.1 Linux/6.5.0 exec-env/AWS_ECS_FARGATE",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/120.0 Safari/537.36",
    "aws-sdk-java/2.25.60 Linux/5.15.0 OpenJDK_64-Bit_Server_VM/17.0.9",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/119.0 Safari/537.36",
    "Boto3/1.34.34 Python/3.11.6 Linux/6.2.0 Botocore/1.34.34",
    "Terraform/1.7.2 (+https://www.terraform.io)",
]

DATA_EVENTS = {"GetObject", "PutObject", "HeadObject", "ListBucket", "DeleteObject"}

#: (event name, aws service, relative weight, mutating?)
BACKGROUND_EVENTS: list[tuple[str, str, int, bool]] = [
    ("DescribeInstances", "ec2.amazonaws.com", 12, False),
    ("GetObject", "s3.amazonaws.com", 30, False),
    ("PutObject", "s3.amazonaws.com", 14, True),
    ("ListBuckets", "s3.amazonaws.com", 8, False),
    ("DescribeSecurityGroups", "ec2.amazonaws.com", 8, False),
    ("AssumeRole", "sts.amazonaws.com", 10, True),
    ("GetCallerIdentity", "sts.amazonaws.com", 10, False),
    ("ConsoleLogin", "signin.amazonaws.com", 8, False),
    ("StartInstances", "ec2.amazonaws.com", 4, True),
    ("StopInstances", "ec2.amazonaws.com", 4, True),
    ("CreateTags", "ec2.amazonaws.com", 6, True),
    ("DescribeLogGroups", "logs.amazonaws.com", 5, False),
    ("GetSecretValue", "secretsmanager.amazonaws.com", 3, False),
    ("DescribeDBInstances", "rds.amazonaws.com", 5, False),
    ("ListUsers", "iam.amazonaws.com", 4, False),
    ("GetUser", "iam.amazonaws.com", 4, False),
    ("DescribeCluster", "eks.amazonaws.com", 3, False),
    ("InvokeFunction", "lambda.amazonaws.com", 4, True),
    ("UpdateFunctionCode", "lambda.amazonaws.com", 2, True),
    ("DescribeTable", "dynamodb.amazonaws.com", 4, False),
    ("Query", "dynamodb.amazonaws.com", 6, False),
    ("SendMessage", "sqs.amazonaws.com", 5, True),
    ("PutMetricData", "monitoring.amazonaws.com", 4, True),
    ("DescribeTrails", "cloudtrail.amazonaws.com", 3, False),
    ("ListAccessKeys", "iam.amazonaws.com", 3, False),
    ("GetBucketPolicy", "s3.amazonaws.com", 4, False),
]

#: Background sessions are bursty: real AWS traffic arrives in short bursts (a CLI
#: session, a CI job, a Lambda run), not uniformly spread over the day.
MIN_SESSION_EVENTS = 3
MAX_SESSION_EVENTS = 28

#: Scenario placement: (id, type, day offset back from end date, hour UTC, minute)
SCENARIO_PLAN = [
    ("S1", "console_brute_force", 0, 2, 0),
    ("S2", "ssh_brute_force", 1, 3, 10),
    ("S3", "iam_privilege_escalation", 2, 4, 12),
    ("S4", "s3_bucket_public", 3, 5, 20),
    ("S5", "security_group_open_to_world", 4, 6, 30),
    ("S6", "logging_tampering", 5, 7, 40),
    ("S7", "root_account_usage", 6, 8, 15),
    ("S8", "credential_stuffing", 8, 9, 25),
    ("S9", "impossible_travel", 10, 10, 5),
    ("S10", "s3_data_exfiltration", 12, 11, 45),
]

HOSTNAME = "ip-10-0-1-23"


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _uuid(rng: random.Random) -> str:
    return str(uuid.UUID(int=rng.getrandbits(128)))


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _syslog_ts(dt: datetime) -> str:
    """Classic syslog timestamp: ``Aug  4 02:11:03`` (day space-padded)."""
    return f"{dt.strftime('%b')} {dt.day:2d} {dt.strftime('%H:%M:%S')}"


def _iam_user(rng: random.Random, user: str, account: str) -> dict:
    return {
        "type": "IAMUser",
        "principalId": "AIDA" + "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567") for _ in range(16)),
        "arn": f"arn:aws:iam::{account}:user/{user}",
        "accountId": account,
        "accessKeyId": "ASIA" + "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567") for _ in range(16)),
        "userName": user,
    }


def _assumed_role(rng: random.Random, user: str, account: str, role: str) -> dict:
    return {
        "type": "AssumedRole",
        "principalId": f"AROA{rng.randrange(10**12):012d}:{user}",
        "arn": f"arn:aws:sts::{account}:assumed-role/{role}/{user}",
        "accountId": account,
        "accessKeyId": "ASIA" + "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ234567") for _ in range(16)),
        "sessionContext": {
            "sessionIssuer": {
                "type": "Role",
                "principalId": f"AROA{rng.randrange(10**12):012d}",
                "arn": f"arn:aws:iam::{account}:role/{role}",
                "accountId": account,
                "userName": role,
            },
            "attributes": {"mfaAuthenticated": "false", "creationDate": "2026-08-01T00:00:00Z"},
        },
    }


def _root_identity(account: str) -> dict:
    return {
        "type": "Root",
        "principalId": account,
        "arn": f"arn:aws:iam::{account}:root",
        "accountId": account,
    }


def _request_for(event_name: str, rng: random.Random, *, bucket: str | None = None) -> dict:
    target_bucket = bucket or rng.choice(BUCKETS)
    if event_name in DATA_EVENTS:
        return {"bucketName": target_bucket, "key": f"logs/{rng.randrange(10**6):06d}.json.gz"}
    if event_name == "ListBuckets":
        return {}
    if event_name == "ConsoleLogin":
        return {"additionalEventData": {"MFAUsed": rng.choice(["Yes", "No"])}}
    if event_name.startswith(("Describe", "List", "Get")) and event_name not in {
        "GetSecretValue",
        "GetBucketPolicy",
    }:
        return {"maxResults": 100}
    if event_name == "GetSecretValue":
        return {"secretId": f"arn:aws:secretsmanager:us-east-1:123456789012:secret:prod/{rng.choice(['db', 'api', 'stripe'])}"}
    if event_name == "GetBucketPolicy":
        return {"bucketName": target_bucket}
    if event_name == "AssumeRole":
        return {
            "roleArn": f"arn:aws:iam::123456789012:role/{rng.choice(ROLES)}",
            "roleSessionName": f"session-{rng.randrange(10**4):04d}",
            "durationSeconds": 3600,
        }
    if event_name in {"StartInstances", "StopInstances"}:
        return {"instancesSet": {"items": [{"instanceId": rng.choice(INSTANCES)}]}}
    if event_name == "CreateTags":
        return {"resourcesSet": {"items": [{"resourceId": rng.choice(INSTANCES)}]}, "tagSet": {"items": [{"key": "env", "value": rng.choice(["prod", "dev"])}]}}
    if event_name == "InvokeFunction":
        return {"functionName": "log-processor", "invocationType": "Event"}
    if event_name == "UpdateFunctionCode":
        return {"functionName": "log-processor"}
    if event_name in {"Query", "DescribeTable"}:
        return {"tableName": "audit-events"}
    if event_name == "SendMessage":
        return {"queueUrl": "https://sqs.us-east-1.amazonaws.com/123456789012/ingest"}
    if event_name == "DescribeCluster":
        return {"name": "prod-cluster"}
    if event_name == "ListUsers" or event_name == "GetUser":
        return {"userName": rng.choice(USERS)}
    if event_name == "ListAccessKeys":
        return {"userName": rng.choice(USERS)}
    return {}


def _ct_record(
    rng: random.Random,
    ts: datetime,
    event_name: str,
    event_source: str,
    identity: dict,
    ip: str,
    region: str,
    account: str,
    *,
    status: str = "success",
    request: dict | None = None,
    response: dict | None = None,
    error_code: str | None = None,
    user_agent: str | None = None,
) -> dict:
    """Build one CloudTrail record with real field names."""
    record = {
        "eventVersion": "1.08",
        "userIdentity": identity,
        "eventTime": _iso(ts),
        "eventSource": event_source,
        "eventName": event_name,
        "awsRegion": region,
        "sourceIPAddress": ip,
        "userAgent": user_agent or rng.choice(USER_AGENTS),
        "requestParameters": request if request is not None else _request_for(event_name, rng),
        "responseElements": response,
        "requestID": _uuid(rng),
        "eventID": _uuid(rng),
        "readOnly": event_name not in DATA_EVENTS and not event_name.startswith(("Put", "Create", "Delete", "Update", "Attach", "Add", "Authorize", "Stop", "Start", "Modify", "Invoke", "Send")),
        "eventType": "AwsApiCall",
        "managementEvent": event_name not in DATA_EVENTS,
        "recipientAccountId": account,
    }
    if status == "failure":
        record["errorCode"] = error_code or "AccessDenied"
        record["errorMessage"] = (
            f"User: {identity.get('arn', 'unknown')} is not authorized to perform: "
            f"{event_name} on resource: * because no identity-based policy allows the "
            f"{event_name} action"
        )
    return record


def _pick_background_event(rng: random.Random) -> tuple[str, str, bool]:
    names = [row[0] for row in BACKGROUND_EVENTS]
    weights = [row[2] for row in BACKGROUND_EVENTS]
    name, source, _, mutating = rng.choices(BACKGROUND_EVENTS, weights=weights, k=1)[0]
    del names, weights
    return name, source, mutating


# --------------------------------------------------------------------------- #
# scenario injection
# --------------------------------------------------------------------------- #


def _inject_scenarios(
    rng: random.Random, end_date: datetime, day_buckets: dict[str, list[dict]], days: int
) -> list[dict]:
    """Plant the 10 attack scenarios and return their ground-truth labels.

    Scenario day offsets are taken modulo ``days`` so that a short corpus (used by the
    tests and by ``--days 2`` smoke runs) still gets every scenario placed inside the
    generated window instead of raising ``KeyError`` on a missing day bucket.
    """
    labels: list[dict] = []
    span = max(1, days)

    def bucket_for(dt: datetime) -> list[dict]:
        return day_buckets[dt.strftime("%Y-%m-%d")]

    def add(dt: datetime, record: dict, events: list[str]) -> None:
        bucket_for(dt).append(record)
        events.append("ct-" + record["eventID"])

    for scenario_id, kind, planned_offset, hour, minute in SCENARIO_PLAN:
        day_offset = planned_offset % span
        day = end_date - timedelta(days=day_offset)
        start = day.replace(hour=hour, minute=minute, second=0, microsecond=0)
        account = ACCOUNTS[0]
        event_ids: list[str] = []
        principal = ""
        src_ip = ""
        notes = ""
        expected: list[str] = []

        if kind == "console_brute_force":
            src_ip = ATTACKER_IPS[0]
            principal = f"arn:aws:iam::{account}:user/alice"
            identity = _iam_user(rng, "alice", account)
            for i in range(240):
                ts = start + timedelta(seconds=i * 6)
                record = _ct_record(
                    rng,
                    ts,
                    "ConsoleLogin",
                    "signin.amazonaws.com",
                    identity,
                    src_ip,
                    "us-east-1",
                    account,
                    status="failure",
                    error_code="FailedAuthentication",
                )
                add(ts, record, event_ids)
            ts = start + timedelta(minutes=25)
            record = _ct_record(
                rng, ts, "ConsoleLogin", "signin.amazonaws.com", identity, src_ip,
                "us-east-1", account, status="success",
            )
            add(ts, record, event_ids)
            notes = "240 failed ConsoleLogin then one success from a single IP"
            expected = [
                "show all failed console logins",
                "failed console logins from 203.0.113.44",
                "brute force against the AWS console",
            ]

        elif kind == "ssh_brute_force":
            src_ip = ATTACKER_IPS[1]
            principal = "root"
            notes = "400 failed ssh passwords for root then a successful login"
            expected = [
                "ssh brute force attempts",
                "failed ssh passwords from 198.51.100.23",
                "successful ssh login after many failures",
            ]
            # syslog lines are emitted by the syslog writer; ids are recorded there.
            labels.append(
                {
                    "id": scenario_id,
                    "type": kind,
                    "principal": principal,
                    "src_ip": src_ip,
                    "ts_start": _iso(start),
                    "ts_end": _iso(start + timedelta(minutes=22)),
                    "event_ids": [],
                    "expected_queries": expected,
                    "notes": notes,
                    "stream": "syslog",
                }
            )
            continue

        elif kind == "iam_privilege_escalation":
            src_ip = ATTACKER_IPS[2]
            principal = f"arn:aws:iam::{account}:user/bob"
            identity = _iam_user(rng, "bob", account)
            steps = [
                ("CreateUser", "iam.amazonaws.com", {"userName": "svc-backup"}, {"user": {"userName": "svc-backup", "arn": f"arn:aws:iam::{account}:user/svc-backup"}}),
                ("CreateLoginProfile", "iam.amazonaws.com", {"userName": "svc-backup"}, {"loginProfile": {"userName": "svc-backup"}}),
                ("AttachUserPolicy", "iam.amazonaws.com", {"userName": "svc-backup", "policyArn": "arn:aws:iam::aws:policy/AdministratorAccess"}, None),
                ("CreateAccessKey", "iam.amazonaws.com", {"userName": "svc-backup"}, {"accessKey": {"accessKeyId": "AKIAIOSFODNN7EXAMPLE", "userName": "svc-backup"}}),
                ("AddUserToGroup", "iam.amazonaws.com", {"userName": "svc-backup", "groupName": "Admins"}, None),
            ]
            for i, (name, source, request, response) in enumerate(steps):
                ts = start + timedelta(minutes=i * 3)
                record = _ct_record(
                    rng, ts, name, source, identity, src_ip, "us-east-1", account,
                    request=request, response=response,
                )
                add(ts, record, event_ids)
            notes = "CreateUser -> CreateLoginProfile -> AttachUserPolicy(AdministratorAccess) -> CreateAccessKey -> AddUserToGroup"
            expected = [
                "did anyone escalate privileges",
                "new IAM users with administrator access",
                "access keys created for a new user",
            ]

        elif kind == "s3_bucket_public":
            src_ip = ATTACKER_IPS[3]
            principal = f"arn:aws:iam::{account}:user/carol"
            identity = _iam_user(rng, "carol", account)
            public_policy = {
                "Version": "2012-10-17",
                "Statement": [{"Effect": "Allow", "Principal": "*", "Action": "s3:GetObject", "Resource": "arn:aws:s3:::customer-exports/*"}],
            }
            steps = [
                ("PutBucketPolicy", {"bucketName": "customer-exports", "bucketPolicy": json.dumps(public_policy)}),
                ("PutBucketAcl", {"bucketName": "customer-exports", "AccessControlPolicy": {"Grants": [{"Grantee": {"URI": "http://acs.amazonaws.com/groups/global/AllUsers"}, "Permission": "READ"}]}}),
                ("PutPublicAccessBlock", {"bucketName": "customer-exports", "PublicAccessBlockConfiguration": {"BlockPublicAcls": False, "BlockPublicPolicy": False}}),
            ]
            for i, (name, request) in enumerate(steps):
                ts = start + timedelta(minutes=i * 4)
                record = _ct_record(rng, ts, name, "s3.amazonaws.com", identity, src_ip, "us-east-1", account, request=request)
                add(ts, record, event_ids)
            notes = "bucket policy set to Principal:* and public-read ACL applied"
            expected = [
                "which buckets were made public",
                "s3 bucket policy allowing everyone",
                "public access block disabled",
            ]

        elif kind == "security_group_open_to_world":
            src_ip = ATTACKER_IPS[4]
            principal = f"arn:aws:iam::{account}:user/dave"
            identity = _iam_user(rng, "dave", account)
            for i, port in enumerate((22, 3389, 5432)):
                ts = start + timedelta(minutes=i * 5)
                record = _ct_record(
                    rng, ts, "AuthorizeSecurityGroupIngress", "ec2.amazonaws.com", identity,
                    src_ip, "us-east-1", account,
                    request={
                        "groupId": SECURITY_GROUPS[0],
                        "ipPermissions": {"items": [{"ipProtocol": "tcp", "fromPort": port, "toPort": port, "ipRanges": {"items": [{"cidrIp": "0.0.0.0/0"}]}}]},
                    },
                )
                add(ts, record, event_ids)
            notes = "SSH/RDP/Postgres opened to 0.0.0.0/0"
            expected = [
                "security groups opened to the internet",
                "0.0.0.0/0 ingress rules added",
                "who opened port 22 to the world",
            ]

        elif kind == "logging_tampering":
            src_ip = ATTACKER_IPS[0]
            principal = f"arn:aws:iam::{account}:user/erin"
            identity = _iam_user(rng, "erin", account)
            steps = [
                ("StopLogging", {"name": TRAIL_NAME}),
                ("UpdateTrail", {"name": TRAIL_NAME, "s3BucketName": "attacker-staging"}),
                ("DeleteTrail", {"name": TRAIL_NAME}),
            ]
            for i, (name, request) in enumerate(steps):
                ts = start + timedelta(minutes=i * 2)
                record = _ct_record(rng, ts, name, "cloudtrail.amazonaws.com", identity, src_ip, "us-east-1", account, request=request)
                add(ts, record, event_ids)
            notes = "CloudTrail stopped, retargeted, then deleted"
            expected = [
                "was audit logging disabled",
                "cloudtrail trail deleted",
                "who stopped logging",
            ]

        elif kind == "root_account_usage":
            src_ip = UNUSUAL_IPS[0]
            principal = f"arn:aws:iam::{account}:root"
            identity = _root_identity(account)
            for i in range(6):
                ts = start + timedelta(minutes=i * 7)
                record = _ct_record(
                    rng, ts, "ConsoleLogin", "signin.amazonaws.com", identity, src_ip,
                    "sa-east-1", account, status="success",
                )
                add(ts, record, event_ids)
            notes = "root account console logins from an unusual ASN/region"
            expected = [
                "root account activity",
                "root console logins from an unusual location",
                "use of the account root user",
            ]

        elif kind == "credential_stuffing":
            src_ip = ATTACKER_IPS[1]
            principal = f"arn:aws:iam::{account}:user/frank"
            for i in range(150):
                user = USERS[i % len(USERS)]
                ts = start + timedelta(seconds=i * 9)
                identity = _iam_user(rng, user, account)
                record = _ct_record(
                    rng, ts, "AssumeRole", "sts.amazonaws.com", identity, src_ip,
                    rng.choice(REGIONS), account, status="failure",
                    error_code=rng.choice(["AccessDenied", "InvalidClientTokenId", "ExpiredToken"]),
                )
                add(ts, record, event_ids)
            notes = "150 AssumeRole failures spread across many usernames from one IP"
            expected = [
                "assume role failures",
                "credential stuffing against sts",
                "many users failing to assume a role from one address",
            ]

        elif kind == "impossible_travel":
            principal = f"arn:aws:iam::{account}:user/grace"
            identity = _iam_user(rng, "grace", account)
            hops = [("103.75.190.8", "ap-southeast-1"), ("41.203.77.12", "eu-west-1"), ("198.51.100.10", "us-east-1")]
            for i, (ip, region) in enumerate(hops):
                ts = start + timedelta(minutes=i * 6)
                record = _ct_record(rng, ts, "ConsoleLogin", "signin.amazonaws.com", identity, ip, region, account)
                add(ts, record, event_ids)
            src_ip = hops[0][0]
            notes = "same user logged in from three continents within 12 minutes"
            expected = [
                "impossible travel for grace",
                "unusual login locations for a user",
                "same user logging in from two continents",
            ]

        elif kind == "s3_data_exfiltration":
            src_ip = UNUSUAL_IPS[1]
            principal = f"arn:aws:iam::{account}:user/heidi"
            identity = _iam_user(rng, "heidi", account)
            for i in range(3000):
                ts = start + timedelta(seconds=int(i * 0.4))
                record = _ct_record(
                    rng, ts, "GetObject", "s3.amazonaws.com", identity, src_ip,
                    "us-east-1", account,
                    request={"bucketName": "customer-exports", "key": f"export/{i:05d}.csv"},
                )
                add(ts, record, event_ids)
            notes = "3000 GetObject calls on customer-exports within 20 minutes"
            expected = [
                "unusual s3 download volume",
                "data exfiltration from a bucket",
                "bulk object downloads by a single user",
            ]

        labels.append(
            {
                "id": scenario_id,
                "type": kind,
                "principal": principal,
                "src_ip": src_ip,
                "ts_start": _iso(start),
                "ts_end": _iso(start + timedelta(minutes=30)),
                "event_ids": event_ids,
                "expected_queries": expected,
                "notes": notes,
                "stream": "cloudtrail",
            }
        )

    return labels


# --------------------------------------------------------------------------- #
# syslog writers
# --------------------------------------------------------------------------- #


def _write_auth_log(
    path: Path, rng: random.Random, days: list[datetime], per_day: int, labels: list[dict]
) -> int:
    """Write auth.log: sshd sessions, failures, sudo, cron (+ the S2 brute force)."""
    lines: list[str] = []

    burst_remaining = 0
    burst_start = days[0] if days else datetime.now(timezone.utc)
    burst_user = ""
    burst_ip = ""

    for day in days:
        for _ in range(per_day):
            # Bursty arrival: consecutive log lines belong to one interactive session, so
            # they share a user + source IP and land within a few minutes of each other.
            # Uniform random timestamps would produce one-line incident windows.
            if burst_remaining <= 0:
                burst_start = day + timedelta(seconds=rng.randrange(86400))
                burst_remaining = rng.randint(2, 7)
                burst_user = rng.choice(USERS)
                burst_ip = rng.choice(BENIGN_IPS)
            burst_remaining -= 1
            ts = burst_start + timedelta(seconds=burst_remaining * rng.randint(5, 45))
            roll = rng.random()
            user = burst_user
            ip = burst_ip
            pid = rng.randrange(1000, 99999)
            stamp = _syslog_ts(ts)

            if roll < 0.34:
                lines.append(
                    f"{stamp} {HOSTNAME} sshd[{pid}]: Accepted password for {user} "
                    f"from {ip} port {rng.randrange(30000, 65000)} ssh2"
                )
                lines.append(
                    f"{stamp} {HOSTNAME} sshd[{pid}]: pam_unix(sshd:session): session opened "
                    f"for user {user} by (uid=0)"
                )
            elif roll < 0.52:
                lines.append(
                    f"{stamp} {HOSTNAME} sshd[{pid}]: Failed password for "
                    f"{rng.choice(['invalid user admin', 'invalid user test', user])} from "
                    f"{rng.choice(UNUSUAL_IPS + BENIGN_IPS)} port {rng.randrange(30000, 65000)} ssh2"
                )
            elif roll < 0.62:
                lines.append(
                    f"{stamp} {HOSTNAME} sshd[{pid}]: Invalid user "
                    f"{rng.choice(['admin', 'oracle', 'postgres', 'ubuntu', 'test'])} from "
                    f"{rng.choice(UNUSUAL_IPS)} port {rng.randrange(30000, 65000)}"
                )
            elif roll < 0.72:
                lines.append(
                    f"{stamp} {HOSTNAME} sshd[{pid}]: pam_unix(sshd:session): session closed "
                    f"for user {user}"
                )
            elif roll < 0.86:
                cmd = rng.choice(
                    ["/usr/bin/systemctl restart nginx", "/usr/bin/apt-get update",
                     "/bin/journalctl -u sshd", "/usr/bin/docker ps",
                     "/bin/cat /var/log/auth.log"]
                )
                lines.append(
                    f"{stamp} {HOSTNAME} sudo: {user} : TTY=pts/0 ; PWD=/home/{user} ; "
                    f"USER=root ; COMMAND={cmd}"
                )
            else:
                lines.append(
                    f"{stamp} {HOSTNAME} CRON[{pid}]: (root) CMD "
                    f"(/usr/local/bin/rotate-logs.sh)"
                )

    # --- S2: ssh brute force -------------------------------------------------
    s2 = next((label for label in labels if label["id"] == "S2"), None)
    if s2 is not None:
        start = datetime.fromisoformat(s2["ts_start"].replace("Z", "+00:00"))
        ip = s2["src_ip"]
        ids: list[str] = []
        for i in range(400):
            ts = start + timedelta(seconds=i * 3)
            pid = rng.randrange(1000, 99999)
            stamp = _syslog_ts(ts)
            line = (
                f"{stamp} {HOSTNAME} sshd[{pid}]: Failed password for root from {ip} "
                f"port {rng.randrange(30000, 65000)} ssh2"
            )
            lines.append(line)
            ids.append(line)
        ts = start + timedelta(minutes=21)
        pid = rng.randrange(1000, 99999)
        line = (
            f"{_syslog_ts(ts)} {HOSTNAME} sshd[{pid}]: Accepted password for root from {ip} "
            f"port {rng.randrange(30000, 65000)} ssh2"
        )
        lines.append(line)
        ids.append(line)
        # Record the ground truth as stable ids derived from the line content.
        s2["event_ids"] = [
            "sl-" + hashlib.sha1(f"auth.log:{idx}|{text.strip()}".encode()).hexdigest()[:12]
            for idx, text in enumerate(lines[-len(ids):], start=len(lines) - len(ids) + 1)
        ]
        s2["notes"] = f"{s2['notes']} (400 failed + 1 accepted)"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def _write_syslog(path: Path, rng: random.Random, days: list[datetime], per_day: int) -> int:
    """Write syslog: kernel / systemd / service noise."""
    lines: list[str] = []
    templates = [
        "{ts} {host} kernel: [{pid}] EXT4-fs (nvme0n1p1): mounted filesystem with ordered data mode",
        "{ts} {host} kernel: [{pid}] TCP: request_sock_TCP: Possible SYN flooding on port 443. Sending cookies",
        "{ts} {host} kernel: [{pid}] Out of memory: Killed process {pid} (python3) total-vm:2048000kB",
        "{ts} {host} systemd[1]: Started Daily apt download activities.",
        "{ts} {host} systemd[1]: Started Session {pid} of user {user}.",
        "{ts} {host} systemd[1]: Stopping User Manager for UID 1000...",
        "{ts} {host} systemd[1]: Failed to start Docker Application Container Engine.",
        "{ts} {host} chronyd[{pid}]: Selected source 169.254.169.123",
        "{ts} {host} systemd[1]: Reloading nginx.service.",
        "{ts} {host} kernel: [{pid}] audit: type=1400 audit(1755000000.123:456): apparmor=\"DENIED\" operation=\"open\"",
    ]
    for day in days:
        for _ in range(per_day):
            ts = day + timedelta(seconds=rng.randrange(86400))
            template = rng.choice(templates)
            lines.append(
                template.format(
                    ts=_syslog_ts(ts),
                    host=HOSTNAME,
                    pid=rng.randrange(1000, 99999),
                    user=rng.choice(USERS),
                )
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(
    out_dir: Path,
    *,
    seed: int = 1337,
    days: int = 30,
    events_per_day: int = 4000,
    ssh_per_day: int = 2000,
    syslog_per_day: int = 667,
    end_date: datetime | None = None,
    labels_path: Path | None = None,
    verbose: bool = True,
) -> dict:
    """Generate the corpus. Returns a manifest dict (counts + per-file sha256)."""
    out_dir = Path(out_dir)
    cloudtrail_dir = out_dir / "cloudtrail"
    syslog_dir = out_dir / "syslog"
    cloudtrail_dir.mkdir(parents=True, exist_ok=True)
    syslog_dir.mkdir(parents=True, exist_ok=True)

    end_date = (end_date or datetime.now(timezone.utc)).replace(
        hour=23, minute=59, second=0, microsecond=0
    )
    day_list = [
        (end_date - timedelta(days=offset)).replace(hour=0, minute=0, second=0, microsecond=0)
        for offset in range(days - 1, -1, -1)
    ]
    day_buckets: dict[str, list[dict]] = {day.strftime("%Y-%m-%d"): [] for day in day_list}

    rng_background = random.Random(seed)
    rng_scenarios = random.Random(seed + 1)
    rng_syslog = random.Random(seed + 2)

    # --- background CloudTrail traffic --------------------------------------
    # Real AWS API traffic is bursty: a CLI session, a CI job or a Lambda run issues a
    # handful of calls within a minute or two, then stops. Modelling that is what makes
    # incident windows coherent — uniform random timestamps produce ~one-event windows,
    # which defeats the entire retrieval design (a chunk must contain a trace, not a line).
    if verbose:
        print(
            f"[gen] background CloudTrail: {days} days x {events_per_day} events "
            f"in bursts of {MIN_SESSION_EVENTS}-{MAX_SESSION_EVENTS}"
        )
    for day in day_list:
        bucket = day_buckets[day.strftime("%Y-%m-%d")]
        produced = 0
        while produced < events_per_day:
            user = rng_background.choice(USERS)
            account = rng_background.choice(ACCOUNTS)
            if rng_background.random() < 0.25:
                identity = _assumed_role(
                    rng_background, user, account, rng_background.choice(ROLES)
                )
            else:
                identity = _iam_user(rng_background, user, account)
            ip = rng_background.choice(BENIGN_IPS)
            region = rng_background.choice(REGIONS)

            size = min(
                rng_background.randint(MIN_SESSION_EVENTS, MAX_SESSION_EVENTS),
                events_per_day - produced,
            )
            session_start = day + timedelta(seconds=rng_background.randrange(86400))
            duration_s = rng_background.randint(15, 420)
            day_end = day + timedelta(days=1)

            for index in range(size):
                ts = session_start + timedelta(
                    seconds=int(duration_s * index / max(1, size - 1))
                )
                if ts >= day_end:
                    ts = day_end - timedelta(seconds=1)
                name, source, _mutating = _pick_background_event(rng_background)
                status = "failure" if rng_background.random() < 0.035 else "success"
                bucket.append(
                    _ct_record(
                        rng_background,
                        ts,
                        name,
                        source,
                        identity,
                        ip,
                        region,
                        account,
                        status=status,
                        error_code="AccessDenied" if status == "failure" else None,
                    )
                )
            produced += size

    # --- injected scenarios -------------------------------------------------
    if verbose:
        print(f"[gen] injecting {len(SCENARIO_PLAN)} attack scenarios")
    labels = _inject_scenarios(rng_scenarios, end_date, day_buckets, days)

    # --- write CloudTrail files --------------------------------------------
    written_files: list[Path] = []
    cloudtrail_count = 0
    for day_key, records in day_buckets.items():
        records.sort(key=lambda record: (record["eventTime"], record["eventID"]))
        path = cloudtrail_dir / f"cloudtrail-{day_key}.json"
        path.write_text(json.dumps(records, indent=None, separators=(",", ":")), encoding="utf-8")
        written_files.append(path)
        cloudtrail_count += len(records)

    # --- syslog -------------------------------------------------------------
    auth_count = _write_auth_log(syslog_dir / "auth.log", rng_syslog, day_list, ssh_per_day, labels)
    syslog_count = _write_syslog(syslog_dir / "syslog", rng_syslog, day_list, syslog_per_day)
    written_files.extend([syslog_dir / "auth.log", syslog_dir / "syslog"])

    # --- labels -------------------------------------------------------------
    corpus = {
        "seed": seed,
        "days": days,
        "corpus_start_date": day_list[0].strftime("%Y-%m-%d"),
        "corpus_end_date": end_date.strftime("%Y-%m-%d"),
        "corpus_now": _iso(end_date),
        "scenarios": labels,
    }
    if labels_path is None:
        labels_path = out_dir.parent / "eval" / "labels.json"
    labels_path = Path(labels_path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(corpus, indent=2), encoding="utf-8")

    manifest = {
        "seed": seed,
        "days": days,
        "corpus_start_date": corpus["corpus_start_date"],
        "corpus_end_date": corpus["corpus_end_date"],
        "counts": {
            "cloudtrail_records": cloudtrail_count,
            "auth_log_lines": auth_count,
            "syslog_lines": syslog_count,
            "total_records": cloudtrail_count + auth_count + syslog_count,
            "scenarios": len(labels),
            "scenario_events": sum(len(label["event_ids"]) for label in labels),
        },
        "files": {path.name: _sha256(path) for path in written_files},
        # basename only — an absolute path would make the manifest machine-dependent and
        # break the byte-for-byte determinism check in CI.
        "labels_file": labels_path.name,
    }
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if verbose:
        counts = manifest["counts"]
        print(
            f"[gen] cloudtrail={counts['cloudtrail_records']:,} "
            f"auth.log={counts['auth_log_lines']:,} syslog={counts['syslog_lines']:,} "
            f"TOTAL={counts['total_records']:,}"
        )
        print(f"[gen] labels -> {labels_path}")
        print(f"[gen] manifest -> {manifest_path}")

    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a labelled synthetic security-log corpus."
    )
    parser.add_argument("--out", default="data/raw", help="output directory (default: data/raw)")
    parser.add_argument("--seed", type=int, default=1337, help="RNG seed (default: 1337)")
    parser.add_argument("--days", type=int, default=30, help="number of days (default: 30)")
    parser.add_argument(
        "--events-per-day", type=int, default=4000, help="CloudTrail events per day"
    )
    parser.add_argument("--ssh-per-day", type=int, default=2000, help="auth.log lines per day")
    parser.add_argument(
        "--syslog-per-day", type=int, default=667, help="syslog lines per day"
    )
    parser.add_argument(
        "--end-date", default=None, help="corpus end date YYYY-MM-DD (default: today UTC)"
    )
    parser.add_argument("--labels", default=None, help="labels.json output path")
    parser.add_argument(
        "--full",
        action="store_true",
        help="full corpus: 30 days x 4000 CloudTrail + 2000 auth.log + 667 syslog (~200K records)",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    if args.full:
        args.days = 30
        args.events_per_day = 4000
        args.ssh_per_day = 2000
        args.syslog_per_day = 667

    end_date = None
    if args.end_date:
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    manifest = generate(
        Path(args.out),
        seed=args.seed,
        days=args.days,
        events_per_day=args.events_per_day,
        ssh_per_day=args.ssh_per_day,
        syslog_per_day=args.syslog_per_day,
        end_date=end_date,
        labels_path=Path(args.labels) if args.labels else None,
        verbose=not args.quiet,
    )

    target = 100_000
    total = manifest["counts"]["total_records"]
    if not args.quiet:
        print(f"[gen] target >= {target:,} records -> {'OK' if total >= target else 'SHORT'}")
    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
