# DATA — corpus composition, scenario labels, licensing

## 1. Why a hybrid corpus

We need three properties that no single source provides:

| Property | Synthetic generator | Loghub (real) |
|---|---|---|
| Volume ≥ 100K records | ✅ ~200K | ✅ (SSH alone is 655K lines) |
| **Ground-truth labels** for accuracy measurement | ✅ planted scenarios | ❌ unlabelled |
| Real-world formatting quirks | ⚠️ modelled | ✅ authentic |

So: **synthetic CloudTrail (labelled) + real Loghub SSH/Linux (realism)**.

> Loghub contains **no AWS CloudTrail dataset**. CloudTrail's nested JSON structure
> (`userIdentity`, `requestParameters`, `responseElements`) is what a security RAG has to
> handle, so it is generated with full fidelity and planted attack chains.

## 2. Synthetic corpus (`scripts/generate_logs.py`)

Deterministic: same `--seed` ⇒ byte-identical output (CI enforces this).

| File | Records | Notes |
|---|---|---|
| `data/raw/cloudtrail/cloudtrail-<date>.json` | ~120,000 | one file per day, JSON array of records |
| `data/raw/syslog/auth.log` | ~60,000 | sshd / sudo / su / cron |
| `data/raw/syslog/syslog` | ~20,000 | kernel / systemd |

CloudTrail record shape (real field names, abridged):

```json
{
  "eventVersion": "1.08",
  "eventID": "6f1c9b3e-...",
  "eventTime": "2026-08-14T02:11:03Z",
  "eventName": "ConsoleLogin",
  "eventSource": "signin.amazonaws.com",
  "awsRegion": "us-east-1",
  "sourceIPAddress": "203.0.113.44",
  "userAgent": "Mozilla/5.0 ...",
  "errorCode": "FailedAuthentication",
  "userIdentity": {"type": "IAMUser", "principalId": "...", "arn": "arn:aws:iam::123456789012:user/alice", "accountId": "123456789012", "userName": "alice"},
  "requestParameters": {"...": "..."},
  "responseElements": null
}
```

## 3. Injected scenarios (ground truth)

| ID | Scenario | Planted signature | Typical question |
|---|---|---|---|
| S1 | Console brute force | 200+ `ConsoleLogin` `FailedAuthentication` from one IP, then a success | "failed console logins from 203.0.113.44" |
| S2 | SSH brute force | 400+ `Failed password`, then `Accepted password` | "brute force attempts against ssh" |
| S3 | IAM privilege escalation | `CreateUser` → `AttachUserPolicy(AdministratorAccess)` → `CreateAccessKey` | "did anyone escalate privileges" |
| S4 | S3 bucket made public | `PutBucketPolicy` with `Principal: "*"`, `PutBucketAcl public-read` | "which buckets were made public" |
| S5 | Security group opened | `AuthorizeSecurityGroupIngress` `0.0.0.0/0` on 22/3389 | "security groups opened to the internet" |
| S6 | Logging tampering | `StopLogging`, `DeleteTrail` | "was audit logging disabled" |
| S7 | Root account usage | `ConsoleLogin` as `Root` from an unusual IP | "root account activity" |
| S8 | Credential stuffing | many `AssumeRole`/`GetCallerIdentity` failures across users | "assume role failures" |
| S9 | Impossible travel | same user, two continents, 12 min apart | "unusual login locations for alice" |
| S10 | Data exfiltration | 3,000 `GetObject` on one bucket in 20 min | "unusual S3 download volume" |

Labels are written to `data/eval/labels.json`:

```json
{
  "seed": 1337,
  "generated_at": "2026-09-23T13:00:00Z",
  "scenarios": [
    {
      "id": "S3",
      "type": "iam_privilege_escalation",
      "principal": "arn:aws:iam::123456789012:user/bob",
      "src_ip": "198.51.100.23",
      "ts_start": "2026-08-19T04:12:00Z",
      "ts_end": "2026-08-19T04:31:00Z",
      "event_ids": ["ct-3f9a...", "ct-7b21..."],
      "expected_queries": ["did anyone escalate privileges", "new IAM users with admin policy"],
      "notes": "CreateUser -> AttachUserPolicy(AdministratorAccess) -> CreateAccessKey"
    }
  ]
}
```

`event_ids` is the ground-truth answer set the eval harness scores against.

## 4. Real logs (optional, recommended)

```bash
bash scripts/download_loghub.sh          # Linux + OpenSSH ≈ 72 MB → data/raw/loghub/
bash scripts/download_loghub.sh --all    # adds Apache + Zookeeper + Mac ≈ 103 MB
```

| Dataset | Lines | Download |
|---|---|---|
| Linux | 25,567 | `https://zenodo.org/records/8196385/files/Linux.tar.gz?download=1` |
| OpenSSH | 655,146 | `https://zenodo.org/records/8196385/files/SSH.tar.gz?download=1` |
| Apache | 56,481 | `https://zenodo.org/records/8196385/files/Apache.tar.gz?download=1` |
| Zookeeper | 74,380 | `https://zenodo.org/records/8196385/files/Zookeeper.tar.gz?download=1` |
| Mac | 117,283 | `https://zenodo.org/records/8196385/files/Mac.tar.gz?download=1` |

**License:** Loghub datasets are *freely available for research or academic work*.
Cite: Jieming Zhu, Shilin He, Pinjia He, Jinyang Liu, Michael R. Lyu. *Loghub: A Large
Collection of System Log Datasets for AI-driven Log Analytics.* ISSRE 2023.
Repo: <https://github.com/logpai/loghub>

## 5. What is committed vs generated

Committed: `scripts/generate_logs.py`, `scripts/download_loghub.sh`, `docs/DATA.md`,
`data/eval/eval_set.jsonl` (the curated questions — small, and the labels must be
reproducible for anyone reviewing the repo).
**Not** committed: raw logs, embeddings, SQLite, Chroma. They are one command away
(`make corpus`) and would bloat the repo.
