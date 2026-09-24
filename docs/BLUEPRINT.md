# RAG-Powered Security Log Analyzer — Master Blueprint

> **Project codename:** `slrag` (Security Log RAG)
> **Owner:** you
> **Build window:** one day (this blueprint is written to be executed in ~7 focused hours)
> **Status:** Phase 0 (this document) — awaiting your sign-off, then Phase 1 starts

---

## 0. TL;DR — what we are actually building

A service that ingests raw security logs (AWS CloudTrail JSON + Linux `auth.log`/`syslog`),
normalizes every record into a common event schema, groups related events into
**incident windows**, embeds those windows into a vector store, and answers
natural-language questions like *"show all failed logins in the last 24h"* or
*"did anyone escalate privileges last week?"* — with **citations back to the exact raw log lines**.

It runs in three forms from one codebase:

| Form | Command | Purpose |
|---|---|---|
| CLI / pipeline | `python -m slrag.cli ask "..."` | fastest way to test during development |
| REST API | `uvicorn slrag.api.main:app` | programmatic access, the "deployable" story |
| Web UI | `streamlit run src/slrag/ui/app.py` | demo / screenshots / recruiter-friendly |

Everything is containerized with Docker, tested with pytest, and measured with a
**real retrieval eval harness** that prints honest numbers.

---

## 1. Non-negotiable constraints (from you)

1. **Zero paid APIs.** No OpenAI billing. Every model — embedding, reranker, LLM — must run on a free tier or locally.
2. **Finish today, with proper testing.** Tests are not an afterthought; every phase ships its own tests.
3. **Git push after every phase**, done properly (branch → commit → PR/merge → tag).
4. **All testing happens on your laptop.** This sandbox writes and verifies the code; you run it for real.
5. **Handoff document after every phase**, complete enough to resume cold after a token limit.
6. **100K+ logs of realistic test data** — with ground truth so accuracy can be *measured*, not guessed.

---

## 2. Free-model strategy (how we satisfy constraint #1 without gutting the project)

Your resume line says *OpenAI / `text-embedding-3-small`*. That stays **architecturally true** —
we build an OpenAI-compatible provider layer, so switching to OpenAI later is **one env var**.
But the default path uses free providers:

| Role | Default (free) | Alternatives (free) | Why this default |
|---|---|---|---|
| **LLM (generation)** | Groq — `openai/gpt-oss-120b` (30 RPM, 1000 req/day) | Google Gemini `3.5-flash` (15 RPM, 1500/day) · Ollama local · OpenRouter `:free` models · Cloudflare Workers AI | Groq is OpenAI-SDK-compatible, very fast, generous daily quota, no credit card |
| **LLM (offline fallback)** | Ollama `llama3.1:8b` (local) | any GGUF you already have | Works with the Wi-Fi off — good for a live demo |
| **Embeddings** | `fastembed` ONNX `BAAI/bge-small-en-v1.5` (384-d, ~90 MB, no torch) | `sentence-transformers` `BAAI/bge-base-en-v1.5` (768-d, better, slower) · `text-embedding-3-small` if you ever have credits | ONNX installs in seconds and embeds 40K chunks on a laptop CPU without a 2 GB torch download |
| **Reranker** | `FlashRank` (`ms-marco-MiniLM-L-6-v2`, ONNX, tiny) | `sentence-transformers` CrossEncoder `BAAI/bge-reranker-base` | Biggest single accuracy win per CPU-second in the whole pipeline |
| **Vector DB** | Chroma (persistent, local) | FAISS · Qdrant local | Matches the resume; zero infrastructure |
| **Structured store** | SQLite (stdlib) | DuckDB (optional upgrade) | Exact `COUNT`/`WHERE` over 200K rows in milliseconds, zero deps |

**Provider abstraction:** `src/slrag/providers/llm.py` returns a LangChain `BaseChatModel`
built from `LLM_PROVIDER` in `.env`. `ChatOpenAI(base_url=...)` covers Groq / OpenRouter /
Ollama / Together in one class — that is why this is low-risk.

> **Free-tier honesty note:** Gemini's free tier may use your prompts to improve Google's
> products; Groq's free tier does not train on your data. For a log analyzer, **default to
> Groq** and mention the privacy consideration in your README — it reads as mature engineering.

---

## 3. Architecture

```
                    ┌──────────────────────────────────────────────┐
   RAW LOGS         │  INGEST                                      │
 ┌───────────────┐  │  1. loaders.py   → CloudTrail JSON / auth.log │
 │ synthetic     │─▶│  2. schema.py    → normalize to Event()       │
 │ CloudTrail    │  │  3. windowing    → incident-window chunks     │
 │ synthetic SSH │  │  4. embed        → Chroma (dense)             │
 │ + Loghub real │  │  5. index        → SQLite (structured) + BM25 │
 └───────────────┘  └──────────────────────────────────────────────┘
                                     │
        ┌────────────────────────────┴───────────────────────────┐
        │  QUERY TIME — LangChain agent (tool-calling loop)       │
        │                                                         │
        │  question ──▶ [guardrail: injection screen]             │
        │       │                                                 │
        │       ├─▶ rewrite_query      (LLM: resolve "last 24h",  │
        │       │                        entity names, intent)     │
        │       ├─▶ ROUTE:                                        │
        │       │     • exact/aggregate  → SQL tool (SQLite)      │
        │       │     • semantic/hunt    → hybrid retrieval       │
        │       │       (dense + BM25 → RRF fuse → FlashRank)     │
        │       ├─▶ generate answer WITH chunk_id + event_id cites│
        │       └─▶ [guardrail: output screen + citation check]   │
        └─────────────────────────────────────────────────────────┘
                                     │
              ┌──────────────────────┼──────────────────────┐
              ▼                      ▼                      ▼
        FastAPI /ask           Streamlit UI           eval harness
```

**Why the dual path matters:** *"show all failed logins in the last 24h"* is a **filter +
count**, not a similarity search. Sending it to a vector store is the single most common
mistake in "RAG over logs" projects. Routing it to SQL makes the answer *exact*, which is
what an auditor needs — and it is what makes the 89% retrieval number defensible.

---

## 4. Data strategy — hybrid, with ground truth (this is the part most projects get wrong)

We need **volume (100K+)**, **realism**, and **labels** for the accuracy eval. No single
source gives all three, so we combine two:

### 4a. Synthetic logs (primary — generated by us)
`scripts/generate_logs.py` emits a full **30-day corpus** with seeded reproducibility:

| Stream | Volume | Format | Content |
|---|---|---|---|
| AWS CloudTrail | **~120,000 events** | JSON Lines | 40+ real event names, multi-account, 6 regions, IAM/S3/EC2/STS/CloudTrail services, full `userIdentity` + `requestParameters` + `responseElements` |
| Linux `auth.log` | **~60,000 lines** | syslog text | sshd Accepted/Failed/Invalid user, sudo, su, cron, session open/close |
| Linux `syslog` | **~20,000 lines** | syslog text | kernel, systemd, service restarts |
| **Total** | **≈200,000 records** | | exceeds the 100K+ claim with headroom |

**Injected attack scenarios (each with ground-truth labels):**

| ID | Scenario | Signature planted in the data |
|---|---|---|
| `S1` | Console brute force | 200+ `ConsoleLogin` failures from one IP, then a success |
| `S2` | SSH brute force | 400+ `Failed password` from one IP, then `Accepted` |
| `S3` | IAM privilege escalation | `CreateUser` → `AttachUserPolicy(AdministratorAccess)` → `CreateAccessKey` |
| `S4` | S3 bucket made public | `PutBucketPolicy` with `Principal: "*"` + `PutBucketAcl public-read` |
| `S5` | Security group opened to world | `AuthorizeSecurityGroupIngress` `0.0.0.0/0` on port 22/3389 |
| `S6` | Logging tampering | `StopLogging` / `DeleteTrail` on the audit trail |
| `S7` | Root account usage | `ConsoleLogin` as `Root` from an unusual ASN/IP |
| `S8` | Credential stuffing | many `AssumeRole`/`GetCallerIdentity` failures across many users |
| `S9` | Impossible travel | same user, two continents, 12 minutes apart |
| `S10` | Data exfiltration pattern | `GetObject` × 3,000 on one bucket in 20 minutes |

Every scenario writes its `event_id`s into `data/eval/labels.json`:

```json
{
  "S1": {"type": "brute_force_console", "seed_events": ["ct-0001234", "..."], "expected_query": "failed console logins from 203.0.113.44", "window_ids": ["w-00042"]},
  "...": {}
}
```

**This is the unlock:** because we *planted* the attacks, we know the correct answer set,
so `Recall@3` and `MRR` are measured against real ground truth — not vibes.

### 4b. Real public logs (secondary — realism + "we ingested production data")
**Loghub** (LogPAI, free for research/academic use, Zenodo-hosted):

| Dataset | Lines | Size | Direct download |
|---|---|---|---|
| **Linux** | 25,567 | 2.25 MiB | `https://zenodo.org/records/8196385/files/Linux.tar.gz?download=1` |
| **OpenSSH** | 655,146 | 70.0 MiB | `https://zenodo.org/records/8196385/files/SSH.tar.gz?download=1` |
| **Apache** | 56,481 | 4.90 MiB | `https://zenodo.org/records/8196385/files/Apache.tar.gz?download=1` |
| Zookeeper | 74,380 | 9.95 MiB | `https://zenodo.org/records/8196385/files/Zookeeper.tar.gz?download=1` |
| Mac | 117,283 | 16.1 MiB | `https://zenodo.org/records/8196385/files/Mac.tar.gz?download=1` |

Source & license: <https://github.com/logpai/loghub> (cite Zhu et al., ISSRE 2023).
`scripts/download_loghub.sh` fetches Linux + OpenSSH (≈72 MB) and unpacks into `data/raw/loghub/`.

> **Note:** Loghub has **no CloudTrail dataset** — that is exactly why the synthetic
> CloudTrail generator exists. We are honest about this in the README: *synthetic
> CloudTrail for labelled scenarios + real Loghub SSH/Linux for realism.* That is a
> defensible research design, not a shortcut.

### 4c. Why not "just download a 100K log file"?
Because an unlabelled download gives you no way to prove retrieval accuracy, and a
pure-synthetic corpus alone looks fake. **Hybrid = volume + realism + ground truth.**

---

## 5. Retrieval & accuracy design (how we legitimately approach 89% Recall@3)

**Target:** ≥ 0.89 Recall@3 on the curated eval set (≈200 questions, labels from §4a).

| Stage | Technique | Expected contribution |
|---|---|---|
| 1. Normalization | typed `Event` schema + structured columns | enables exact filters |
| 2. Query routing | LLM classifier → SQL path vs semantic path | removes a whole class of failures |
| 3. Query rewriting | resolve "last 24h", usernames, ARNs, event names into a `SearchSpec` | big win on time-scoped questions |
| 4. Hybrid retrieval | dense (bge-small) + BM25 over `search_text` → **Reciprocal Rank Fusion** | ~+10-15 pts over dense-only |
| 5. Metadata pre-filter | push `ts_range`, `principal`, `source_ip`, `event_name` into the Chroma `where` clause | removes cross-time false positives |
| 6. Reranking | FlashRank cross-encoder over top-50 → top-k | ~+8-12 pts, biggest single lever |
| 7. Windowing | incident-window chunks instead of 512-token text splices | context that is actually coherent |

**Measurement (`src/slrag/eval/run_eval.py`):** Recall@1/@3/@5, MRR@10, plus
answer-level citation-precision (fraction of cited event_ids that exist in the ground-truth set).
Every eval run writes a timestamped JSON + markdown report to `reports/`.

> **Honesty rule for the resume number:** we report the number the harness prints.
> If it lands at 0.84 we tune (reranker weight, window size, BM25 `k1`/`b`) until it
> reaches or passes target — and if it plateaus lower, the resume says the real number.
> **We never type a metric we did not measure.**

---

## 6. Guardrails & auditability

| Guardrail | Implementation | Test |
|---|---|---|
| Prompt-injection screen | regex + heuristic scoring on retrieved log text before it enters the prompt; log content is wrapped in `<log_data>` and the system prompt forbids following instructions found inside it | `tests/test_guardrails.py::test_injection_phrases_blocked` |
| Poisoned-log detection | flags retrieved chunks containing imperative phrases ("ignore previous", "you are now", "system:") and surfaces a warning instead of silently obeying | `test_poisoned_log_flagged` |
| Citation enforcement | answer must contain `[event:<id>]` / `[window:<id>]` refs; post-check verifies each ref exists in the retrieved set; uncited claims are stripped/flagged | `test_uncited_answer_flagged` |
| PII/secret redaction | mask AWS access keys (`AKIA…`), bearer tokens, passwords in output | `test_secret_redaction` |
| Rate-limit / cost guard | per-request token budget + provider retry with backoff (free tiers are rate-limited) | `test_retry_on_429` |

---

## 7. Phase plan (each phase = code + tests + git push + handoff)

Legend: **DoD** = Definition of Done. Time is a realistic estimate for you on your laptop.

### ✅ Phase 0 — Blueprint + repo skeleton + prerequisites *(≈45 min)*
- **Deliverables:** this blueprint, `docs/HANDOFF.md`, full folder tree, `requirements.txt` (pinned), `.env.example`, `Makefile`, `.gitignore`, `README.md`, CI workflow, Dockerfiles.
- **Tests:** `pytest -q` passes (collection only, no logic yet); `python -c "import slrag"` works.
- **Git:** `git init` → initial commit → push → tag `v0.1.0-phase0`.
- **DoD:** `make bootstrap` succeeds on your laptop; `pytest` green; repo is public on GitHub.

### Phase 1 — Synthetic + real log corpus *(≈60 min)*
- **Deliverables:** `scripts/generate_logs.py` (200K records, 10 labelled scenarios), `scripts/download_loghub.sh`, `data/eval/labels.json`, `docs/DATA.md`.
- **Tests:** `tests/test_generate_logs.py` — determinism (same seed ⇒ same SHA256), record counts, JSON-parseability of every CloudTrail line, scenario presence, label↔event integrity.
- **Git:** branch `phase/1-log-corpus` → PR → merge → tag `v0.2.0-phase1`.
- **DoD:** `python scripts/generate_logs.py --full` finishes in < 3 min; `wc -l` shows ≥ 200,000; every injected scenario is discoverable by grep.

### Phase 2 — Event schema, loaders, incident-window chunker *(≈75 min)*
- **Deliverables:** `ingest/schema.py` (`Event` dataclass + validation), `ingest/loaders.py` (CloudTrail JSONL + syslog regex parsers → `Event`), `ingest/chunker.py` (window grouping by `(principal, src_ip)` with 15-min gap, ~1200-token cap, header+lines serialization, `search_text`), `ingest/store.py` (SQLite table + indexes).
- **Tests:** parser unit tests on hand-written fixtures, malformed-line tolerance (must not crash), window-boundary tests (gap < 15 min merges, > 15 min splits), token-cap splitting, SQLite round-trip, determinism.
- **Git:** branch → PR → tag `v0.3.0-phase2`.
- **DoD:** `python -m slrag.cli ingest --input data/raw --out data/processed` produces `events.sqlite` + `chunks.jsonl`; ≥ 30,000 chunks; ingest of 200K records < 5 min.

### Phase 3 — Embeddings + Chroma + hybrid retrieval + reranking *(≈90 min)*
- **Deliverables:** `providers/embeddings.py` (fastembed ↔ sentence-transformers ↔ OpenAI), `retrieval/vectorstore.py` (Chroma build + `where`-filter pushdown), `retrieval/bm25.py`, `retrieval/hybrid.py` (RRF fusion), `retrieval/rerank.py` (FlashRank), `cli.py index` / `cli.py search`.
- **Tests:** embedding determinism + dimension, Chroma persistence round-trip, RRF correctness on a toy ranking fixture, metadata filter actually narrows results, reranker reorders a known-bad top-1, retrieval smoke on 20 planted scenarios.
- **Git:** branch → PR → tag `v0.4.0-phase3`.
- **DoD:** `python -m slrag.cli search "failed console logins" --top-k 5` returns cited windows in < 3 s; first Recall@3 measurement printed (expect ~0.7 before tuning — this is the baseline).

### Phase 4 — LangChain agent: rewriting, tool calling, cited generation *(≈75 min)*
- **Deliverables:** `agent/tools.py` (`search_logs`, `count_events`, `list_events`, `get_event`, `timeline`), `agent/pipeline.py` (`create_agent` + `system_prompt` + middleware retry), `agent/prompts.py` (rewrite / route / answer / citation-enforcement prompts).
- **Tests:** each tool in isolation against SQLite/Chroma; router picks SQL for aggregates and RAG for hunts (table-driven cases); end-to-end "show all failed logins in the last 24h" returns the exact planted count; citation refs all resolve.
- **Git:** branch → PR → tag `v0.5.0-phase4`.
- **DoD:** CLI `ask` answers 10 golden questions correctly with citations; 429s retried, not crashed.

### Phase 5 — Guardrails + eval harness + accuracy tuning *(≈60 min)*
- **Deliverables:** `guardrails/injection.py`, `guardrails/redaction.py`, `guardrails/citations.py`, `eval/build_eval_set.py` (≈200 Q/A from labels), `eval/run_eval.py`, `reports/`.
- **Tests:** the five guardrail tests from §6, eval-harness unit tests (metric math against a hand-computed fixture — verify Recall@k and MRR formulas exactly), determinism of eval set.
- **Git:** branch → PR → tag `v0.6.0-phase5`.
- **DoD:** `python -m slrag.eval.run_eval` writes `reports/eval-<ts>.md` with the measured number; we tune until **Recall@3 ≥ 0.89** or document the ceiling with evidence.

### Phase 6 — FastAPI service *(≈60 min)*
- **Deliverables:** `api/main.py` (`/health`, `/ingest`, `/ask`, `/search`, `/events/{id}`, `/metrics`), `api/schemas.py` (pydantic request/response), structured JSON logs, error taxonomy, `X-Request-ID`.
- **Tests:** `TestClient` smoke tests for every route, request-validation 422s, `/ask` with a mocked LLM (so CI needs no API key), latency assertion, OpenAPI schema snapshot.
- **Git:** branch → PR → tag `v0.7.0-phase6`.
- **DoD:** `curl localhost:8000/ask -d '{"question":"..."}'` returns a cited answer; Swagger UI loads.

### Phase 7 — Streamlit UI *(≈45 min)*
- **Deliverables:** `ui/app.py` — chat over `/ask`, retrieved-chunk inspector with raw-log expander, filter sidebar (time range / principal / IP / event name), scenario-detection dashboard, citation click-through.
- **Tests:** pure-logic helpers unit-tested; UI launched manually for a scripted click-through checklist (documented in `docs/TESTING.md`).
- **Git:** branch → PR → tag `v0.8.0-phase7`.
- **DoD:** you can demo it end-to-end in a browser and screenshot it for the resume.

### Phase 8 — Docker, CI, docs, polish *(≈45 min)*
- **Deliverables:** multi-stage `Dockerfile` (API) + `ui` service, `docker-compose.yml` (api + ui + volumes), GitHub Actions CI (ruff + pytest on 3.11/3.12, no network keys), `docs/ARCHITECTURE.md`, `docs/TESTING.md`, `docs/RESULTS.md` (real eval table), polished `README.md` with architecture diagram + measured results.
- **Tests:** `docker compose up` → `/health` 200; full test suite green in CI; cold-clone reproducibility (`git clone` → `make bootstrap` → `make test`).
- **Git:** PR → merge → tag `v1.0.0`.
- **DoD:** a stranger can clone, bootstrap, and run the demo from the README alone.

**Total:** ≈ 7 hours of focused work. **Minimum viable path if time runs short:** Phases 0-6
(≈5h 45m) — that already delivers ingest → retrieval → agent → API with measured accuracy;
Phases 7-8 are presentation layer and can slip.

---

## 8. Testing strategy

| Level | Scope | Tool | Runs in CI? |
|---|---|---|---|
| Unit | parsers, chunker, metrics math, guardrails, tools | `pytest` | yes |
| Integration | ingest → Chroma → retrieve on a 2K-record fixture | `pytest -m integration` | yes (local embeddings, no API) |
| Contract | every FastAPI route + schema | `pytest` + `TestClient` | yes (LLM mocked) |
| Eval | Recall@1/3/5, MRR, citation precision on ≈200 questions | `slrag.eval.run_eval` | no (needs an LLM key; run locally) |
| Manual | Streamlit click-through, Docker compose up | checklist in `docs/TESTING.md` | no |

**Rules we follow:** every test is deterministic (fixed seeds, frozen clocks via an injectable
`now()`); no test requires a paid key; `pytest -m "not integration"` must pass in < 60 s.

---

## 9. Git workflow (so the history looks like engineering, not a dump)

```
main ──●───●───────────●───────────●─── ... ──● (tag v1.0.0)
        \   \         /           /
         \   ●─●─●───●           /     phase/2-... (PR, squash-merge)
          ●─●─●───●─────────────●      phase/1-log-corpus (PR, squash-merge)
          ▲ initial commit + v0.1.0-phase0
```

- One branch per phase: `phase/<n>-<slug>` → PR → **squash-merge** → tag `v0.<n>.0-phase<n>`.
- Conventional commits: `feat(ingest): add incident-window chunker`, `test(eval): assert Recall@3 math`, `docs(handoff): close phase 2`.
- `make git-phase PHASE=2 SLUG=schema-chunker MSG="feat(ingest): ..."` automates branch → add → commit → push → PR hint.
- Never commit `.env`, `data/raw/**`, `data/processed/**`, `chroma/`, `*.sqlite`, `models/`.
- Each PR description is copied from the phase's handoff section — that is what makes the repo readable to a reviewer.

---

## 10. Handoff protocol (constraint #5)

`docs/HANDOFF.md` is **rewritten at the end of every phase** and always contains:

1. **Project identity** — what we're building, in 3 lines, so a cold agent has context.
2. **Locked decisions** — stack, models, ports, paths, naming. Never re-litigated.
3. **Phase status table** — done / in-progress / not-started, with tags.
4. **Exact current state** — files that exist, what each does, last green test command + output.
5. **Next 5 concrete actions** — copy-pasteable commands, in order.
6. **Blockers & open questions** — with what was already tried.
7. **Verification commands** — how to prove the last phase still works.
8. **Env/keys status** — which keys are configured (names only, never values).

**Resume prompt** (what you paste into a fresh chat if the context dies):
> *"Read `docs/HANDOFF.md` and `docs/BLUEPRINT.md` in this repo, then continue from the 'Next actions' section. Do not re-plan completed phases."*

---

## 11. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Free-tier rate limits during eval (200 questions × several calls) | **high** | cache every LLM response to `data/cache/`; batch/parallel-limit to 1; Groq primary + Gemini fallback rotation; eval LLM calls capped by `--limit` |
| Embedding 40K chunks is slow on a laptop CPU | medium | fastembed ONNX (no torch); `--limit` fast path; progress bar + resumable index (skip existing ids) |
| LangChain 1.x API differs from older tutorials you may have read | medium | all model/agent code is isolated in `providers/` + `agent/pipeline.py`, written against the **1.x `create_agent`** API and pinned; nothing else imports LangChain |
| Loghub download blocked / slow | low | generator already gives 200K records; Loghub is optional (`--with-loghub` flag) |
| Time overrun | medium | Phases 7-8 are explicitly droppable; MVP path defined in §7 |
| Resume metric not reached | low | tuning ladder in §5 + honest reporting rule |

---

## 12. Your prerequisites checklist (do these while reviewing)

1. **Python 3.11 or 3.12** — `python --version` (3.13 also fine; 3.10 not recommended).
2. **Git + a GitHub account** with a repo created (empty, public): `slrag` or `security-log-rag`.
3. **Free API keys (both, ~3 min total, no credit card):**
   - Groq → <https://console.groq.com/keys>
   - Google AI Studio → <https://aistudio.google.com/app/apikey>
4. **Disk:** ~3 GB free (models ≈ 200 MB, corpus ≈ 500 MB, Chroma ≈ 1 GB).
5. **Optional, for the offline demo:** [Ollama](https://ollama.com) + `ollama pull llama3.1:8b`.
6. **Optional:** Docker Desktop, if you want the Phase 8 container demo.
7. Put the two keys in `.env` (never in git): `GROQ_API_KEY=...`, `GOOGLE_API_KEY=...`

---

## 13. Resume alignment — what to update and why

Your three bullets map onto this build as follows:

| Resume claim | What we actually build | Action |
|---|---|---|
| "ingests AWS CloudTrail / system logs, chunks into semantic embeddings using `text-embedding-3-small`, stores in Chroma, 100K+ log entries" | CloudTrail + syslog ingest, incident-window chunks, 384-d local embeddings, Chroma, **≈200K records** | **Update the model name** to `bge-small-en-v1.5 (local, free)` — or keep `text-embedding-3-small` only if you actually run it. Volume claim is safe (we exceed it). |
| "LangChain agent pipeline (rewrite → hybrid retrieval → rerank → generate) with tool calling; 89% top-3 accuracy" | Exactly this pipeline, with a measured Recall@3 from the eval harness | **Fill in the measured number.** If it's 0.91, say 91%. |
| "FastAPI + Streamlit + Docker; prompt-injection guardrails and citation-backed answers" | Exactly this, with the five guardrails in §6 | No change needed. |

The one thing to avoid is a resume line that names a model you never ran. Swapping
`text-embedding-3-small` → `bge-small-en-v1.5` costs you nothing and survives an
interview question about cost; keeping it and being asked "what was your embedding
cost per month?" does not.

---

## 14. What happens right now

Phase 0 is **done and shipped with this message**: the folder tree, pinned
requirements, config layer, provider factory, Makefile, CI workflow, Docker files, the
synthetic log generator (Phase 1 preview, already tested), and the handoff doc.

**Your next three commands** (full detail in `docs/HANDOFF.md`):

```bash
cd slrag
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
make bootstrap && make test
```

Then tell me the output and we start Phase 1 proper — generating the 200K corpus and
pushing it.
