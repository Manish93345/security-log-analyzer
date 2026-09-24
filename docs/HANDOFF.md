# HANDOFF — slrag (RAG-Powered Security Log Analyzer)

**Living state document.** Rewritten at the end of every phase. If a fresh session picks this
project up, read `docs/BLUEPRINT.md` (the plan) and this file (the state) — nothing else is
required to continue.

| | |
|---|---|
| **Current phase** | Phase 2 complete — SQLite store + CLI (Phase 3 next) |
| **Repo** | `https://github.com/Manish93345/security-log-analyzer` |
| **Last tag** | `v0.1.0-phase0` (Phase 2 tag: `v0.2.0-phase2`) |
| **Python** | 3.13.7 on Windows (project supports 3.11–3.13) |
| **Platform** | Windows 11 / PowerShell (primary), macOS / Linux (secondary) |

---

## 1. What works right now (verified)

```
python scripts/tasks.py doctor     # all green
python scripts/tasks.py test       # 87 passed
python scripts/tasks.py corpus     # TOTAL=224,275 records
python scripts/tasks.py ingest     # 224,275 events -> 38,767 windows, 19.3s
python scripts/tasks.py stats      # corpus overview
```

Measured on the full generated corpus (not estimates):

| Metric | Value |
|---|---|
| Log files parsed | 32 |
| Records read | 224,275 |
| Events kept / skipped | 224,275 / **0** |
| By source | `cloudtrail=123,414` · `auth=80,851` · `syslog=20,010` |
| Incident windows (chunks) | **38,767** |
| Window size | median 2 / p90 19 / max 64 events |
| High-risk windows (risk ≥ 50) | 333 (max risk 100) |
| Failures | 27,676 (12.3% of events) |
| Corpus time span | 2026-08-25T00:00:45Z → 2026-09-24T00:00:23Z |
| Ingest wall time | 19.3s end-to-end |
| Peak RSS during ingest | ~450 MB |
| Unit tests | 87 passing, ~2s |

---

## 2. Pipeline (Phase 2 data flow)

```
data/raw/**.json  data/raw/**/*.log
        │
        ├─ loaders.py    discover_files() -> iter_cloudtrail_file() / iter_syslog_file()
        │                tolerant: unparseable lines counted, never fatal
        ▼
   Event  (schema.py)   14 flat fields, deterministic ids, raw line kept for citation
        │
        ├─ store.py      events table  (exact/aggregate queries)
        │                chunks + chunk_events tables (windows + membership)
        ├─ chunker.py    build_windows(): bucket by (source, principal, src_ip),
        │                merge while gap <= 15 min, cap 120 events / 4800 chars
        ▼
data/processed/events.sqlite        data/processed/chunks.jsonl
   (dual retrieval: exact)             (hand-off to Phase 3 vector index)
```

**Why two stores.** Exact questions ("how many failed logins from 203.0.113.7 on Tuesday")
must return a count, not a similar-looking neighbour. SQLite answers those; Chroma (Phase 3)
answers semantic hunts. Both are written in one pass so they cannot drift apart.

**Idempotency.** Every write is `INSERT OR REPLACE` on a deterministic id, so re-running
`ingest` leaves row counts unchanged (covered by `test_run_ingest_is_idempotent`).

---

## 3. Repository map (Phase 2 files marked •)

```
security-log-analyzer/
├── .github/workflows/ci.yml        ruff + pytest on Python 3.11 & 3.12
├── .gitattributes  .gitignore      line endings; ignores + sample-corpus exception
├── Makefile                        macOS/Linux (Windows: use scripts/tasks.py)
├── README.md  pyproject.toml  requirements.txt  requirements-dev.txt
├── config/default.yaml             every tunable
├── docker/                         Dockerfile, Dockerfile.ui, docker-compose.yml
├── docs/  BLUEPRINT.md  DATA.md  SETUP-WINDOWS.md  HANDOFF.md (this file)
├── scripts/
│   ├── generate_logs.py            labelled synthetic corpus
│   ├── download_loghub.sh  git_phase.sh
│   └── tasks.py                    • ingest / stats tasks added
├── src/slrag/
│   ├── cli.py                      • argparse CLI: ingest | stats | version
│   ├── __main__.py                 • `python -m slrag`
│   ├── config.py  logging_setup.py
│   ├── providers/                  llm.py, embeddings.py (free providers)
│   └── ingest/
│       ├── schema.py  loaders.py  chunker.py
│       ├── store.py                • SQLite store (events/chunks/chunk_events/meta)
│       └── pipeline.py             • run_ingest() orchestration + IngestResult
├── tests/  conftest.py  test_generate_logs.py  test_loaders.py  test_chunker.py
│        • test_store.py  • test_pipeline.py
└── data/ (gitignored)  raw/  processed/{events.sqlite,chunks.jsonl}  eval/labels.json
```

---

## 4. Commands

| Task | Windows (PowerShell) | macOS / Linux |
|---|---|---|
| Environment check | `python scripts/tasks.py doctor` | `make doctor` |
| Unit tests | `python scripts/tasks.py test` | `make test` |
| Generate corpus | `python scripts/tasks.py corpus` | `make corpus` |
| Ingest | `python scripts/tasks.py ingest` | `make ingest` |
| Ingest, fresh | `python scripts/tasks.py ingest --rebuild` | `make ingest` |
| Statistics | `python scripts/tasks.py stats` | `make stats` |
| Phase commit + push | `python scripts/tasks.py git-phase --phase 2 --slug store --message "..."` | `make git-phase` |

Direct CLI (works either way):

```powershell
python -m slrag ingest --raw data/raw --rebuild
python -m slrag stats
python -m slrag stats --json          # machine-readable
python -m slrag version
```

> After pulling this phase, re-run `python scripts/tasks.py bootstrap` once to register the
> `slrag` console script (`[project.scripts]` was added to `pyproject.toml`). `python -m slrag`
> works without it.

---

## 5. Phase plan

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, config, provider layer, CI, docs | ✅ complete (`v0.1.0-phase0`) |
| 1 | Corpus generator + loaders + incident-window chunker | ✅ complete |
| 2 | SQLite store, ingest pipeline, CLI, tests | ✅ complete |
| 3 | Embeddings + Chroma index + BM25 hybrid retrieval + RRF + FlashRank rerank | ⏭ next |
| 4 | LangChain agent + tool routing (SQLite vs vector) + guardrails | pending |
| 5 | Eval harness: ~200 labelled questions, Recall@1/3/5, MRR | pending |
| 6 | FastAPI service (`:8000`) | pending |
| 7 | Streamlit analyst UI (`:8501`) | pending |
| 8 | Docker, README polish, final handoff | pending |

---

## 6. Decisions locked in

| Decision | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` via `fastembed` (384-d, ONNX) | free, local, **no torch** (avoids a 2.5 GB install on Windows) |
| Reranker | `ms-marco-MiniLM-L-6-v2` via `flashrank` | ONNX, ~40 MB, no API cost |
| LLM | Groq `openai/gpt-oss-120b` → Gemini `gemini-3.5-flash` → Ollama | free tiers, no card; OpenAI-compatible layer makes switching one `.env` line |
| Structured store | stdlib `sqlite3` | no dependency, portable single file |
| Chunking | incident windows, not fixed-size slices | a fixed slice cuts an attack chain in half and retrieves nothing useful |
| Window identity | `(source, principal, src_ip)` bucket **first**, then gap-merge | merging over a global timeline fragments sessions when traffic interleaves |
| Chunk ids | `w-00001…` chronological; CloudTrail `ct-<eventID>`; syslog `sl-<sha1[:12]>` | deterministic → idempotent re-ingest |

**Resume note.** The résumé line about `text-embedding-3-small` (OpenAI, paid) does not match
the implementation — the project uses a free local embedder. Either update the résumé or set
`EMBEDDING_BACKEND=openai` in `.env` and accept the cost. Report the Recall@3 that the Phase 5
harness actually prints; never a hand-written number.

---

## 7. Issues found and fixed this phase

1. **`Chunk.from_dict()` crash on read-back** — `event_ids` is not a `chunks` column (membership
   is normalized into `chunk_events`), so `_row_to_chunk()` rebuilt a `Chunk` without it and
   raised `TypeError`. Fixed by rehydrating `event_ids` from `chunk_events`, with a
   `load_members=False` fast path for bulk streaming (avoids N+1 over 38k windows).
2. **`slrag stats` ranked-list formatting** — the label column was repeated on every row instead
   of only the first. Fixed.
3. **`.github/workflows/ci.yml` missing locally** — it *is* inside the scaffold zip
   (`slrag/.github/workflows/ci.yml`) but did not survive extraction, so CI never ran. Re-shipped
   in the Phase 2 bundle.
4. **`*.log` in `.gitignore` silently dropped `slrag-sample-corpus/syslog/auth.log`** — added
   `!tests/fixtures/**` and `!slrag-sample-corpus/**` negations so fixture data is committable.

---

## 8. Next phase (3) — embeddings, index, hybrid retrieval

Files to create:

- `src/slrag/retrieval/embeddings.py` — fastembed wrapper, batched, cached
- `src/slrag/retrieval/vector_store.py` — Chroma collection over `chunks.jsonl` (`text` embedded,
  `search_text` for BM25, metadata: `chunk_id`, `ts_start`, `risk_score`, `source`, `principal`)
- `src/slrag/retrieval/bm25.py` — rank-bm25 index over `search_text`
- `src/slrag/retrieval/fusion.py` — Reciprocal Rank Fusion (`k=60`)
- `src/slrag/retrieval/rerank.py` — FlashRank top-50 → top-5
- `src/slrag/retrieval/pipeline.py` — `retrieve(question, k=5)` returning ranked windows + citations
- CLI: `slrag index`, `slrag search "<query>"`
- Tests: fusion maths, rerank ordering, index round-trip (marked `integration`)

Acceptance for Phase 3: `slrag index` builds `data/chroma` from 38,767 windows; `slrag search
"failed sudo commands on the web server"` returns a window whose `text` visibly contains that
activity; fusion + rerank unit-tested without network.

---

## 9. Resume prompt (paste into a fresh session)

> The project is `slrag`, a RAG-powered security log analyzer, at
> `https://github.com/Manish93345/security-log-analyzer`. Read `docs/BLUEPRINT.md` and
> `docs/HANDOFF.md` first. Phases 0–2 are complete and verified: 224,275 events and 38,767
> incident windows in `data/processed/events.sqlite` + `chunks.jsonl`, 87 unit tests green.
> Continue with **Phase 3** exactly as scoped in HANDOFF section 8 — free APIs only, testing on
> Windows via `python scripts/tasks.py`, handoff doc rewritten at the end of the phase, and a
> proper GitHub push with `git-phase`.
