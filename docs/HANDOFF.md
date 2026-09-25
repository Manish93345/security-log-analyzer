# HANDOFF — slrag (RAG-Powered Security Log Analyzer)

**Living state document.** Rewritten at the end of every phase. If a fresh session picks this
project up, read `docs/BLUEPRINT.md` (the plan) and this file (the state) — nothing else is
required to continue.

| | |
|---|---|
| **Current phase** | Phase 3 complete — hybrid retrieval (Phase 4 next) |
| **Repo** | `https://github.com/Manish93345/security-log-analyzer` |
| **Last tag** | `v0.3.0-phase3` |
| **Python** | 3.13.7 on Windows (project supports 3.11–3.13) |
| **Platform** | Windows 11 / PowerShell (primary), macOS / Linux (secondary) |

---

## 1. What works right now (verified)

```
python scripts/tasks.py doctor     # all green
python scripts/tasks.py test       # 158 passed
python scripts/tasks.py corpus     # TOTAL=224,275 records
python scripts/tasks.py ingest     # 224,275 events -> 38,767 windows, ~20s
python scripts/tasks.py stats      # corpus overview
python scripts/tasks.py index      # windows -> data/chroma      <-- see the caveat below
python scripts/tasks.py search "failed sudo commands on the web server"
```

### Phase 2 numbers — full generated corpus (measured)

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

### Phase 3 numbers — measured on a 2-day smoke corpus (5,320 records → 554 windows)

| Metric | Value |
|---|---|
| `slrag index` | 554 windows in 140.5 s → **3.9 windows/s**, 384-d vectors |
| `index_meta.json` | written by `index`; read back by `slrag index --check` |
| `slrag search` (rerank off) | **904 ms**, top hit found by `dense#8` *and* `bm25#4` |
| `slrag search` (fastembed rerank) | **613 ms**, cross-encoder scores 0.6374 / 0.5377 / 0.4999 |
| Tests | **158 passed** (incl. a real cross-encoder rerank over a downloaded model) |
| `ruff check src tests scripts` | **clean** |

> **⚠️ Two things are NOT yet measured — do not put them on the résumé yet.**
> 1. **The full 38,767-window index has never been built.** Only a 554-window smoke index exists.
> 2. **Recall@3 does not exist yet** — the eval harness is Phase 5.
>
> **Measure indexing speed before committing to a full run.** 3.9 windows/s was measured inside
> a 2-vCPU / 2 GB container; a multi-core laptop will be substantially faster, but the honest
> move is to time it:
>
> ```powershell
> python scripts/tasks.py index --limit 500     # read the "windows/s" it prints
> ```
>
> At the container's 3.9/s a full run would take ~2.7 hours. If your laptop prints anything like
> that, do not run it in the foreground — see section 4 for the low-memory option.

---

## 2. Pipeline (Phase 3 data flow)

```
data/raw/**.json  data/raw/**/*.log
        │
        ├─ loaders.py    discover_files() -> iter_cloudtrail_file() / iter_syslog_file()
        ▼
   Event  (schema.py)   14 flat fields, deterministic ids, raw line kept for citation
        │
        ├─ chunker.py    build_windows(): bucket by (source, principal, src_ip),
        │                merge while gap <= 15 min, cap 120 events / 4800 chars
        ▼
data/processed/events.sqlite                data/processed/chunks.jsonl
   events / chunks / chunk_events / meta        (one window per line)
        │                                              │
        │  exact + aggregate queries                   │  slrag index
        │  (Phase 4 tool)                              ▼
        │                                    data/chroma  (Chroma, cosine, 384-d)
        │                                    data/chroma/index_meta.json
        │                                              │
        └──────────────────┬───────────────────────────┘
                           ▼
                  slrag search "<question>"
                           │
        ┌──────────────────┴───────────────────┐
        │ dense:  Chroma over window `text`    │  -> top 50
        │ lexical: BM25 over `search_text`     │  -> top 50
        └──────────────────┬───────────────────┘
                           ▼
              RRF fusion (k=60, rank-based)
                           ▼
        rehydrate window text from SQLite  (ONE query, not 50)
                           ▼
        fastembed cross-encoder over top-50  -> final top-5
```

**Why hybrid.** Embeddings answer "what is this window *about*"; BM25 answers "does this window
literally contain `203.0.113.44`". Security questions need both, and a window one retriever
misses can still be rescued by the other.

**Why RRF and not a weighted score blend.** BM25 scores are unbounded and cosine similarity lives
in [-1, 1]; any score-level blend needs per-corpus calibration that breaks the moment the corpus
changes. RRF uses *ranks*, which are comparable by construction.

**Why windows come back from SQLite, not Chroma.** The vector store holds only ids, metadata and
vectors — the ~190 MB of window text stays in one place, so the two views cannot drift and
citations stay one join away.

**Why incident windows instead of a text splitter.** A 512-token slice can cut an attack chain in
half; similarity search over half a chain retrieves nothing useful.

---

## 3. Repository map (Phase 3 files marked •)

```
security-log-analyzer/
├── .github/workflows/ci.yml        ruff + pytest on Python 3.11 & 3.12
├── .gitattributes  .gitignore      line endings; generated corpus is ignored
├── Makefile                        macOS/Linux (Windows: use scripts/tasks.py)
├── README.md  pyproject.toml  requirements.txt  requirements-dev.txt
├── config/default.yaml             every tunable
├── docker/                         Dockerfile, Dockerfile.ui, docker-compose.yml
├── docs/  BLUEPRINT.md  DATA.md  SETUP-WINDOWS.md  HANDOFF.md (this file)
├── scripts/
│   ├── generate_logs.py            labelled synthetic corpus
│   ├── download_loghub.sh  git_phase.sh
│   └── tasks.py                    • index / search tasks added
├── src/slrag/
│   ├── cli.py                      • argparse CLI: ingest | stats | index | search | version
│   ├── __main__.py                 `python -m slrag`
│   ├── config.py                   • retrieval knobs (bm25, rrf, rerank_backend, batch sizes)
│   ├── logging_setup.py
│   ├── providers/                  llm.py, embeddings.py (free providers)
│   ├── ingest/
│   │   ├── schema.py  loaders.py  chunker.py
│   │   ├── store.py                • SQLite store + chunks_by_ids()
│   │   └── pipeline.py             run_ingest() orchestration + IngestResult
│   └── retrieval/                  • THE PHASE 3 LAYER
│       ├── bm25.py                 • log-aware tokeniser + rank-bm25 index
│       ├── vector_store.py         • Chroma wrapper + index_meta.json
│       ├── indexer.py              • build_index() over chunks.jsonl
│       ├── fusion.py               • Reciprocal Rank Fusion (k=60)
│       ├── rerank.py               • cross-encoder (fastembed default, flashrank optional)
│       └── pipeline.py             • RetrievalEngine.retrieve() -> ranked windows + citations
├── tests/  conftest.py  test_generate_logs.py  test_loaders.py  test_chunker.py
│   test_store.py  test_pipeline.py
│        • test_fusion.py  • test_bm25.py  • test_retrieval.py  • test_rerank.py
└── data/ (gitignored)  raw/  processed/{events.sqlite,chunks.jsonl}  chroma/  eval/labels.json
```

---

## 4. Commands

| Task | Windows (PowerShell) | macOS / Linux |
|---|---|---|
| Environment check | `python scripts/tasks.py doctor` | `make doctor` |
| Unit tests | `python scripts/tasks.py test` | `make test` |
| All tests (incl. models) | `python scripts/tasks.py test-all` | `make test-all` |
| Generate corpus | `python scripts/tasks.py corpus` | `make corpus` |
| Ingest | `python scripts/tasks.py ingest` | `make ingest` |
| Statistics | `python scripts/tasks.py stats` | `make stats` |
| **Build the vector index** | `python scripts/tasks.py index` | `make index` |
| **Index health check** | `python scripts/tasks.py index --check` | — |
| **Rebuild the index** | `python scripts/tasks.py index --reset` | — |
| **Search** | `python scripts/tasks.py search "failed sudo commands"` | `make search` |
| Phase commit + push | `python scripts/tasks.py git-phase --phase 3 --slug retrieval --message "..."` | `make git-phase` |

Direct CLI (works either way):

```powershell
python -m slrag ingest --raw data/raw --rebuild
python -m slrag stats
python -m slrag index                       # full run; --limit N for a smoke test
python -m slrag index --check               # model, dimension, window count
python -m slrag search "failed sudo commands on the web server" -k 3
python -m slrag search "failed sudo commands" --show-events 3    # raw records behind hit #1
python -m slrag search "..." --json         # machine-readable
python -m slrag search "..." --no-rerank    # skip the cross-encoder
python -m slrag version
```

**Low-memory / slow-machine options.** If the full index is too slow or the machine is tight on
RAM, set these in `.env` before running:

```dotenv
INDEX_BATCH_SIZE=64          # smaller batches = flatter memory while indexing
EMBEDDING_BATCH_SIZE=32
RETRIEVE_TOP_K=5             # windows returned
CANDIDATE_POOL=50            # per-retriever depth before fusion (lower it to cut query RAM)
RERANK_TOP_N=50              # cross-encoder depth; lower it to cut query RAM
RERANK_ENABLED=false         # temporarily drop the cross-encoder entirely
```

Run the full index in a second terminal (or `Start-Process`) rather than blocking your shell.

---

## 5. Phase plan

| Phase | Scope | Status |
|---|---|---|
| 0 | Scaffold, config, provider layer, CI, docs | ✅ complete (`v0.1.0-phase0`) |
| 1 | Corpus generator + loaders + incident-window chunker | ✅ complete |
| 2 | SQLite store, ingest pipeline, CLI, tests | ✅ complete (`v0.2.0-phase2`) |
| 3 | Chroma index + BM25 + RRF fusion + cross-encoder rerank | ✅ complete (`v0.3.0-phase3`) |
| 4 | LangChain agent + tool routing (SQLite vs vector) + guardrails | ⏭ next |
| 5 | Eval harness: ~200 labelled questions, Recall@1/3/5, MRR | pending |
| 6 | FastAPI service (`:8000`) | pending |
| 7 | Streamlit analyst UI (`:8501`) | pending |
| 8 | Docker, README polish, final handoff | pending |

---

## 6. Decisions locked in

| Decision | Choice | Why |
|---|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` via `fastembed` (384-d, ONNX) | free, local, **no torch** (avoids a 2.5 GB install on Windows) |
| Reranker | **`Xenova/ms-marco-MiniLM-L-6-v2` via `fastembed`** (`RERANK_BACKEND=fastembed`) | same MiniLM-L6 cross-encoder, but through a dependency we already have and a download path that provably works — see issue 2 below |
| Reranker (optional) | FlashRank `ms-marco-TinyBERT-L-2-v2` | kept as a second backend; its historical default model cannot be downloaded |
| Lexical | `rank-bm25` over `search_text`, log-aware tokeniser | rare literal tokens (IPs, action names, paths) that embeddings blur away |
| Fusion | RRF, `k=60`, rank-based | no per-corpus weight calibration; a window found by both retrievers outranks one found by either |
| Dense index | Chroma, `hnsw:space=cosine`, persistent | vectors only — text stays in SQLite |
| LLM | Groq `openai/gpt-oss-120b` → Gemini `gemini-3.5-flash` → Ollama | free tiers, no card; OpenAI-compatible layer makes switching one `.env` line |
| Structured store | stdlib `sqlite3` | no dependency, portable single file |
| Chunking | incident windows, not fixed-size slices | a fixed slice cuts an attack chain in half and retrieves nothing useful |
| Window identity | `(source, principal, src_ip)` bucket **first**, then gap-merge | merging over a global timeline fragments sessions when traffic interleaves |
| Chunk ids | `w-00001…` chronological; CloudTrail `ct-<eventID>`; syslog `sl-<sha1[:12]>` | deterministic → idempotent re-ingest |

**Resume note.** The résumé line about `text-embedding-3-small` (OpenAI, paid) does not match the
implementation — the project uses a free local embedder. Either update the résumé or set
`EMBEDDING_BACKEND=openai` in `.env` and accept the cost. Report the Recall@3 that the Phase 5
harness actually prints; never a hand-written number.

---

## 7. Issues found and fixed this phase

1. **`ruff --fix --unsafe-fixes` silently broke every row read (10 tests).** SIM118 rewrote
   `for key in row.keys()` to `for key in row`, but **`sqlite3.Row` iterates its *values*, not its
   column names** — so the `if key in Event.__dataclass_fields__` guard rejected every column and
   `Event`/`Chunk` came back empty
   (`TypeError: Chunk.__init__() missing 11 required positional arguments`). Restored `.keys()`
   with `# noqa: SIM118` and an explanatory comment. **Lesson: never run `--unsafe-fixes`
   unattended on this repo.**
2. **FlashRank's default model URL is a dead 404 upstream.** `flashrank==0.2.10` hardcodes
   `https://huggingface.co/prithivida/flashrank/resolve/main/<model>.zip`, and
   `ms-marco-MiniLM-L-6-v2.zip` returns **404** (verified 2026-09-24) while
   `ms-marco-TinyBERT-L-2-v2.zip` returns **200**. Reranking now defaults to fastembed's
   `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder — the *same* model, a dependency we already
   have, and a download path that demonstrably works. FlashRank is kept as an optional backend
   and moved out of `requirements.txt` / `doctor` so a fresh bootstrap cannot fail on it.
3. **BM25's "score > 0" relevance gate was wrong.** On a small corpus every IDF can come out
   negative, so a *perfect* match scored below zero and was discarded (5 tests caught it).
   Candidates are now selected by "the window literally contains at least one query term", with
   the score used only for ordering — which is all RRF consumes anyway.
4. **`RetrievedWindow.found_by` was order-dependent.** It reported `('dense', 'bm25')` or
   `('bm25', 'dense')` depending on dict insertion order; now sorted for deterministic output.
5. **Secret-scanning alerts + committed sample corpus.** The generated sample corpus was tracked
   and GitHub's secret scanner raised 20 alerts on synthetic `ASIA…` keys from
   `generate_logs.py`. The keys are fake (65 distinct values in one file; nothing to rotate), but
   the corpus is now gitignored (`slrag-sample-corpus/`, `tests/fixtures/sample-corpus/`) and was
   purged from history with `git filter-repo`. Note: GitHub does **not** auto-close secret-scanning
   alerts when the secret leaves history — they must be closed by hand.

---

## 8. Next phase (4) — agent, tool routing, guardrails

Files to create:

- `src/slrag/rag/tools.py` — the tool surface the agent may call:
  - `search_windows(question, k)` → hybrid retrieval from Phase 3 (semantic hunts)
  - `count_events(filters)` / `top_principals()` / `top_src_ips()` → SQLite exact + aggregate
  - `events_in_window(chunk_id)` → the raw records behind a hit (citations)
  - `top_risk_windows(n)` → highest-risk windows straight from SQLite
- `src/slrag/rag/agent.py` — LangChain 1.x `create_agent` with those tools + a system prompt that
  **requires** every claim to carry a `[WINDOW w-xxxxx]` citation
- `src/slrag/rag/guardrails.py` — redact `AKIA[0-9A-Z]{16}`, `ASIA[0-9A-Z]{16}`, bearer tokens and
  `password=…` before anything reaches the LLM; enforce `MAX_PROMPT_TOKENS`
- `src/slrag/rag/prompts.py` — system prompt + answer schema
- CLI: `slrag ask "<question>"`
- Tests: tool routing with a fake LLM; `llm`-marked tests for the real provider

Acceptance for Phase 4:

1. `slrag ask "how many failed logins came from 203.0.113.7?"` returns an **exact count** read
   from SQLite — not a similarity-based guess.
2. `slrag ask "show me suspicious privilege escalation"` returns windows **with citations**.
3. The redactor is unit-tested: no `AKIA…`/`ASIA…` string ever appears in an outbound prompt.

---

## 9. Resume prompt (paste into a fresh session)

> The project is `slrag`, a RAG-powered security log analyzer, at
> `https://github.com/Manish93345/security-log-analyzer`. Read `docs/BLUEPRINT.md` and
> `docs/HANDOFF.md` first. Phases 0–3 are complete and verified: 224,275 events and 38,767
> incident windows in `data/processed/events.sqlite` + `chunks.jsonl`; hybrid retrieval
> (Chroma dense + BM25 → RRF k=60 → fastembed cross-encoder) in `src/slrag/retrieval/`;
> 158 tests green; `ruff check` clean. Continue with **Phase 4** exactly as scoped in HANDOFF
> section 8 — free APIs only, testing on Windows via `python scripts/tasks.py`, handoff doc
> rewritten at the end of the phase, and a proper GitHub push with `git-phase`. Two things are
> deliberately unmeasured and must stay off the résumé until they are: the full 38,767-window
> index, and Recall@3 (Phase 5).
