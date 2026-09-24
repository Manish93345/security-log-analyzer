# slrag — RAG-Powered Security Log Analyzer

Natural-language query over AWS CloudTrail and Linux system logs, with **citation-backed
answers** and a **measured** retrieval accuracy. Built with LangChain, Chroma, FastAPI and
Streamlit — running on **free / local models only** (no paid API required).

```
 "show all failed logins in the last 24h"
        │
        ├─ guardrail: prompt-injection screen on retrieved log text
        ├─ rewrite  : resolve time range, principal, IP into a SearchSpec
        ├─ route    : exact/aggregate → SQLite   |   semantic hunt → hybrid retrieval
        ├─ retrieve : dense (bge-small) + BM25  →  RRF fusion  →  FlashRank rerank
        └─ generate : answer with [event:<id>] / [window:<id>] citations
```

---

## Why this design

Most "RAG over logs" projects embed raw log lines and hope similarity search answers
*"how many failed logins in the last 24h?"* — it cannot. Counting and filtering are exact
operations, so this project routes them to **SQL**, and reserves vector search for what it is
actually good at: *semantic hunts* over incident context. Logs are chunked into
**incident windows** (related events grouped by principal + source IP within a 15-minute gap)
rather than arbitrary text splices, so retrieved context is coherent.

Test data is **hybrid**: a labelled synthetic corpus (≈200,000 records, 10 injected attack
scenarios) plus real public logs from [Loghub](https://github.com/logpai/loghub). Because the
attacks are planted, retrieval accuracy is measured against real ground truth.

---

## Quick start

```bash
git clone <your-repo-url> slrag && cd slrag
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
make bootstrap                     # install deps
cp .env.example .env               # then paste your free API keys
make test                          # run the test suite
```

Generate the corpus and run the pipeline:

```bash
python scripts/generate_logs.py --full          # ≈200,000 records → data/raw/
python -m slrag.cli ingest --input data/raw --out data/processed
python -m slrag.cli index  --chunks data/processed/chunks.jsonl
python -m slrag.cli ask "did anyone escalate privileges in the last 7 days?"
```

Serve it:

```bash
make api      # FastAPI  → http://localhost:8000/docs
make ui       # Streamlit → http://localhost:8501
```

---

## Models used (all free)

| Role | Default | Alternatives |
|---|---|---|
| Generation | Groq `openai/gpt-oss-120b` (free tier) | Gemini 3.5 Flash free tier · Ollama local · OpenRouter `:free` |
| Embeddings | `BAAI/bge-small-en-v1.5` via **fastembed** (ONNX, 384-d, no torch) | sentence-transformers `bge-base-en-v1.5` · OpenAI `text-embedding-3-small` (paid) |
| Reranker | FlashRank `ms-marco-MiniLM-L-6-v2` (ONNX) | CrossEncoder `bge-reranker-base` |
| Vector store | Chroma (persistent, local) | FAISS · Qdrant |
| Structured store | SQLite (stdlib) | DuckDB |

The provider layer is OpenAI-SDK compatible, so swapping to a paid provider later is a single
`.env` change (`LLM_PROVIDER=openai`). Free-tier note: Groq does not train on your data;
Google's free tier may. Default is Groq.

---

## Results

Measured by `python -m slrag.eval.run_eval` on the curated eval set (≈200 questions with
ground-truth event IDs from the planted scenarios). Numbers are filled in from actual runs —
see `reports/`.

| Metric | Value |
|---|---|
| Recall@1 | _measured in Phase 5_ |
| **Recall@3** | _measured in Phase 5_ |
| Recall@5 | _measured in Phase 5_ |
| MRR@10 | _measured in Phase 5_ |
| Citation precision | _measured in Phase 5_ |

---

## Project layout

```
src/slrag/
  config.py          settings from .env
  providers/         LLM + embedding factories (free providers)
  ingest/            schema · loaders · incident-window chunker · SQLite store
  retrieval/         Chroma · BM25 · RRF hybrid · FlashRank rerank
  agent/             tools · LangChain create_agent pipeline · prompts
  guardrails/        injection screen · redaction · citation enforcement
  eval/              eval-set builder · metric harness
  api/               FastAPI service
  ui/                Streamlit app
scripts/
  generate_logs.py   synthetic CloudTrail + auth.log/syslog corpus with labels
  download_loghub.sh real public log datasets
tests/               unit + integration + contract tests
docs/                BLUEPRINT.md · HANDOFF.md · DATA.md · TESTING.md
```

## Documentation

| Doc | Contents |
|---|---|
| [`docs/BLUEPRINT.md`](docs/BLUEPRINT.md) | full architecture, phase plan, accuracy strategy, risks |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | living handoff: current state, next actions, resume prompt |
| [`docs/DATA.md`](docs/DATA.md) | corpus composition, scenario labels, licensing |

## License

Code: MIT. Loghub datasets: free for research/academic use — cite Zhu et al., *Loghub: A Large
Collection of System Log Datasets for AI-driven Log Analytics*, ISSRE 2023.
