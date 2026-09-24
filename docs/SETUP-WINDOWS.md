# SETUP — Windows

Everything in this project runs natively on Windows. There is exactly **one** Windows
gotcha to handle: Windows does not ship `make`. Every Makefile target therefore has a
Python twin in `scripts/tasks.py`, which works identically on Windows, macOS and Linux.

---

## 0. Fix the folder nesting first ⚠️

The scaffold zip contains a **top-level `slrag/` folder**. If you extracted it directly into
`security-log-analyzer\`, your repo root now looks like this:

```
security-log-analyzer\slrag\README.md        <-- WRONG
security-log-analyzer\slrag\src\...
```

It must be flat — the repo root itself is the project root:

```
security-log-analyzer\README.md              <-- RIGHT
security-log-analyzer\src\...
```

**PowerShell** (adjust the path):

```powershell
cd C:\path\to\security-log-analyzer
Get-ChildItem -Force .\slrag | Move-Item -Destination .
Remove-Item -Force .\slrag
Get-ChildItem -Force          # you should now see README.md, src, tests, docs, Makefile …
```

`-Force` matters: it includes dotfiles such as `.gitignore`, `.gitattributes`, `.env.example`.
If you would rather start clean, delete the folder and re-extract with **"Extract Here"**
(not "Extract All"), which unpacks into the current directory instead of a subfolder.

---

## 1. Prerequisites

| Tool | Needed for | Install |
|---|---|---|
| **Python 3.11–3.13** | everything | <https://www.python.org/downloads/> — **tick "Add python.exe to PATH"** |
| **Git for Windows** | version control, and it also gives you **Git Bash** for the `.sh` scripts | <https://git-scm.com/download/win> |
| **Groq + Google AI Studio keys** (free, no card) | Phase 4+ only | <https://console.groq.com/keys> · <https://aistudio.google.com/app/apikey> |
| Docker Desktop | Phase 8 only (optional) | <https://www.docker.com/products/docker-desktop/> |
| GNU make | optional — `tasks.py` replaces it | `winget install ezwinports.make` |

**You do NOT need Visual Studio, CUDA, or `torch`.** The default embedding and reranking
models are ONNX (`fastembed` + `flashrank`), so no 2.5 GB PyTorch download. If you ever
uncomment `sentence-transformers` in `requirements.txt`, that is when torch appears.

One-time Git hardening for deep folder trees (Chroma writes long paths):

```powershell
git config --global core.longpaths true
git config --global core.autocrlf false     # .gitattributes handles line endings instead
```

---

## 2. Create the environment

```powershell
cd C:\path\to\security-log-analyzer
python scripts/tasks.py venv
.venv\Scripts\activate
```

If PowerShell blocks the activation script:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.venv\Scripts\activate
```

Then install everything:

```powershell
python scripts/tasks.py bootstrap
```

`bootstrap` does three things: upgrades pip, installs `requirements-dev.txt`, and runs
`pip install -e .` so `slrag` is importable from anywhere in the project.

---

## 3. Configure keys (needed from Phase 4 onward)

```powershell
Copy-Item .env.example .env
notepad .env
```

Fill in `GROQ_API_KEY` (and optionally `GOOGLE_API_KEY`). **Phases 0–3 need no keys at all** —
parsing, chunking, embedding and retrieval are fully local. `.env` is gitignored; never commit it.

---

## 4. Verify, test, generate the corpus

```powershell
python scripts/tasks.py doctor        # checks Python, venv, git, keys, deps, directories
python scripts/tasks.py test          # unit tests: no network, no keys, no model downloads
python scripts/tasks.py corpus        # ~224,275 records in about a minute
python scripts/tasks.py tree          # show the repo layout
```

Expected corpus output:

```
[gen] cloudtrail=123,414  auth.log=80,851  syslog=20,010  TOTAL=224,275
[gen] labels -> data\eval\labels.json
[gen] target >= 100,000 records -> OK
```

Optional — real public logs (needs Git Bash, which came with Git for Windows):

```powershell
bash scripts/download_loghub.sh
```

---

## 5. First push to GitHub

Create an empty **public** repo named `security-log-analyzer` on GitHub (no README, no
`.gitignore` — the project already has them), then:

```powershell
git init -b main
git add -A
git commit -m "feat: phase 0+1 - scaffold, providers, labelled synthetic corpus"
git remote add origin https://github.com/<your-username>/security-log-analyzer.git
git push -u origin main
git tag -a v0.1.0-phase0 -m "Phase 0: scaffold"
git push origin --tags
```

Then, for every later phase:

```powershell
python scripts/tasks.py git-phase --phase 2 --slug store --message "feat(ingest): add sqlite store and cli"
```

which creates `phase/2-store`, commits, pushes, tags `v0.2.0-phase2`, and prints the PR steps.

---

## 6. Repository structure

Repo root = `security-log-analyzer\`. `[G]` marks generated / gitignored paths.

```
security-log-analyzer\
├── .github\workflows\ci.yml        CI: ruff + unit tests on 3.11 and 3.12
├── .gitattributes                  line-ending rules (Windows-safe)
├── .gitignore
├── .env.example                    every tunable, with free-key links
├── .env                            [G] you create this — never committed
├── Makefile                        for macOS/Linux/WSL (tasks.py is the Windows twin)
├── README.md                       quick start, model table, measured results
├── pyproject.toml                  setuptools + pytest config + ruff config
├── requirements.txt                pinned runtime dependencies
├── requirements-dev.txt            + pytest, pytest-cov, ruff
├── config\
│   └── default.yaml                chunking / retrieval / guardrail tunables
├── docker\
│   ├── Dockerfile                  multi-stage API image, non-root, healthcheck
│   ├── Dockerfile.ui               Streamlit image
│   └── docker-compose.yml          api + ui + shared volumes
├── docs\
│   ├── BLUEPRINT.md                the plan: architecture, 8 phases, accuracy strategy
│   ├── HANDOFF.md                  living state + resume prompt (rewritten each phase)
│   ├── DATA.md                     corpus composition, scenario labels, licensing
│   └── SETUP-WINDOWS.md            this file
├── scripts\
│   ├── generate_logs.py            labelled synthetic corpus (~224K records)
│   ├── download_loghub.sh          real public datasets (run in Git Bash)
│   ├── git_phase.sh                bash git helper
│   └── tasks.py                    cross-platform task runner  <-- use this on Windows
├── src\slrag\                      the import package (repo name ≠ package name, by design)
│   ├── __init__.py
│   ├── config.py                   frozen Settings dataclass, read from .env
│   ├── logging_setup.py
│   ├── providers\                  llm.py + embeddings.py (free providers only)
│   ├── ingest\                     schema.py, loaders.py, chunker.py   (store.py in Phase 2)
│   ├── retrieval\                  Phase 3 — Chroma + BM25 + RRF + rerank
│   ├── agent\                      Phase 4 — LangChain create_agent + tools
│   ├── guardrails\                 Phase 5 — injection screen, redaction, citations
│   ├── eval\                       Phase 5 — eval-set builder + metric harness
│   ├── api\                        Phase 6 — FastAPI service
│   └── ui\                         Phase 7 — Streamlit app
├── tests\
│   ├── conftest.py                 fixtures: small corpus, CT record, syslog lines
│   ├── test_generate_logs.py       determinism, counts, JSON validity, label integrity
│   ├── test_loaders.py             every parser branch, malformed tolerance, id stability
│   └── test_chunker.py             window merge/split, caps, risk score, citations
├── data\                           [G] everything below is gitignored
│   ├── raw\                        cloudtrail\ + syslog\ + manifest.json
│   ├── processed\                  events.sqlite + chunks.jsonl      (Phase 2)
│   ├── eval\labels.json            ground truth for the 10 planted scenarios
│   ├── chroma\                     vector index                      (Phase 3)
│   └── cache\                      LLM response cache (free-tier rate limits)
└── reports\                        [G] eval reports, timestamped
```

**Why is the package `slrag` and not `security_log_analyzer`?** The repo name is for humans
and GitHub; the import name is a short internal handle. That is a normal split (e.g. repo
`langchain-ai/langchain` importing as `langchain`). If you would rather they match, say so and
it becomes a mechanical find-and-replace in Phase 2 — better done before the code grows.

---

## 7. Where to put the sample corpus zip

**Keep it outside the repo.** `slrag-sample-corpus.zip` is only there so you can inspect the
log format and `labels.json` *before* running anything. The real corpus is regenerated by
`python scripts/tasks.py corpus` (deterministic — same seed, same bytes).

Recommended:

```
C:\Users\<you>\Documents\slrag-sample-corpus\
├── cloudtrail\cloudtrail-2026-09-22.json
├── cloudtrail\cloudtrail-2026-09-23.json
├── syslog\auth.log
├── syslog\syslog
├── labels.json
└── manifest.json
```

Do **not** unzip it into `security-log-analyzer\data\raw\` — the archive has its own
`slrag-sample-corpus\` folder inside it, so you would end up with
`data\raw\slrag-sample-corpus\cloudtrail\…`, which the loader would still read (it walks
directories) but which makes the layout confusing, and it would collide with the full corpus
generated later. If you do want it in the repo, move the *contents* up:

```powershell
Move-Item .\data\raw\slrag-sample-corpus\* .\data\raw\ -Force
Remove-Item .\data\raw\slrag-sample-corpus
```

Either way it is gitignored, so nothing leaks into your commit.

---

## 8. Makefile ↔ tasks.py cheat sheet

| Makefile (macOS/Linux/WSL) | Windows |
|---|---|
| `make bootstrap` | `python scripts/tasks.py bootstrap` |
| `make test` | `python scripts/tasks.py test` |
| `make test-all` | `python scripts/tasks.py test-all` |
| `make lint` / `make fmt` | `python scripts/tasks.py lint` / `fmt` |
| `make corpus` | `python scripts/tasks.py corpus` |
| `make ingest` | `python scripts/tasks.py ingest` |
| `make index` | `python scripts/tasks.py index` |
| `make ask Q="..."` | `python scripts/tasks.py ask "..."` |
| `make eval` | `python scripts/tasks.py eval` |
| `make api` | `python scripts/tasks.py api` |
| `make ui` | `python scripts/tasks.py ui` |
| `make git-phase PHASE=2 SLUG=store MSG="..."` | `python scripts/tasks.py git-phase --phase 2 --slug store --message "..."` |
| `make clean` | `python scripts/tasks.py clean` |
| — | `python scripts/tasks.py doctor` (Windows-only extra: environment check) |
| — | `python scripts/tasks.py tree` (Windows-only extra: show the layout) |

---

## 9. Common Windows errors and their fixes

| Symptom | Cause | Fix |
|---|---|---|
| `'python' is not recognized` | Python not on PATH | reinstall with "Add python.exe to PATH", or use `py` instead of `python` |
| `make: command not found` | make not installed | use `python scripts/tasks.py <task>` |
| `.venv\Scripts\activate : cannot be loaded because running scripts is disabled` | PowerShell execution policy | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` |
| `ModuleNotFoundError: No module named 'slrag'` | editable install missing | `python scripts/tasks.py bootstrap`, or run via `tasks.py` which sets `PYTHONPATH` for you |
| `ModuleNotFoundError: No module named 'pytest'` | wrong venv active | activate `.venv`, then `python scripts/tasks.py bootstrap` |
| `Filename too long` / `path too long` during `git add` | Windows path limit + Chroma | `git config --global core.longpaths true` |
| Every file shows as modified after a clone | CRLF conversion | `git config --global core.autocrlf false`; `.gitattributes` is already in the repo |
| `bash: command not found` when running `download_loghub.sh` | no bash | run it from **Git Bash**, or skip it (the synthetic corpus is complete without it) |
| `pip` is slow / wheels fail to build | missing build tools for a source package | the pinned versions all ship Windows wheels — check you are on 64-bit Python 3.12 |
| Antivirus slows ingestion to a crawl | real-time scanning of `data\` | add the repo folder to your AV exclusion list |
