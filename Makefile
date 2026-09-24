# slrag — developer tasks
# Uses .RECIPEPREFIX so the file survives copy/paste and editors that eat tabs.
# On Windows (no make): use `python scripts/tasks.py <task>` — same targets.
.RECIPEPREFIX = >
.DEFAULT_GOAL := help
PY ?= python
PHASE ?= 0
SLUG ?= work
MSG ?= chore: update

.PHONY: help venv bootstrap doctor test test-all lint fmt clean corpus corpus-sample loghub ingest index ask api ui docker-build docker-up eval git-phase tree

help:
> @echo "slrag — available targets:"
> @echo "  venv          create .venv and print the activation command"
> @echo "  bootstrap     install runtime + dev deps (editable install)"
> @echo "  doctor        check python, venv, git, keys, directories, deps"
> @echo "  test          fast unit tests (no network, no models)"
> @echo "  test-all      all tests including integration"
> @echo "  lint / fmt    ruff check / ruff format"
> @echo "  corpus        generate the full ~224K synthetic corpus"
> @echo "  corpus-sample fast 2-day corpus for smoke tests"
> @echo "  loghub        download real public log datasets"
> @echo "  ingest        parse raw logs -> events.sqlite + chunks.jsonl"
> @echo "  index         embed chunks -> Chroma + BM25"
> @echo "  ask           ask a question:  make ask Q=\"...\""
> @echo "  eval          run the retrieval eval harness"
> @echo "  api / ui      run the FastAPI service / Streamlit app"
> @echo "  tree          print the repository layout"
> @echo "  docker-build  build both container images"
> @echo "  docker-up     docker compose up"
> @echo "  git-phase     branch + commit + push: make git-phase PHASE=2 SLUG=store MSG=\"feat(ingest): ...\""
> @echo "  clean         remove caches and generated artefacts"
> @echo ""
> @echo "  Windows users: python scripts/tasks.py <target>"

venv:
> $(PY) -m venv .venv
> @echo "activate:  source .venv/bin/activate   (Windows: .venv\\Scripts\\activate)"

bootstrap:
> $(PY) -m pip install --upgrade pip
> $(PY) -m pip install -r requirements-dev.txt
> $(PY) -m pip install -e .
> @echo "OK — next: cp .env.example .env, add free keys, then 'make doctor'"

doctor:
> $(PY) scripts/tasks.py doctor

test:
> $(PY) -m pytest -m "not integration and not llm and not slow"

test-all:
> $(PY) -m pytest

lint:
> $(PY) -m ruff check src tests scripts

fmt:
> $(PY) -m ruff check --fix src tests scripts
> $(PY) -m ruff format src tests scripts

tree:
> $(PY) scripts/tasks.py tree

corpus:
> $(PY) scripts/generate_logs.py --full

corpus-sample:
> $(PY) scripts/generate_logs.py --out data/raw --days 2 --events-per-day 1000 --ssh-per-day 400 --syslog-per-day 150

loghub:
> bash scripts/download_loghub.sh

ingest:
> $(PY) -m slrag.cli ingest --input data/raw --out data/processed

index:
> $(PY) -m slrag.cli index --chunks data/processed/chunks.jsonl

ask:
> $(PY) -m slrag.cli ask "$(Q)"

eval:
> $(PY) -m slrag.eval.run_eval

api:
> $(PY) -m uvicorn slrag.api.main:app --host 0.0.0.0 --port 8000 --reload

ui:
> $(PY) -m streamlit run src/slrag/ui/app.py

docker-build:
> docker build -f docker/Dockerfile -t slrag:latest .
> docker build -f docker/Dockerfile.ui -t slrag-ui:latest .

docker-up:
> docker compose -f docker/docker-compose.yml up --build

git-phase:
> bash scripts/git_phase.sh $(PHASE) $(SLUG) "$(MSG)"

clean:
> rm -rf .pytest_cache .ruff_cache htmlcov .coverage
> find . -type d -name __pycache__ -prune -exec rm -rf {} +
