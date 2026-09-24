#!/usr/bin/env python3
"""Cross-platform task runner — the Windows-friendly equivalent of the Makefile.

Windows does not ship ``make``, so every Makefile target has a Python twin here:

    python scripts/tasks.py doctor          # check your environment first
    python scripts/tasks.py venv            # create .venv
    python scripts/tasks.py bootstrap       # install dependencies
    python scripts/tasks.py test            # run the unit tests
    python scripts/tasks.py corpus          # generate the full ~224K corpus
    python scripts/tasks.py git-phase --phase 2 --slug store --message "feat(ingest): ..."

Run ``python scripts/tasks.py --help`` for the full list. Works identically on
Windows, macOS and Linux.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
IS_WINDOWS = os.name == "nt"

REQUIRED_PYTHON = (3, 11)
DOCTOR_IMPORTS = (
    "pytest",
    "langchain",
    "langchain_openai",
    "langchain_chroma",
    "chromadb",
    "fastembed",
    "flashrank",
    "fastapi",
    "streamlit",
    "dotenv",
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


def _printable(cmd: list[str]) -> str:
    return " ".join(f'"{part}"' if " " in part else part for part in cmd)


def run(
    cmd: list[str],
    *,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a command from the repository root, echoing it first."""
    print(f"$ {_printable(cmd)}", flush=True)
    merged = dict(os.environ)
    if env:
        merged.update(env)
    return subprocess.run(cmd, cwd=ROOT, check=check, text=True, env=merged)


def py_module(args: list[str], *, check: bool = False) -> subprocess.CompletedProcess:
    """Run ``python -m ...`` with ``src/`` on the path.

    This means every command works even if you skipped the editable install, which
    removes the most common Windows 'ModuleNotFoundError: slrag' failure.
    """
    src = str(ROOT / "src")
    existing = os.environ.get("PYTHONPATH", "")
    path = f"{src}{os.pathsep}{existing}" if existing else src
    return run([PY, "-m", *args], check=check, env={"PYTHONPATH": path})


def _git(*args: str, check: bool = False) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=ROOT, check=check, text=True, capture_output=True
    )


# --------------------------------------------------------------------------- #
# environment
# --------------------------------------------------------------------------- #


def task_venv(args: argparse.Namespace) -> int:
    venv_dir = ROOT / ".venv"
    if venv_dir.exists():
        print(f".venv already exists at {venv_dir}")
    else:
        run([PY, "-m", "venv", str(venv_dir)])
        print(f"created {venv_dir}")

    activate = r".venv\Scripts\activate" if IS_WINDOWS else "source .venv/bin/activate"
    print(
        "\nActivate it with:\n"
        f"    {activate}\n\n"
        "Then:\n"
        "    python scripts/tasks.py bootstrap\n"
        "    python scripts/tasks.py doctor"
    )
    if IS_WINDOWS:
        print(
            "\nIf PowerShell refuses to run the activate script:\n"
            "    Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass\n"
            "    .venv\\Scripts\\activate"
        )
    return 0


def task_bootstrap(args: argparse.Namespace) -> int:
    run([PY, "-m", "pip", "install", "--upgrade", "pip"])
    run([PY, "-m", "pip", "install", "-r", "requirements-dev.txt"])
    # Editable install makes `slrag` importable from anywhere in the project.
    run([PY, "-m", "pip", "install", "-e", "."])
    print("\nBootstrap complete. Next: python scripts/tasks.py doctor")
    return 0


def task_doctor(args: argparse.Namespace) -> int:
    problems: list[str] = []
    print("slrag doctor")
    print("=" * 68)

    def report(label: str, ok: bool, hint: str = "") -> None:
        print(f"[{'OK  ' if ok else 'FAIL'}] {label}")
        if not ok:
            problems.append(label)
            if hint:
                print(f"         -> {hint}")

    version = sys.version.split()[0]
    report(
        f"Python {version} (need >= {REQUIRED_PYTHON[0]}.{REQUIRED_PYTHON[1]})",
        sys.version_info[:2] >= REQUIRED_PYTHON,
        "install Python 3.12: https://www.python.org/downloads/  (tick 'Add python.exe to PATH')",
    )

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix) or bool(
        os.environ.get("VIRTUAL_ENV")
    )
    report(
        "virtual environment active",
        in_venv,
        r"Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate",
    )

    git = shutil.which("git")
    report("git on PATH", bool(git), "install Git for Windows: https://git-scm.com/download/win")
    if git:
        inside = _git("rev-parse", "--is-inside-work-tree").returncode == 0
        report(
            "repository initialised (git init)",
            inside,
            "git init -b main && git remote add origin <your repo url>",
        )
        if inside and IS_WINDOWS:
            longpaths = _git("config", "--get", "core.longpaths").stdout.strip()
            report(
                "git core.longpaths enabled",
                longpaths.lower() == "true",
                "git config --global core.longpaths true",
            )

    make = shutil.which("make")
    print(
        f"[INFO] make {'found' if make else 'not found — use: python scripts/tasks.py <task>'}",
    )
    if IS_WINDOWS:
        print("[INFO] bash (Git Bash) lets you run scripts/download_loghub.sh and scripts/git_phase.sh")

    env_path = ROOT / ".env"
    report(
        ".env exists",
        env_path.exists(),
        "copy .env.example .env      (PowerShell: Copy-Item .env.example .env)",
    )

    keys: dict[str, str] = {}
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
            stripped = line.strip()
            if "=" in stripped and not stripped.startswith("#"):
                key, _, value = stripped.partition("=")
                keys[key.strip()] = value.strip()

    report(
        "GROQ_API_KEY set (needed from Phase 4 onward)",
        bool(keys.get("GROQ_API_KEY")),
        "free key, no card: https://console.groq.com/keys",
    )
    print(
        f"[INFO] GOOGLE_API_KEY {'set' if keys.get('GOOGLE_API_KEY') else 'not set (optional fallback)'}"
    )

    for name in ("data/raw", "data/processed", "data/eval", "data/cache", "reports"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    report("data directories writable", (ROOT / "data" / "raw").is_dir())

    missing = [m for m in DOCTOR_IMPORTS if importlib.util.find_spec(m) is None]
    report(
        f"dependencies installed ({len(DOCTOR_IMPORTS) - len(missing)}/{len(DOCTOR_IMPORTS)})",
        not missing,
        f"missing: {', '.join(missing)} — run: python scripts/tasks.py bootstrap",
    )

    print("=" * 68)
    if problems:
        print(f"{len(problems)} problem(s) to fix — see the hints above.")
        return 1
    print("Environment looks good. Next: python scripts/tasks.py test")
    return 0


# --------------------------------------------------------------------------- #
# quality
# --------------------------------------------------------------------------- #


def task_test(args: argparse.Namespace) -> int:
    return run(
        [PY, "-m", "pytest", "-m", "not integration and not llm and not slow"], check=False
    ).returncode


def task_test_all(args: argparse.Namespace) -> int:
    return run([PY, "-m", "pytest"], check=False).returncode


def task_lint(args: argparse.Namespace) -> int:
    return run([PY, "-m", "ruff", "check", "src", "tests", "scripts"], check=False).returncode


def task_fmt(args: argparse.Namespace) -> int:
    run([PY, "-m", "ruff", "check", "--fix", "src", "tests", "scripts"], check=False)
    return run([PY, "-m", "ruff", "format", "src", "tests", "scripts"], check=False).returncode


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #


def task_corpus(args: argparse.Namespace) -> int:
    out = args.out or "data/raw"
    return run([PY, "scripts/generate_logs.py", "--full", "--out", out], check=False).returncode


def task_corpus_sample(args: argparse.Namespace) -> int:
    """Small, fast corpus for smoke tests (2 days, all 10 scenarios still planted)."""
    out = args.out or "data/raw"
    return run(
        [
            PY,
            "scripts/generate_logs.py",
            "--out",
            out,
            "--days",
            "2",
            "--events-per-day",
            "1000",
            "--ssh-per-day",
            "400",
            "--syslog-per-day",
            "150",
        ],
        check=False,
    ).returncode


def task_loghub(args: argparse.Namespace) -> int:
    """Download real public log datasets (needs bash — use Git Bash on Windows)."""
    bash = shutil.which("bash")
    if not bash:
        print(
            "bash not found. On Windows, run this from Git Bash:\n"
            "    bash scripts/download_loghub.sh\n"
            "Git Bash ships with Git for Windows."
        )
        return 1
    cmd = [bash, "scripts/download_loghub.sh"]
    if args.all:
        cmd.append("--all")
    return run(cmd, check=False).returncode


# --------------------------------------------------------------------------- #
# pipeline (available from Phase 2 onward)
# --------------------------------------------------------------------------- #


def task_ingest(args: argparse.Namespace) -> int:
    """Phase 2: parse raw logs into sqlite + chunks.jsonl."""
    cmd = ["slrag.cli", "ingest", "--raw", args.input or "data/raw"]
    if getattr(args, "rebuild", False):
        cmd.append("--rebuild")
    if getattr(args, "limit", None):
        cmd += ["--limit", str(args.limit)]
    return py_module(cmd).returncode


def task_stats(args: argparse.Namespace) -> int:
    """Phase 2: corpus overview read straight from the sqlite store."""
    return py_module(["slrag.cli", "stats"]).returncode


def task_index(args: argparse.Namespace) -> int:
    cmd = ["slrag.cli", "index", "--chunks", "data/processed/chunks.jsonl"]
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    return py_module(cmd).returncode


def task_ask(args: argparse.Namespace) -> int:
    return py_module(["slrag.cli", "ask", args.question]).returncode


def task_eval(args: argparse.Namespace) -> int:
    return py_module(["slrag.eval.run_eval"]).returncode


def task_api(args: argparse.Namespace) -> int:
    return py_module(
        ["uvicorn", "slrag.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
    ).returncode


def task_ui(args: argparse.Namespace) -> int:
    return py_module(["streamlit", "run", "src/slrag/ui/app.py"]).returncode


# --------------------------------------------------------------------------- #
# git
# --------------------------------------------------------------------------- #


def task_git_phase(args: argparse.Namespace) -> int:
    """One phase = one branch + one commit + one push + one tag (Windows-safe)."""
    if _git("rev-parse", "--git-dir").returncode != 0:
        print("Not a git repository yet. Run:\n    git init -b main")
        print("    git remote add origin <your repo url>")
        return 1

    branch = f"phase/{args.phase}-{args.slug}"
    tag = f"v0.{args.phase}.0-phase{args.phase}"

    created = _git("checkout", "-b", branch)
    if created.returncode != 0:
        print(f"branch {branch} already exists — switching to it")
        run(["git", "checkout", branch])

    run(["git", "add", "-A"])
    if _git("diff", "--cached", "--quiet").returncode == 0:
        print("Nothing staged — did you save your files?")
        return 1

    run(["git", "status", "--short"])
    run(["git", "commit", "-m", args.message])
    run(["git", "push", "-u", "origin", branch])

    if not _git("tag", "-l", tag).stdout.strip():
        run(["git", "tag", "-a", tag, "-m", f"Phase {args.phase}: {args.message}"])
        print(f"tagged {tag} — push it with: git push origin {tag}")

    print(
        "\n" + "-" * 62 + "\n"
        "NEXT (on GitHub, not here):\n"
        f"  1. Open the PR:  {branch} -> main\n"
        "     gh pr create --fill --base main --head " + branch + "\n"
        "  2. Confirm CI is green (Actions tab).\n"
        "  3. Squash-merge into main.\n"
        "  4. Locally:\n"
        "       git checkout main && git pull\n"
        f"       git push origin {tag}\n"
        "  5. Update docs/HANDOFF.md and commit that on main.\n"
        + "-" * 62
    )
    return 0


# --------------------------------------------------------------------------- #
# housekeeping
# --------------------------------------------------------------------------- #


def task_clean(args: argparse.Namespace) -> int:
    removed = 0
    for path in list(ROOT.glob("**/__pycache__")):
        shutil.rmtree(path, ignore_errors=True)
        removed += 1
    for name in (".pytest_cache", ".ruff_cache", "htmlcov", "build", "dist"):
        target = ROOT / name
        if target.exists():
            shutil.rmtree(target, ignore_errors=True)
            removed += 1
    for name in (".coverage", "coverage.xml"):
        target = ROOT / name
        if target.exists():
            target.unlink()
            removed += 1
    print(f"cleaned {removed} path(s)")
    return 0


def task_tree(args: argparse.Namespace) -> int:
    """Print the repository tree, marking generated (gitignored) paths."""
    generated = {"data", "reports", ".venv", "__pycache__", ".git", ".pytest_cache", ".ruff_cache"}
    skip = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache"}

    def walk(directory: Path, prefix: str = "") -> None:
        entries = sorted(
            (p for p in directory.iterdir() if p.name not in skip),
            key=lambda p: (p.is_file(), p.name.lower()),
        )
        for index, entry in enumerate(entries):
            last = index == len(entries) - 1
            branch = "└── " if last else "├── "
            marker = "  (generated, gitignored)" if entry.name in generated else ""
            print(f"{prefix}{branch}{entry.name}{marker}")
            if entry.is_dir():
                walk(entry, prefix + ("    " if last else "│   "))

    print(f"{ROOT.name}/")
    walk(ROOT)
    return 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tasks.py",
        description="Cross-platform task runner for slrag (Windows-friendly Makefile).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "examples:\n"
            "  python scripts/tasks.py doctor\n"
            "  python scripts/tasks.py bootstrap && python scripts/tasks.py test\n"
            "  python scripts/tasks.py corpus\n"
            "  python scripts/tasks.py git-phase --phase 2 --slug store "
            '--message "feat(ingest): add sqlite store and cli"\n'
        ),
    )
    sub = parser.add_subparsers(dest="task", required=True, metavar="TASK")

    def add(name: str, func, help_text: str):
        node = sub.add_parser(name, help=help_text, description=help_text)
        node.set_defaults(func=func)
        return node

    add("venv", task_venv, "create .venv and print the activation command")
    add("bootstrap", task_bootstrap, "install runtime + dev dependencies (editable install)")
    add("doctor", task_doctor, "check Python, venv, git, keys, directories and dependencies")
    add("test", task_test, "run the fast unit tests (no network, no models, no keys)")
    add("test-all", task_test_all, "run every test including integration")
    add("lint", task_lint, "ruff check")
    add("fmt", task_fmt, "ruff check --fix + ruff format")
    add("tree", task_tree, "print the repository tree")

    node = add("corpus", task_corpus, "generate the full ~224K labelled corpus")
    node.add_argument("--out", default=None, help="output directory (default: data/raw)")
    node = add("corpus-sample", task_corpus_sample, "generate a fast 2-day corpus for smoke tests")
    node.add_argument("--out", default=None, help="output directory (default: data/raw)")
    node = add("loghub", task_loghub, "download real public log datasets (needs bash)")
    node.add_argument("--all", action="store_true", help="include the extra datasets")

    node = add("ingest", task_ingest, "parse raw logs into events.sqlite + chunks.jsonl (Phase 2)")
    node.add_argument("--input", default=None, help="raw file or directory (default: data/raw)")
    node.add_argument("--rebuild", action="store_true", help="drop and recreate the tables")
    node.add_argument("--limit", type=int, default=None, help="stop after N events (smoke test)")
    add("stats", task_stats, "corpus overview from the sqlite store (Phase 2)")
    node = add("index", task_index, "embed chunks into Chroma (Phase 3)")
    node.add_argument("--limit", type=int, default=None, help="index only the first N chunks")
    node = add("ask", task_ask, "ask a question from the CLI (Phase 4)")
    node.add_argument("question", help="the natural-language question")
    add("eval", task_eval, "run the retrieval eval harness (Phase 5)")
    add("api", task_api, "serve the FastAPI app on :8000 (Phase 6)")
    add("ui", task_ui, "serve the Streamlit app on :8501 (Phase 7)")

    node = add("git-phase", task_git_phase, "branch + commit + push + tag for one phase")
    node.add_argument("--phase", required=True, help="phase number, e.g. 2")
    node.add_argument("--slug", required=True, help="short branch slug, e.g. store")
    node.add_argument("--message", required=True, help="commit message")

    add("clean", task_clean, "remove caches and build artefacts")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
