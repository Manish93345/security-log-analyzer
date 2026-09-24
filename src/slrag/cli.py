"""``slrag`` command-line interface.

    slrag ingest --raw data/raw     # parse -> normalize -> window -> sqlite + chunks.jsonl
    slrag stats                     # corpus overview: counts, top talkers, risk
    slrag version

Runs as the installed console script (``slrag ...``) or as ``python -m slrag ...``.
Later phases add ``index``, ``search``, ``ask`` and ``eval`` subcommands here.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .config import get_settings
from .ingest.pipeline import run_ingest
from .ingest.store import EventStore
from .logging_setup import setup_logging

_RULE = "=" * 68


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("slrag")
    except Exception:  # pragma: no cover - only when running from a bare checkout
        return "0.2.0"


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    raw_paths = [Path(p) for p in args.raw]
    missing = [str(p) for p in raw_paths if not p.exists()]
    if missing:
        print(f"error: raw path(s) not found: {', '.join(missing)}", file=sys.stderr)
        print(
            "hint: generate a corpus first ->  python scripts/generate_logs.py --full "
            "--out data/raw",
            file=sys.stderr,
        )
        return 2

    try:
        result = run_ingest(
            raw_paths,
            db_path=args.db,
            chunks_path=args.chunks,
            assumed_year=args.year,
            rebuild=args.rebuild,
            limit=args.limit,
            verbose=not args.json,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        print(f"\nNext:  slrag stats --db {result.db_path}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    db_path = Path(args.db) if args.db else get_settings().sqlite_path
    if not db_path.exists():
        print(f"error: {db_path} not found", file=sys.stderr)
        print("hint: run `slrag ingest` first", file=sys.stderr)
        return 2

    with EventStore(db_path, read_only=True) as store:
        stats = store.stats(top_n=args.top)

    if args.json:
        print(json.dumps(stats, indent=2))
        return 0

    events, chunks = stats["events"], stats["chunks"]
    by_source = " ".join(f"{k}={v:,}" for k, v in events["by_source"].items())
    total = events["total"] or 1
    failure_pct = 100.0 * events["failures"] / total

    print(f"slrag stats — {stats['db_path']} ({stats['db_size_mb']} MB, schema v{stats['schema_version']})")
    print(_RULE)
    print(f"events           {events['total']:>10,}   {by_source}")
    print(f"failures         {events['failures']:>10,}   ({failure_pct:.1f}% of events)")
    print(f"time range       {events['first_ts']} -> {events['last_ts']}")
    print(
        f"windows          {chunks['total']:>10,}   high risk (>=50): {chunks['high_risk']:,} | "
        f"max risk {chunks['max_risk']}"
    )
    print(
        f"window size      median {chunks['median_events']} / p90 {chunks['p90_events']} / "
        f"max {chunks['max_events']} events"
    )
    print(_RULE)
    _print_ranked("top actions", events["top_actions"], "action")
    _print_ranked("top principals", events["top_principals"], "principal", failures=True)
    _print_ranked("top source IPs", events["top_src_ips"], "src_ip", failures=True)
    return 0


def _print_ranked(
    label: str, rows: list[dict[str, Any]], key: str, *, failures: bool = False
) -> None:
    if not rows:
        print(f"{label:<16} -")
        return
    lines: list[str] = []
    for index, row in enumerate(rows):
        value = str(row[key])
        if len(value) > 46:
            value = value[:43] + "..."
        cell = f"{value} {row['n']:,}"
        if failures and row.get("failures"):
            cell += f" ({row['failures']:,} fail)"
        lines.append(f"{label:<16} {cell}" if index == 0 else f"{'':<16} {cell}")
    print("\n".join(lines))


def cmd_version(args: argparse.Namespace) -> int:
    del args
    print(f"slrag {_version()}")
    return 0


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="slrag", description="RAG-powered security log analyzer"
    )
    parser.add_argument("--version", action="version", version=f"slrag {_version()}")
    parser.add_argument(
        "--log-level", default=None, help="DEBUG | INFO | WARNING (default: from .env)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="parse raw logs into the sqlite + jsonl stores")
    p_ingest.add_argument("--raw", nargs="+", default=["data/raw"], help="files or directories")
    p_ingest.add_argument("--db", default=None, help="sqlite output path")
    p_ingest.add_argument("--chunks", default=None, help="chunks.jsonl output path")
    p_ingest.add_argument("--year", type=int, default=None, help="year for year-less syslog lines")
    p_ingest.add_argument("--limit", type=int, default=None, help="stop after N events (smoke test)")
    p_ingest.add_argument("--rebuild", action="store_true", help="drop and recreate the tables")
    p_ingest.add_argument("--json", action="store_true", help="machine-readable output")
    p_ingest.set_defaults(func=cmd_ingest)

    p_stats = sub.add_parser("stats", help="show corpus statistics")
    p_stats.add_argument("--db", default=None, help="sqlite path (default: data/processed/events.sqlite)")
    p_stats.add_argument("--top", type=int, default=5, help="rows per ranked list")
    p_stats.add_argument("--json", action="store_true", help="machine-readable output")
    p_stats.set_defaults(func=cmd_stats)

    p_version = sub.add_parser("version", help="print the version")
    p_version.set_defaults(func=cmd_version)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    setup_logging(args.log_level or get_settings().log_level)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
