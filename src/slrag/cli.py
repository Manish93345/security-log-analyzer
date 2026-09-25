"""``slrag`` command-line interface.

    slrag ingest --raw data/raw     # parse -> normalize -> window -> sqlite + chunks.jsonl
    slrag stats                     # corpus overview: counts, top talkers, risk
    slrag index                     # embed every window into the Chroma vector store
    slrag search "failed sudo ..."  # hybrid retrieval: dense + BM25 -> RRF -> rerank
    slrag version

Runs as the installed console script (``slrag ...``) or as ``python -m slrag ...``.
Later phases add ``ask`` and ``eval`` subcommands here.
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
from .retrieval.indexer import build_index, read_index_meta
from .retrieval.pipeline import RetrievalEngine
from .retrieval.vector_store import INDEX_META_FILE

_RULE = "=" * 68


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("slrag")
    except Exception:  # pragma: no cover - only when running from a bare checkout
        return "0.3.0"


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def cmd_ingest(args: argparse.Namespace) -> int:
    get_settings()
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


def _collection_count(settings) -> int:
    """Live collection size, without loading an embedding model."""
    from .retrieval.vector_store import VectorStore

    try:
        return VectorStore(settings.chroma_dir, settings=settings).count()
    except Exception:  # pragma: no cover - defensive
        return 0


def cmd_index(args: argparse.Namespace) -> int:
    settings = get_settings()
    chunks_path = Path(args.chunks) if args.chunks else settings.processed_dir / "chunks.jsonl"

    if args.check:
        meta = read_index_meta(settings.chroma_dir)
        live = _collection_count(settings)
        if not meta:
            if live:
                print(f"unfinished index at {settings.chroma_dir}")
                print(_RULE)
                print(f"{'windows_in_collection':<22} {live:,}")
                print(_RULE)
                print(f"no {INDEX_META_FILE} — the run never reached its final step, but every")
                print("window it committed is still on disk. Re-run `slrag index`: resume")
                print("skips those ids and embeds only what is missing.")
                return 2
            print(f"no index found at {settings.chroma_dir}", file=sys.stderr)
            print("hint: run `slrag index`", file=sys.stderr)
            return 2
        print(f"slrag index — {settings.chroma_dir}")
        print(_RULE)
        for key in (
            "built_at",
            "collection",
            "embedding_model",
            "dimension",
            "embedding_device",
            "chunks_indexed",
            "embedded_this_run",
            "total_in_collection",
            "windows_per_second",
            "chunks_path",
        ):
            print(f"{key:<22} {meta.get(key, '-')}")
        if live and int(meta.get("total_in_collection") or 0) != live:
            print(f"{'live_count':<22} {live:,}  (differs from the recorded build)")
        print(_RULE)
        print(f"meta file: {settings.chroma_dir / INDEX_META_FILE}")
        return 0

    try:
        result = build_index(
            chunks_path,
            limit=args.limit,
            reset=args.reset,
            resume=not getattr(args, "no_resume", False),
            verbose=not args.json,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        print("hint: run `slrag ingest` first", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(result.as_dict(), indent=2))
    else:
        if result.resumed:
            print(
                f"\nembedded {result.chunks_indexed:,} new windows in "
                f"{result.elapsed_s:.1f}s ({result.windows_per_second:.0f}/s) — "
                f"{result.already_indexed:,} were already indexed and skipped"
            )
        else:
            print(
                f"\nindexed {result.chunks_indexed:,} windows into '{result.collection}' "
                f"({result.embedding_model}, {result.dimension}-d, {result.device}) "
                f"in {result.elapsed_s:.1f}s ({result.windows_per_second:.0f}/s)"
            )
        print(f"collection now holds {result.already_indexed + result.chunks_indexed:,} windows")
        print(f"stored in {result.persist_dir}")
        print('\nNext:  slrag search "failed sudo commands on the web server"')
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    try:
        engine = RetrievalEngine.from_settings()
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        result = engine.retrieve(
            args.query,
            k=args.k,
            rerank=False if args.no_rerank else None,
        )
        if args.json:
            print(json.dumps(result.as_dict(), indent=2))
            return 0

        _print_search(result)

        if args.show_events and result.hits:
            top = result.hits[0]
            events = engine.store.events_for_chunk(top.chunk_id)
            print(f"\nraw records behind {top.chunk_id} (first {args.show_events} of {len(events)}):")
            for event in events[: args.show_events]:
                print(f"  [{event.event_id}] {event.ts} {event.action} {event.status}")
                print(f"      {event.raw.strip()[:180]}")
    finally:
        engine.close()
    return 0


def _print_search(result) -> None:
    diagnostics = result.diagnostics
    print(f'slrag search — "{result.question}"')
    print(_RULE)

    if not result.hits:
        print("no windows matched.")
        print("  - is the vector index built?   slrag index")
        print("  - does the corpus have this?   slrag stats")
        return

    for hit in result.hits:
        chunk = hit.chunk
        provenance = " ".join(
            part
            for part in (
                f"dense#{hit.dense_rank}" if hit.dense_rank else "",
                f"bm25#{hit.bm25_rank}" if hit.bm25_rank else "",
            )
            if part
        )
        print(
            f"{hit.rank:>2}. {chunk.chunk_id}  score={hit.score:.4f}  "
            f"risk={chunk.risk_score:<3} {provenance}"
        )
        print(
            f"    {chunk.ts_start} -> {chunk.ts_end} | {chunk.source} | "
            f"principal={chunk.principal or '-'} | src_ip={chunk.src_ip or '-'}"
        )
        print(
            f"    events={chunk.n_events} failures={chunk.n_failures} | "
            f"{', '.join(chunk.event_names[:6])}"
        )
        body = [line for line in chunk.text.split("\n")[1:] if line.strip()]
        if body:
            print(f"    {body[0][:160]}")
            if len(body) > 1:
                print(f"    {body[1][:160]}")
        print()

    timings = diagnostics["timings_ms"]
    print(_RULE)
    print(
        f"dense {diagnostics['dense_candidates']} | bm25 {diagnostics['bm25_candidates']} | "
        f"fused {diagnostics['fused_candidates']} | reranker={diagnostics['reranker']} | "
        f"{timings['total_ms']} ms"
    )
    print(
        f"index {diagnostics['indexed_windows']:,} windows | "
        f"bm25 {diagnostics['bm25_windows']:,} windows | rrf k={diagnostics['rrf_k']}"
    )


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

    p_index = sub.add_parser("index", help="embed every window into the Chroma vector store")
    p_index.add_argument("--chunks", default=None, help="chunks.jsonl path (default: data/processed/chunks.jsonl)")
    p_index.add_argument("--limit", type=int, default=None, help="index only the first N windows")
    p_index.add_argument("--reset", action="store_true", help="drop the existing collection first")
    p_index.add_argument(
        "--no-resume",
        action="store_true",
        help="re-embed every window even if it is already in the collection",
    )
    p_index.add_argument("--check", action="store_true", help="report the existing index instead of building")
    p_index.add_argument("--json", action="store_true", help="machine-readable output")
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="hybrid retrieval over the incident windows")
    p_search.add_argument("query", help="the natural-language question")
    p_search.add_argument(
        "-k",
        "--k",
        "--top-k",
        dest="k",
        type=int,
        default=None,
        help="windows to return (default: RETRIEVE_TOP_K)",
    )
    p_search.add_argument("--no-rerank", action="store_true", help="skip the cross-encoder")
    p_search.add_argument("--show-events", type=int, default=0, metavar="N", help="print N raw records behind the top window")
    p_search.add_argument("--json", action="store_true", help="machine-readable output")
    p_search.set_defaults(func=cmd_search)

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
