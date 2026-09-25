#!/usr/bin/env python
"""Where does indexing time actually go?

``slrag index`` does two very different things for every window:

1. a transformer forward pass — parallel, and the part a GPU can accelerate;
2. a write into Chroma's SQLite + HNSW store — serial, CPU-only, and completely indifferent
   to how fast the GPU is.

The blended ``windows/s`` the indexer prints cannot tell these apart, which leaves "would a
GPU help?" unanswerable. This script times the two separately on a small sample, prints the
split, and states the decision that follows.

It never touches your real index — the store phase writes into a throwaway temp directory.

    python scripts/bench_embed.py                       # 200 windows, batch 64, device auto
    python scripts/bench_embed.py --limit 500
    python scripts/bench_embed.py --store-batch-size 1024   # try a bigger Chroma batch
    python scripts/bench_embed.py --device cuda             # after installing onnxruntime-gpu
"""

from __future__ import annotations

import argparse
import gc
import itertools
import shutil
import sys
import tempfile
import time
from pathlib import Path

FULL_CORPUS = 38767


def _ensure_slrag_importable() -> None:
    """Allow running from a checkout that was never ``pip install -e .``'d."""
    try:
        import slrag  # noqa: F401

        return
    except ModuleNotFoundError:
        pass
    candidate = Path(__file__).resolve().parents[1] / "src"
    if candidate.is_dir():
        sys.path.insert(0, str(candidate))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--chunks",
        default="data/processed/chunks.jsonl",
        help="chunks.jsonl to sample (default: data/processed/chunks.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=200, help="windows to sample (default: 200)")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="embedding batch size (default: 64 = EMBEDDING_BATCH_SIZE)",
    )
    parser.add_argument(
        "--store-batch-size",
        type=int,
        default=256,
        help="Chroma add batch size (default: 256 = INDEX_BATCH_SIZE)",
    )
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--threads", type=int, default=0, help="ONNX threads (0 = auto)")
    return parser.parse_args()


def load_windows(path: Path, limit: int):
    from slrag.ingest.store import iter_chunks_jsonl

    if not path.exists():
        sys.exit(f"error: {path} not found — run `slrag ingest` first")
    chunks = list(itertools.islice(iter_chunks_jsonl(path), limit))
    if not chunks:
        sys.exit(f"error: {path} contained no windows")
    return chunks


def bench_embed(texts, args):
    from fastembed import TextEmbedding

    kwargs: dict = {"model_name": args.model}
    if args.threads:
        kwargs["threads"] = args.threads
    if args.device in {"cpu", "cuda"}:
        kwargs["cuda"] = args.device == "cuda"

    model = TextEmbedding(**kwargs)
    # Warm up first: the initial inference pays for the ONNX session and memory arena, which
    # would otherwise be charged to whichever batch happened to run first.
    model.embed([texts[0]], batch_size=1)

    started = time.perf_counter()
    vectors = [list(map(float, v)) for v in model.embed(list(texts), batch_size=args.batch_size)]
    elapsed = time.perf_counter() - started

    # Drop the model before the store phase so peak memory stays flat on a small machine.
    del model
    gc.collect()
    return elapsed, vectors


def _rmtree_best_effort(path: str) -> None:
    """Remove a temp dir, tolerating Windows' refusal to delete files still held open."""
    for attempt in range(5):
        try:
            shutil.rmtree(path)
            return
        except OSError:
            gc.collect()
            time.sleep(0.25 * (attempt + 1))
    shutil.rmtree(path, ignore_errors=True)


def bench_store(chunks, vectors, batch_size: int) -> float:
    import chromadb

    from slrag.retrieval.vector_store import chunk_metadata

    tmp = tempfile.mkdtemp(prefix="slrag-bench-")
    client = None
    try:
        client = chromadb.PersistentClient(path=tmp)
        collection = client.create_collection("bench", metadata={"hnsw:space": "cosine"})
        started = time.perf_counter()
        for start in range(0, len(chunks), batch_size):
            part = chunks[start : start + batch_size]
            collection.add(
                ids=[c.chunk_id for c in part],
                documents=[c.text for c in part],
                metadatas=[chunk_metadata(c) for c in part],
                embeddings=vectors[start : start + batch_size],
            )
        return time.perf_counter() - started
    finally:
        # Chroma holds the HNSW segment file open, and Windows will not delete an open file
        # (WinError 32). Drop our handle, collect, then delete with retries — and never let
        # cleanup failure destroy the measurement we already have.
        del client
        gc.collect()
        _rmtree_best_effort(tmp)


def main() -> int:
    args = parse_args()
    _ensure_slrag_importable()
    chunks = load_windows(Path(args.chunks), args.limit)
    texts = [c.text for c in chunks]
    count = len(texts)

    print(f"slrag bench — {count} windows from {args.chunks}")
    print(f"  avg window {sum(map(len, texts)) // count:,} chars (max {max(map(len, texts)):,})")
    print(
        f"  device={args.device}  threads={args.threads or 'auto'}  "
        f"embed batch={args.batch_size}  store batch={args.store_batch_size}\n"
    )

    embed_s, vectors = bench_embed(texts, args)
    print(f"  1. embedding only  {embed_s:7.1f}s  {count / embed_s:8.1f} windows/s")

    store_s = bench_store(chunks, vectors, args.store_batch_size)
    print(f"  2. chroma write    {store_s:7.1f}s  {count / store_s:8.1f} windows/s")

    total_s = embed_s + store_s
    blended = count / total_s
    embed_pct = 100.0 * embed_s / total_s
    print(f"\n  3. combined        {total_s:7.1f}s  {blended:8.1f} windows/s  <- what `slrag index` reports")
    print(f"\n  full corpus ({FULL_CORPUS:,} windows) projected: {FULL_CORPUS / blended / 60:.0f} min")

    print(f"\n  split: {embed_pct:.0f}% embedding / {100 - embed_pct:.0f}% Chroma write")
    if store_s >= embed_s:
        print("  -> Chroma's write path is the larger half. It is serial and CPU-only, so a GPU")
        print("     cannot speed it up. Try a bigger INDEX_BATCH_SIZE instead: set")
        print("     INDEX_BATCH_SIZE=1024 in .env and rerun with --store-batch-size 1024.")
    else:
        print(f"  -> Embedding dominates. A GPU can only improve that {embed_pct:.0f}% share,")
        print(f"     so the best case is about {100 / embed_pct:.1f}x overall — and only if")
        print("     EMBEDDING_DEVICE=cuda really engages CUDAExecutionProvider.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
