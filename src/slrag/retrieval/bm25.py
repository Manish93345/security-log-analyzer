"""Lexical retrieval — BM25 over each window's ``search_text``.

Why keep a lexical retriever when we already have embeddings? Because security questions are
full of *rare literal tokens* that a 384-dimension sentence embedding blurs away: an IP
address, an IAM action name, a filesystem path, a hostname. Ask "failed sudo commands on the
web server" and a dense retriever will happily return any window that is *about*
authentication; BM25 puts the window that literally contains ``sudo`` and ``Failed password``
on top. Fusing the two gives a candidate pool better than either alone.

Tokenisation is tuned for logs rather than prose:

* case-folded, so ``ConsoleLogin`` and ``consolelogin`` both match;
* CamelCase is split (``ConsoleLogin`` -> ``console login``) so a plain-English question can
  hit an AWS action name;
* path / host separators are split (``/usr/bin/systemctl`` -> ``usr bin systemctl``);
* IPv4 literals are kept whole — splitting ``203.0.113.44`` into ``203 113 44`` would make
  every address match every other address, which is the one thing BM25 must not do here.

``rank-bm25`` is imported lazily, so the unit tests and ``slrag version`` work before the
dependencies are installed.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

#: Tokens keep ``.``/``:``/``/``/``-``/``@``/``+`` inside them, so IPv4 addresses, ARNs,
#: paths, timestamps and email-shaped principals survive as one token each.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:\-/@+]*")

#: CamelCase / PascalCase / acronym splitter: ``ConsoleLogin`` -> Console, Login.
_CAMEL_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z])|[A-Z]?[a-z]+|[A-Z]+|\d+")

#: Separators we additionally split on, so ``systemctl`` is reachable inside a full path.
_SEPARATOR_RE = re.compile(r"[._:\-/@+]+")

_IPV4_RE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")

#: Sub-tokens shorter than this are noise (``s3`` and ``ec2`` must survive, ``a`` must not).
MIN_SUBTOKEN_LEN = 2


def tokenize(text: str) -> list[str]:
    """Split a log line or a question into BM25 terms.

    Repeated tokens are kept (BM25 uses term frequency), so ``"failed failed"`` yields two
    ``failed`` tokens.
    """
    tokens: list[str] = []
    if not text:
        return tokens

    for raw in _TOKEN_RE.findall(text):
        lowered = raw.lower()
        tokens.append(lowered)

        if _IPV4_RE.fullmatch(raw):
            continue  # never split an address apart — see the module docstring

        if any(char.isupper() for char in raw):
            for part in _CAMEL_RE.findall(raw):
                candidate = part.lower()
                if len(candidate) >= MIN_SUBTOKEN_LEN and candidate != lowered:
                    tokens.append(candidate)

        if _SEPARATOR_RE.search(raw):
            for part in _SEPARATOR_RE.split(raw):
                candidate = part.lower()
                if len(candidate) >= MIN_SUBTOKEN_LEN and candidate != lowered:
                    tokens.append(candidate)

    return tokens


def _intern(tokens: list[str], cache: dict[str, str]) -> list[str]:
    """Deduplicate token *objects* across documents.

    A 38k-window corpus is mostly the same few hundred words repeated. Sharing one string
    object per distinct token cuts the index's memory footprint substantially at no cost.
    """
    return [cache.setdefault(token, token) for token in tokens]


@dataclass(frozen=True, slots=True)
class BM25Hit:
    """One lexical hit: window id, BM25 score, and 1-based rank."""

    chunk_id: str
    score: float
    rank: int


class BM25Index:
    """In-memory BM25 over incident windows.

    Build it once and keep it: construction tokenises the whole corpus (the expensive part),
    while a query only scores the terms it actually contains.
    """

    def __init__(self, *, k1: float = 1.5, b: float = 0.75, max_tokens: int = 512) -> None:
        self.k1 = float(k1)
        self.b = float(b)
        self.max_tokens = int(max_tokens)
        self._ids: list[str] = []
        self._bm25 = None

    # ------------------------------------------------------------------ build
    @classmethod
    def from_pairs(
        cls,
        pairs: Iterable[tuple[str, str]],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        max_tokens: int = 512,
    ) -> BM25Index:
        """Build from ``(chunk_id, text)`` pairs."""
        return cls(k1=k1, b=b, max_tokens=max_tokens).build(pairs)

    @classmethod
    def from_chunks(
        cls,
        chunks: Iterable[object],
        *,
        text_attr: str = "search_text",
        k1: float = 1.5,
        b: float = 0.75,
        max_tokens: int = 512,
    ) -> BM25Index:
        """Build from objects carrying ``chunk_id`` plus ``search_text`` (falls back to ``text``)."""
        return cls(k1=k1, b=b, max_tokens=max_tokens).build(

                (
                    str(chunk.chunk_id),
                    str(getattr(chunk, text_attr, "") or getattr(chunk, "text", "")),
                )
                for chunk in chunks

        )

    def build(self, pairs: Iterable[tuple[str, str]]) -> BM25Index:
        """Tokenise every document and fit the IDF table. Returns ``self`` for chaining."""
        from rank_bm25 import BM25Okapi

        cache: dict[str, str] = {}
        ids: list[str] = []
        corpus: list[list[str]] = []

        for chunk_id, text in pairs:
            tokens = _intern(tokenize(text), cache)
            if self.max_tokens > 0:
                tokens = tokens[: self.max_tokens]
            ids.append(str(chunk_id))
            corpus.append(tokens)

        self._ids = ids
        # ``any()`` guards the degenerate corpus where every document is empty: BM25Okapi
        # would divide by a zero average document length.
        self._bm25 = BM25Okapi(corpus, k1=self.k1, b=self.b) if any(corpus) else None
        if self._bm25 is not None:
            # get_scores()/get_top_n() only read doc_freqs, doc_len, idf, avgdl and corpus_size,
            # so the raw token lists can be released (rank-bm25 is pinned to 0.2.2).
            self._bm25.corpus = None
        return self

    # ----------------------------------------------------------------- search
    def search(self, query: str, k: int = 50) -> list[BM25Hit]:
        """Return the ``k`` best windows that actually contain at least one query term."""
        if self._bm25 is None or not self._ids or k <= 0:
            return []
        terms = tokenize(query)
        if not terms:
            return []

        scores = self._bm25.get_scores(terms)
        query_terms = set(terms)

        # A window is a lexical candidate only if it literally contains a query term.
        # Filtering on "score > 0" instead would be wrong twice over: on a small corpus every
        # IDF can come out negative (so a *perfect* match scores below zero), and even in a
        # large corpus a common word can. RRF only consumes ranks, so the score is needed for
        # ordering alone — never as a relevance gate.
        candidates: list[tuple[float, str]] = []
        for index, document_terms in enumerate(self._bm25.doc_freqs):
            if query_terms.isdisjoint(document_terms):
                continue
            candidates.append((float(scores[index]), self._ids[index]))

        candidates.sort(key=lambda item: (-item[0], item[1]))
        return [
            BM25Hit(chunk_id=chunk_id, score=score, rank=position + 1)
            for position, (score, chunk_id) in enumerate(candidates[:k])
        ]

    # ------------------------------------------------------------------ dunder
    def __len__(self) -> int:
        return len(self._ids)

    @property
    def vocabulary_size(self) -> int:
        return len(self._bm25.idf) if self._bm25 is not None else 0


__all__ = ["MIN_SUBTOKEN_LEN", "BM25Hit", "BM25Index", "tokenize"]
