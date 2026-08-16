"""BM25 keyword search engine for hybrid retrieval.

NOTE: the live RAG query path (rag/manager_read.py -> rag/hybrid.py) does NOT
use this class -- it builds its own rank_bm25 index directly against ChromaDB
collections for efficiency/caching reasons. This class is kept as a small,
standalone, CORRECT BM25 engine (previously it was a stub that ignored the
query entirely and returned documents in input order with a fabricated
score) for any caller that wants BM25 search over an arbitrary in-memory
Document list without a ChromaDB collection backing it.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from vay.types import Document

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25SearchEngine:
    """Sparse BM25 search index for keyword retrieval."""

    def __init__(self) -> None:
        self.documents: list[Document] = []
        self._bm25: BM25Okapi | None = None

    def index_documents(self, docs: list[Document]) -> None:
        """Index documents for BM25 keyword matching."""
        self.documents = docs
        tokenized = [_tokenize(d.content) for d in docs]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Perform a real BM25 keyword search (term frequency / inverse
        document frequency over the indexed documents), returning the top_k
        highest-scoring documents with their actual BM25 score.

        Args:
            query: Search query text.
            top_k: Number of top matched documents to retrieve.

        Returns:
            List of Document objects with BM25 scores, highest first.
        """
        if not self.documents or self._bm25 is None:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: -scores[i])[:top_k]

        results: list[Document] = []
        for i in ranked:
            if scores[i] <= 0:
                continue
            src = self.documents[i]
            results.append(
                Document(
                    id=src.id,
                    content=src.content,
                    metadata=src.metadata,
                    score=float(scores[i]),
                )
            )
        return results
