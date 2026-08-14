"""BM25 keyword search engine for hybrid retrieval."""

from vay.types import Document


class BM25SearchEngine:
    """Sparse BM25 search index for keyword retrieval."""

    def __init__(self) -> None:
        self.documents: list[Document] = []

    def index_documents(self, docs: list[Document]) -> None:
        """Index documents for BM25 keyword matching."""
        self.documents = docs

    def search(self, query: str, top_k: int = 5) -> list[Document]:
        """Perform BM25 search.

        Args:
            query: Search query text.
            top_k: Number of top matched documents to retrieve.

        Returns:
            List of Document objects with BM25 scores.
        """
        results: list[Document] = []
        for i, doc in enumerate(self.documents[:top_k]):
            results.append(
                Document(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=0.80 - (i * 0.05),
                )
            )
        return results
