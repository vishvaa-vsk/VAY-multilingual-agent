"""Tests for hybrid RAG retriever."""

from vay.rag.retriever import HybridRetriever


def test_hybrid_retriever_initialization() -> None:
    retriever = HybridRetriever()
    assert retriever.confidence_threshold >= 0.75
    res = retriever.retrieve("bill amount query")
    assert res.query == "bill amount query"
