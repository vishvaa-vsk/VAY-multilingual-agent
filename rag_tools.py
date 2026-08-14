"""
rag_tools.py

Scoped RAG retriever tools — one per Nexatel knowledge base (see
chroma_setup.KB_COLLECTIONS / kb_docs/*.md). Each sub-agent gets ONLY its own
retriever tool bound into its tool-calling loop, per the mentor doc's "why
scoped RAG per sub-agent" rationale (precise retrieval, low hallucination,
independently testable).

Retrieval similarity of the best hit from the most recent call is recorded
on a RetrievalTracker so agent_graph.py's confidence-gate node can read it
without having to parse the LLM's tool-call output.
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.tools import tool

import content_manager
import chroma_setup


@dataclass
class RetrievalTracker:
    """Records the best similarity score seen across RAG tool calls in a turn,
    so the graph's confidence-gate node can act on it after the sub-agent runs."""
    last_score: float = 0.0
    called: bool = False

    def reset(self) -> None:
        self.last_score = 0.0
        self.called = False

    def record(self, score: float) -> None:
        self.called = True
        self.last_score = max(self.last_score, score)


def _format_hits(results: dict, tracker: RetrievalTracker) -> str:
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    if not docs:
        tracker.record(0.0)
        return "No relevant knowledge-base passages found."

    blocks = []
    for doc, meta, dist in zip(docs, metas, dists):
        sim = 1 - dist
        tracker.record(sim)
        heading = meta.get("heading") or meta.get("category") or "General"
        blocks.append(f"[relevance={sim:.2f} | {heading}]\n{doc}")
    return "\n\n".join(blocks)


def _make_retriever(collection_name: str, tool_name: str, description: str, tracker: RetrievalTracker):
    def _retriever(query: str) -> str:
        results = content_manager.read(query, n_results=3, collection_name=collection_name)
        return _format_hits(results, tracker)

    _retriever.__name__ = tool_name
    _retriever.__doc__ = description
    return tool(tool_name)(_retriever)


def build_billing_rag_tool(tracker: RetrievalTracker):
    """Billing-Policy RAG — tariff rules, late-fee/roaming policy, charge glossary, refund rules."""
    return _make_retriever(
        chroma_setup.KB_COLLECTIONS["billing_policy"],
        "search_billing_policy",
        "Search Nexatel's Billing-Policy knowledge base (tariff rules, late fees, roaming "
        "policy, charge glossary, refund rules) to ground an answer about a bill or charge.",
        tracker,
    )


def build_product_rag_tool(tracker: RetrievalTracker):
    """Product-Catalog RAG — current plans, prices, add-ons, offer terms & eligibility."""
    return _make_retriever(
        chroma_setup.KB_COLLECTIONS["product_catalog"],
        "search_product_catalog",
        "Search Nexatel's Product-Catalog knowledge base (current plans, prices, add-ons, "
        "offer terms & eligibility) to ground a plan recommendation or comparison.",
        tracker,
    )


def build_support_rag_tool(tracker: RetrievalTracker):
    """Support-KB / FAQ RAG — troubleshooting guides, known-issue articles, SLA/complaint policy."""
    return _make_retriever(
        chroma_setup.KB_COLLECTIONS["support_faq"],
        "search_support_kb",
        "Search Nexatel's Support-KB / FAQ knowledge base (troubleshooting guides, known "
        "issues, SLA and complaint policy) to ground a complaint or troubleshooting answer.",
        tracker,
    )


def build_technical_rag_tool(tracker: RetrievalTracker):
    """Technical-KB RAG — device/APN setup, SIM/eSIM guides, coverage FAQs."""
    return _make_retriever(
        chroma_setup.KB_COLLECTIONS["technical_kb"],
        "search_technical_kb",
        "Search Nexatel's Technical-KB knowledge base (device/APN setup, SIM/eSIM guides, "
        "coverage FAQs) to ground a coverage or technical-settings answer.",
        tracker,
    )


def compliance_policy_search(query: str, n_results: int = 3) -> str:
    """Direct (non-tool) helper used by the guardrail layer to pull mandated consent
    scripts / do-don't-say rules from the Compliance/Policy RAG before finalizing a
    reply that touches a sensitive action."""
    tracker = RetrievalTracker()
    results = content_manager.read(
        query, n_results=n_results, collection_name=chroma_setup.KB_COLLECTIONS["compliance_policy"]
    )
    return _format_hits(results, tracker)
