"""
utils.py (nodes/utils.py)
-------------------------
Utility LangGraph nodes and routing functions for the Nexatel voice assistant.

Nodes:
- guardrail_node: confidence gate + handoff-gate + PII/consent scan
- human_handoff_node: logs and prepares the human handoff message
- warning_node: speaks a firm warning to aggressive callers (1st offence)
- clarify_node: asks the customer to repeat/clarify (unclear turns)
- closing_node: generates a warm closing line when the customer ends the call
- tts_node: speaks the final_reply via edge-tts

Routing:
- route_after_orchestrator: decides which node to visit after the orchestrator
- route_after_guardrail: human_handoff or tts
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(override=True)

from vay.tts import engine as tts

from vay.graph.state import GraphState
from vay.graph.utils import (
    CHITCHAT_TEMPLATES,
    CLARIFY_TEMPLATES,
    CLOSING_FALLBACK_TEMPLATES,
    DEFAULT_NLU_CONFIDENCE,
    HANDOFF_MESSAGE_TEMPLATES,
    HUMAN_REQUEST_PATTERNS,
    PII_LEAK_PATTERNS,
    UNCERTAINTY_PATTERNS,
    _llm,
    localized,
    log_handoff,
)
from vay.rag.retriever import compliance_policy_search


# ---------------------------------------------------------------------------
# Guardrail node
# ---------------------------------------------------------------------------

def guardrail_node(state: GraphState) -> GraphState:
    """Layer 3 output guardrail: confidence gate + handoff-gate + PII/consent scan.

    Changes from the original:
    - Uncertainty in the draft reply alone no longer triggers a handoff; it must
      ALSO have a low retrieval score. An LLM that says "I'm not sure" but pulled
      a high-confidence KB hit is expressing appropriate caveating, not failure.
    - Compliance policy is checked for any draft that touches a sensitive action
      keyword, so the KB guardrail rules are actually enforced.
    """
    # A sub-agent tool already requested escalation — short-circuit
    if state.get("handoff"):
        return {}

    draft = state.get("draft_reply", "")
    min_similarity = state.get("min_similarity", 0.3)
    retrieval_score = state.get("retrieval_score", 0.0)

    print(f"  [Guardrail] retrieval_score={retrieval_score:.2f} | min_similarity={min_similarity} | handoff_already={state.get('handoff')}")
    print(f"  [Guardrail] draft_reply (first 200): {draft[:200]}")

    # --- Confidence gate ---
    if retrieval_score < min_similarity:
        print(f"  [Guardrail] FAILED confidence gate ({retrieval_score:.2f} < {min_similarity}) -> HANDOFF")
        return {
            "handoff": True,
            "handoff_reason": f"Low retrieval confidence ({retrieval_score:.2f} < {min_similarity}).",
        }

    # --- Human request in the CUSTOMER's own transcript ---
    # BUGFIX: this used to ALSO scan the assistant's own draft reply for the same
    # pattern. Since the sub-agent prompt explicitly tells the LLM to say things like
    # "I can connect you to a human agent" when offering escalation as an option, that
    # phrase itself matched HUMAN_REQUEST_PATTERNS on the DRAFT and silently replaced an
    # otherwise good, specific, grounded answer with the generic handoff template -- i.e.
    # every sub-agent reply that so much as mentioned "human agent" was self-sabotaging.
    # A sub-agent that actually decides to escalate should do so via the escalateToHuman
    # tool (session.escalation_requested), which is already checked above via
    # state.get("handoff") -- not by pattern-matching its own prose.
    if HUMAN_REQUEST_PATTERNS.search(state["transcript"]):
        return {"handoff": True, "handoff_reason": "Customer requested a human agent."}

    # --- Uncertainty: only handoff if retrieval score is ALSO below a relaxed threshold ---
    # Previously, uncertainty alone (e.g. "I'm not fully sure") would immediately trigger
    # a handoff even when the RAG returned a high-confidence result. This was too aggressive.
    UNCERTAINTY_CONFIDENCE_CUTOFF = 0.5
    if UNCERTAINTY_PATTERNS.search(draft) and retrieval_score < UNCERTAINTY_CONFIDENCE_CUTOFF:
        return {
            "handoff": True,
            "handoff_reason": "Assistant signaled uncertainty with low retrieval confidence.",
        }

    # --- PII leak guard ---
    if PII_LEAK_PATTERNS.search(draft):
        return {
            "handoff": True,
            "handoff_reason": "Draft reply referenced sensitive credentials (guardrail block).",
        }

    # --- Compliance policy check for sensitive-action keywords ---
    CONSENT_TRIGGER_WORDS = re.compile(
        r"\b(change plan|payment link|cancel|switch plan|modify)\b", re.IGNORECASE
    )
    if CONSENT_TRIGGER_WORDS.search(draft):
        compliance_context = compliance_policy_search(
            "consent required actions plan change payment cancellation"
        )
        # If compliance KB says "must read consent script" and the draft doesn't contain
        # a confirmation ask, flag it — this is a lightweight check, not a full parse.
        if "consent" in compliance_context.lower() and "confirm" not in draft.lower():
            # Log but don't handoff — the sub-agent should have included a confirmation;
            # we just note it so it's visible in debug output.
            if state.get("show_debug"):
                print(
                    "  [guardrail: compliance policy recommends consent script for this action "
                    "— draft may be missing it]"
                )

    return {"final_reply": draft}


import re  # noqa: E402 (import after the function that uses it — kept here to avoid circular issues)


# ---------------------------------------------------------------------------
# Human handoff node
# ---------------------------------------------------------------------------

def human_handoff_node(state: GraphState) -> GraphState:
    """Log the handoff context and return the localized handoff message."""
    log_handoff(
        {
            "phone_number": state["phone_number"],
            "transcript": state["transcript"],
            "intent": state.get("intent"),
            "entities": state.get("entities"),
            "normalized_query": state.get("normalized_query"),
            "route": state.get("route"),
            "reason": state.get("handoff_reason", "unspecified"),
            "draft_reply_at_handoff": state.get("draft_reply"),
        }
    )
    return {
        "final_reply": localized(HANDOFF_MESSAGE_TEMPLATES, state.get("language", "en")),
        "handoff": True,
    }


# ---------------------------------------------------------------------------
# Warning node (aggressive / abusive caller — 1st offence)
# ---------------------------------------------------------------------------

def warning_node(state: GraphState) -> GraphState:
    """Set the final_reply to the pre-built warning text stored in state.

    The warning text is already localized by orchestrator_node (using the
    caller's detected language) and stored in state['warning_reply'].
    """
    warning_text = state.get("warning_reply", "")
    if not warning_text:
        # Fallback: shouldn't happen, but be safe
        warning_text = localized(
            {
                "en": "Please note that abusive language is not acceptable on this call. "
                      "Severe action will be taken if this continues."
            },
            state.get("language", "en"),
        )
    return {"final_reply": warning_text}


# ---------------------------------------------------------------------------
# Clarify node
# ---------------------------------------------------------------------------

def chitchat_node(state: GraphState) -> GraphState:
    """A fixed, non-LLM reply for acknowledgements/thanks/greetings with nothing
    actionable in them -- see ORCHESTRATOR_SYSTEM_PROMPT's chitchat routing guide.
    Deliberately separate from clarify_node: this turn WAS understood, so the reply
    doesn't ask the customer to repeat themselves."""
    return {"final_reply": localized(CHITCHAT_TEMPLATES, state["language"])}


def clarify_node(state: GraphState) -> GraphState:
    """A human-agent-free re-prompt for a turn the orchestrator couldn't understand/route
    confidently, used instead of an immediate human_handoff (see route_after_orchestrator) --
    per the solution doc's Layer 1 guidance (\"ask to repeat\" before escalating) and so a single
    garbled, off-topic, or rude utterance doesn't consume a live agent's time. Deliberately a
    fixed localized template, not an LLM call: there's nothing to ground yet this turn."""
    return {"final_reply": localized(CLARIFY_TEMPLATES, state["language"])}


# ---------------------------------------------------------------------------
# Closing node
# ---------------------------------------------------------------------------

def closing_node(state: GraphState) -> GraphState:
    # BUGFIX: route_after_orchestrator sends the 2nd-offence abusive-caller call-cut here too
    # ("Call was cut due to repeated abuse -- treat as closing so TTS plays the goodbye"), but
    # this node used to ALWAYS ask the LLM to generate a generic warm "thank you for calling"
    # line -- completely discarding the deterministic, hand-written CALL_CUT_TEMPLATES text
    # (orchestrator_node already built it into state["warning_reply"] specifically so the
    # customer is told WHY the call is ending, per core_utils.py's comment that this text
    # "cannot be side-stepped by a hallucinating model"). A customer being cut off for abuse was
    # instead hearing a cheerful, unrelated goodbye and never learning their call was terminated
    # for that reason. If warning_reply is set, this IS that forced-closing case -- speak it
    # verbatim, no LLM call needed (there's nothing to generate).
    if state.get("warning_reply"):
        return {"final_reply": state["warning_reply"]}

    llm = _llm()
    # TOKEN EFFICIENCY: this used to send the FULL SUBAGENT_SYSTEM_PROMPT_TEMPLATE (~2000+
    # tokens of tool-use rules, grounding/escalation guardrails, anti-repetition rules, etc.)
    # plus the entire trimmed conversation history just to generate a single generic "thanks
    # for calling" line -- closing_node calls no tools and touches no account data, so none of
    # that applies. That token cost was being paid on every call-ending turn and contributing to
    # hitting Groq's per-minute token cap (see run.log 429s). A minimal, closing-specific prompt
    # with just the last transcript line covers this task just as well for a fraction of the cost.
    messages = [
        SystemMessage(
            content=(
                f'You are a warm Nexatel Communications voice-support agent ending a call in '
                f'language "{state["language"]}". Reply with ONE brief, natural closing line '
                f'thanking the customer for calling Nexatel -- no new information, no questions, '
                f'no markdown. Write entirely in "{state["language"]}".'
            )
        ),
        HumanMessage(content=f"Customer's last message: {state['transcript']}"),
    ]
    fallback = localized(CLOSING_FALLBACK_TEMPLATES, state["language"])
    try:
        reply = llm.invoke(messages).content.strip()
    except Exception:
        reply = fallback
    return {"final_reply": reply or fallback}


# ---------------------------------------------------------------------------
# TTS node
# ---------------------------------------------------------------------------

def tts_node(state: GraphState) -> GraphState:
    """Speak the final reply via edge-tts in the caller's detected language."""
    if os.environ.get("STREAMLIT_UI") != "1":
        tts.speak(state.get("final_reply", ""), lang=state.get("language", "en"))
    return {}


# ---------------------------------------------------------------------------
# Routing (conditional edges)
# ---------------------------------------------------------------------------

def route_after_orchestrator(state: GraphState) -> str:
    """Decide the next node after the orchestrator runs.

    Priority order:
    1. Call cut (2nd aggressive offence) → closing (final word + TTS)
    2. Warning (1st aggressive offence) → warning node
    3. Normal call end requested → closing
    4. Sensitive intent → human_handoff
    4b. LLM unavailable (every failover candidate failed this turn) → human_handoff
    5. Chitchat (understood, nothing actionable) → chitchat node
    6. Unclear/low-confidence (with escalation) → human_handoff
    7. Unclear/low-confidence (first time) → clarify
    8. Valid route → billing / plans / complaints / coverage
    """
    # Call was cut due to repeated abuse — treat as closing so TTS plays the goodbye
    if state.get("call_end_requested") and state.get("warning_reply"):
        return "closing"

    # Warning reply set (1st aggressive offence, no cut yet)
    if state.get("warning_reply"):
        return "warning"

    if state.get("call_end_requested"):
        return "closing"

    if state.get("sensitive"):
        return "human_handoff"

    # Every LLM failover candidate failed outright this turn -- there's no real NLU result to
    # route on (route/confidence/sensitive are just the "unclear" fallback defaults). Escalate
    # honestly instead of clarify_node's "I didn't catch that", which would be false here.
    if state.get("llm_unavailable"):
        return "human_handoff"

    if state.get("route") == "chitchat":
        return "chitchat"

    if state.get("route") == "unclear" or state.get("nlu_confidence", 0.0) < DEFAULT_NLU_CONFIDENCE:
        return "human_handoff" if state.get("unclear_escalate") else "clarify"

    return state["route"]  # billing | plans | complaints | coverage


def route_after_guardrail(state: GraphState) -> str:
    return "human_handoff" if state.get("handoff") else "tts"
