"""
agent_graph.py
--------------
Nexatel orchestrator + sub-agents voice-assistant application (LangGraph).

Mirrors this part of the architecture diagram, across a full multi-turn call,
picking up exactly where the ASR/VAD/Language-ID stage leaves off — input to
this module is (transcript, language_code, phone_number), not raw audio:

  ORCHESTRATOR (Groq LLM, NLU + intent/entity extraction)
      --sensitive/unclear/low-confidence--> HUMAN HANDOFF (bypasses retrieval)
      --else--> routed to one of 4 SUB-AGENTS (Billing / Plans / Complaints / Coverage)
                    each: tool-calling loop over its own backend tools + its
                    OWN scoped RAG retriever (rag_tools.py)
      --> GUARDRAIL NODE (confidence gate + handoff-gate + PII/consent scan)
             --low confidence / explicit handoff--> HUMAN HANDOFF
             --else--> FINALIZE
      --> TTS (tts.speak) --> loop for next utterance

Replaces voice_rag_pipeline.py as the entry point from the transcript stage
onward.

SETUP
-----
    pip install -r requirements.txt
    python build_kb.py          # build the 5 Nexatel RAG collections
    python customer_db.py       # seed the mock customer database
    set GROQ_API_KEY=your_key_here   (Windows)  /  export GROQ_API_KEY=...  (bash)

USAGE
-----
    python agent_graph.py
    python agent_graph.py --show_debug
    python agent_graph.py --min_similarity 0.3 --max_history_turns 8
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()  # picks up .env in the current/parent directory (GROQ_API_KEY, GROQ_MODEL)

from vay.tts import engine as tts


# Which sub-agent route owns each tool that can create a pending_action -- used to force
# routing back to the right sub-agent for a bare "yes"/"no" confirmation turn, since the
# orchestrator LLM can't reliably infer a route from a one-word reply alone.
PENDING_ACTION_ROUTE = {"changePlan": "plans"}

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")  # required, no hardcoded fallback (security fix)
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")

DEFAULT_MIN_SIMILARITY = 0.3  # confidence gate on the sub-agent's best RAG hit
DEFAULT_NLU_CONFIDENCE = 0.4  # orchestrator confidence floor before routing to a sub-agent
DEFAULT_MAX_HISTORY_TURNS = 6
MAX_TOOL_ITERATIONS = 6  # cap on tool-call round-trips per sub-agent turn (raised from 4
# -- a getBalance+getDueDate+RAG-search+sendPaymentLink turn was
# routinely hitting the old cap and forcing a rushed, ungrounded
# wrap-up reply)
HANDOFF_LOG_PATH = "handoff_log.jsonl"
UNCLEAR_ESCALATION_THRESHOLD = 2  # consecutive unclear/low-confidence turns before we give up
# asking the customer to clarify and hand off to a human

VALID_ROUTES = {"billing", "plans", "complaints", "coverage"}

# Fixed, deterministic per-language templates for the handful of replies that must be spoken
# WITHOUT going through a sub-agent LLM call (guardrail handoff, a hard tool-loop failure, the
# call-closing fallback, the pre-routing clarify re-prompt) -- same rationale as
# tools.CONSENT_TEMPLATES: these are safety/flow-control text, not something we want an LLM
# paraphrasing under a stressed or degraded call. Falls back to English for any language not
# yet covered here (add more entries rather than leaving callers silently in English forever --
# this dict, not "always English", was the actual bug: the *language* was already being
# detected correctly, but these fixed strings never varied with it).
HANDOFF_MESSAGE_TEMPLATES = {
    "en": "I want to make sure I get this right for you, and I'm not fully confident I can "
    "help with that myself right now. Let me connect you with a live Nexatel agent who "
    "can take it from here.",
    "hi": "मैं चाहता हूँ कि आपकी सही तरीके से मदद हो, और अभी मुझे पूरा भरोसा नहीं है कि मैं इसे "
    "खुद संभाल पाऊँगा। मैं आपको Nexatel के एक लाइव एजेंट से जोड़ रहा हूँ जो आगे मदद करेंगे।",
    "ta": "உங்களுக்குச் சரியாக உதவ விரும்புகிறேன், ஆனால் இதை என்னால் இப்போது சரியாகக் கையாள "
    "முடியுமா என்பதில் முழு நம்பிக்கை இல்லை. இதைத் தொடர்ந்து கவனிக்க ஒரு நேரடி Nexatel "
    "முகவரிடம் உங்களை இணைக்கிறேன்.",
}
TOOL_LOOP_FAILURE_TEMPLATES = {
    "en": "I'm not fully sure I can complete that here -- let me connect you with a human agent.",
    "hi": "मुझे पूरा यकीन नहीं है कि मैं इसे यहाँ पूरा कर पाऊँगा — मैं आपको एक मानव एजेंट से जोड़ता हूँ।",
    "ta": "இதை என்னால் இங்கு முழுமையாக முடிக்க முடியுமா என்று உறுதியாக இல்லை — நான் உங்களை ஒரு "
    "மனித முகவரிடம் இணைக்கிறேன்.",
}
CLOSING_FALLBACK_TEMPLATES = {
    "en": "Thank you for calling Nexatel. Have a great day!",
    "hi": "Nexatel को कॉल करने के लिए धन्यवाद। आपका दिन शुभ हो!",
    "ta": "Nexatel-ஐ அழைத்ததற்கு நன்றி. உங்கள் நாள் இனிதாக அமையட்டும்!",
}
CLARIFY_TEMPLATES = {
    "en": "Sorry, I didn't quite catch that. Could you tell me a little more about what you "
    "need help with -- for example your bill, your plan, a complaint, or network coverage?",
    "hi": "माफ़ कीजिए, मैं ठीक से समझ नहीं पाया। कृपया थोड़ा और बताएं कि आपको किस बारे में मदद "
    "चाहिए — जैसे आपका बिल, आपका प्लान, कोई शिकायत, या नेटवर्क कवरेज।",
    "ta": "மன்னிக்கவும், எனக்குச் சரியாகப் புரியவில்லை. உங்களுக்கு எதில் உதவி தேவை என்று கொஞ்சம் "
    "விளக்கமாகச் சொல்ல முடியுமா — உங்கள் பில், திட்டம், புகார், அல்லது நெட்வொர்க் கவரேஜ் "
    "போன்றவை?",
}


from vay.graph.state import GraphState
from vay.graph.utils import (
    CLARIFY_TEMPLATES,
    CLOSING_FALLBACK_TEMPLATES,
    DEFAULT_NLU_CONFIDENCE,
    HANDOFF_MESSAGE_TEMPLATES,
    HUMAN_REQUEST_PATTERNS,
    PII_LEAK_PATTERNS,
    SUBAGENT_SYSTEM_PROMPT_TEMPLATE,
    UNCERTAINTY_PATTERNS,
    _llm,
    localized,
    log_handoff,
)


def guardrail_node(state: GraphState) -> GraphState:
    """Layer 3 output guardrail: confidence gate + handoff-gate + a lightweight
    PII/consent scan, per kb_docs/compliance_policy.md."""
    if state.get("handoff"):
        return {}  # a sub-agent tool already requested escalation

    draft = state.get("draft_reply", "")
    min_similarity = state.get("min_similarity", DEFAULT_MIN_SIMILARITY)

    if state.get("retrieval_score", 0.0) < min_similarity:
        return {
            "handoff": True,
            "handoff_reason": "Low retrieval confidence on knowledge-base grounding.",
        }

    if HUMAN_REQUEST_PATTERNS.search(state["transcript"]) or HUMAN_REQUEST_PATTERNS.search(draft):
        return {"handoff": True, "handoff_reason": "Customer requested a human agent."}

    if UNCERTAINTY_PATTERNS.search(draft):
        return {
            "handoff": True,
            "handoff_reason": "Assistant signaled uncertainty in its draft reply.",
        }

    if PII_LEAK_PATTERNS.search(draft):
        return {
            "handoff": True,
            "handoff_reason": "Draft reply referenced sensitive credentials (guardrail block).",
        }

    return {"final_reply": draft}


def human_handoff_node(state: GraphState) -> GraphState:
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


def clarify_node(state: GraphState) -> GraphState:
    """A human-agent-free re-prompt for a turn the orchestrator couldn't understand/route
    confidently, used instead of an immediate human_handoff (see route_after_orchestrator) --
    per the solution doc's Layer 1 guidance ("ask to repeat" before escalating) and so a single
    garbled, off-topic, or rude utterance doesn't consume a live agent's time. Deliberately a
    fixed localized template, not an LLM call: there's nothing to ground yet this turn."""
    return {"final_reply": localized(CLARIFY_TEMPLATES, state["language"])}


def closing_node(state: GraphState) -> GraphState:
    llm = _llm()
    messages = [
        SystemMessage(
            content=SUBAGENT_SYSTEM_PROMPT_TEMPLATE.format(
                agent_name="call-closing assistant",
                phone_number=state["phone_number"],
                language=state["language"],
            )
        )
    ]
    messages.extend(state.get("conversation_history", []))
    messages.append(
        HumanMessage(
            content=f"Customer's latest message: {state['transcript']}\n\n"
            "The customer is ending the call. Reply with ONE brief, warm closing line "
            "thanking them for calling Nexatel -- no new information, no questions back."
        )
    )
    fallback = localized(CLOSING_FALLBACK_TEMPLATES, state["language"])
    try:
        reply = llm.invoke(messages).content.strip()
    except Exception:
        reply = fallback
    return {"final_reply": reply or fallback}


def tts_node(state: GraphState) -> GraphState:
    tts.speak(state.get("final_reply", ""), lang=state.get("language", "en"))
    return {}


# ---------------------------------------------------------------------------
# Routing (conditional edges)
# ---------------------------------------------------------------------------
def route_after_orchestrator(state: GraphState) -> str:
    if state.get("call_end_requested"):
        return "closing"
    if state.get("sensitive"):
        return "human_handoff"
    if state.get("route") == "unclear" or state.get("nlu_confidence", 0.0) < DEFAULT_NLU_CONFIDENCE:
        # Only actually escalate once unclear_escalate says so (repeated failure to
        # understand, or an explicit human request) -- otherwise give the customer one more
        # chance via a human-free clarify re-prompt. See orchestrator_node.
        return "human_handoff" if state.get("unclear_escalate") else "clarify"
    return state["route"]  # billing | plans | complaints | coverage


def route_after_guardrail(state: GraphState) -> str:
    return "human_handoff" if state.get("handoff") else "tts"


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
