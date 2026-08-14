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

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()  # picks up .env in the current/parent directory (GROQ_API_KEY, GROQ_MODEL)

from vay.rag.retriever import RetrievalTracker
from vay.tools.session import SessionContext, confirm_pending_action


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
    AFFIRMATION_PATTERN,
    AGENT_NAMES,
    CLARIFY_TEMPLATES,
    CLOSING_FALLBACK_TEMPLATES,
    DEFAULT_MIN_SIMILARITY,
    DEFAULT_NLU_CONFIDENCE,
    HANDOFF_MESSAGE_TEMPLATES,
    HUMAN_REQUEST_PATTERNS,
    NEGATION_PATTERN,
    ORCHESTRATOR_SYSTEM_PROMPT,
    PENDING_ACTION_ROUTE,
    SUBAGENT_SYSTEM_PROMPT_TEMPLATE,
    TOOL_LOOP_FAILURE_TEMPLATES,
    UNCLEAR_ESCALATION_THRESHOLD,
    VALID_ROUTES,
    _llm,
    extract_json,
    localized,
    run_tool_agent,
)



def orchestrator_node(state: GraphState) -> GraphState:
    llm = _llm()
    messages = [SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT)]
    messages.extend(state.get("conversation_history", []))
    messages.append(
        HumanMessage(
            content=f"[caller language preference: {state['language']}]\n{state['transcript']}"
        )
    )

    try:
        raw = llm.invoke(messages).content
    except Exception as e:
        print(f"  [Orchestrator LLM call failed: {e}]")
        raw = None

    if state.get("show_debug") and raw is not None:
        print(f"  [Orchestrator raw LLM output] {raw.strip()[:500]}")

    parsed = extract_json(raw) if raw else None
    if parsed is None:
        # Fail safe: treat as unclear/low-confidence rather than guessing a route.
        parsed = {
            "language": state["language"],
            "intent": "unclear",
            "route": "unclear",
            "normalized_query": state["transcript"],
            "entities": {},
            "confidence": 0.0,
            "sensitive": False,
            "call_end_requested": False,
        }

    route = parsed.get("route") if parsed.get("route") in VALID_ROUTES else "unclear"
    confidence = (
        float(parsed.get("confidence", 0.0))
        if isinstance(parsed.get("confidence"), (int, float))
        else 0.0
    )
    sensitive = bool(parsed.get("sensitive", False))

    # A bare "yes"/"no" answering a pending consent script is easy for the orchestrator LLM
    # to misroute/mark low-confidence -- if one is waiting, force the route back to its
    # owning sub-agent so the code-enforced confirmation check in _run_subagent actually runs.
    session = state.get("session")
    forced_by_pending_action = bool(session and session.pending_action)
    if forced_by_pending_action:
        route = PENDING_ACTION_ROUTE.get(session.pending_action.get("tool"), route)
        sensitive = False
        confidence = 1.0

    # Per the solution doc's Layer 4 hand-off triggers (sensitive intent, explicit human
    # request, or -- for merely unclear/low-confidence turns -- REPEATED failure to
    # understand), decide here whether an unclear turn should get a human-free clarify
    # re-prompt (route_after_orchestrator -> "clarify") or actually escalate. A single
    # garbled/off-topic/rude utterance is not by itself a reason to consume a live agent's
    # time -- see tools.SessionContext.consecutive_unclear.
    unclear_escalate = False
    is_unclear = route == "unclear" or confidence < DEFAULT_NLU_CONFIDENCE
    explicit_human_request = bool(HUMAN_REQUEST_PATTERNS.search(state["transcript"]))
    if session and not forced_by_pending_action:
        if sensitive or explicit_human_request:
            unclear_escalate = True
            session.consecutive_unclear = 0
        elif is_unclear:
            session.consecutive_unclear += 1
            unclear_escalate = session.consecutive_unclear >= UNCLEAR_ESCALATION_THRESHOLD
        else:
            session.consecutive_unclear = 0  # a turn we understood -- reset the streak

    handoff_reason = ""
    if sensitive:
        handoff_reason = "Sensitive intent detected (billing dispute, cancellation, fraud/security, or distress)."
    elif explicit_human_request:
        handoff_reason = "Customer explicitly asked for a human agent."
    elif unclear_escalate:
        handoff_reason = f"Unclear/low-confidence for {session.consecutive_unclear if session else '?'} consecutive turns despite a clarify re-prompt."

    if state.get("show_debug"):
        print("  [Orchestrator JSON]")
        print(" ", json.dumps(parsed, indent=2, ensure_ascii=False).replace("\n", "\n  "))

    return {
        "intent": parsed.get("intent", "unclear"),
        "entities": parsed.get("entities") or {},
        "normalized_query": parsed.get("normalized_query") or state["transcript"],
        "nlu_confidence": confidence,
        "sensitive": sensitive,
        "route": route,
        "unclear_escalate": unclear_escalate,
        "handoff_reason": handoff_reason,
        "call_end_requested": bool(parsed.get("call_end_requested", False)),
        # Prefer the utterance-detected language for TTS if the LLM gave one, else keep session default.
        "language": parsed.get("language") or state["language"],
    }


def _run_subagent(state: GraphState, route: str, tools_builder, rag_tool_builder) -> GraphState:
    # session is created ONCE per call in main() and threaded through state every turn, so
    # pending_action/verified/escalation state survives across turns (see tools.py's design
    # note on why consent must be enforced in code, not just prompted for).
    session = state.get("session") or SessionContext(
        phone_number=state["phone_number"], verified=True
    )
    session.language = state["language"]

    # Code-enforced consent gate: if a sensitive action is awaiting confirmation from a
    # PRIOR turn, resolve it now from the customer's own transcript -- never from the LLM.
    if session.pending_action:
        said_yes = bool(
            AFFIRMATION_PATTERN.search(state["transcript"])
        ) and not NEGATION_PATTERN.search(state["transcript"])
        outcome = confirm_pending_action(session, customer_said_yes=said_yes)
        if outcome is not None:
            return {"draft_reply": outcome, "retrieval_score": 1.0}

    tracker = RetrievalTracker()
    domain_tools = tools_builder(session)
    rag_tool = rag_tool_builder(tracker)
    llm = _llm()

    reply = run_tool_agent(
        llm,
        domain_tools + [rag_tool],
        SUBAGENT_SYSTEM_PROMPT_TEMPLATE.format(
            agent_name=AGENT_NAMES[route],
            phone_number=state["phone_number"],
            language=state["language"],
        ),
        state["normalized_query"],
        state.get("conversation_history", []),
        language=state["language"],
        show_debug=state.get("show_debug", False),
    )

    result: GraphState = {
        "draft_reply": reply,
        "retrieval_score": tracker.last_score
        if tracker.called
        else 1.0,  # no RAG call needed -> don't penalize
    }
    if session.escalation_requested:
        result["handoff"] = True
        result["handoff_reason"] = session.escalation_reason or "Escalation requested by sub-agent."
    return result
