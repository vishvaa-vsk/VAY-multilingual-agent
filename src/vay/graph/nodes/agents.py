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

load_dotenv(override=True)  # picks up .env in the current/parent directory (GROQ_API_KEY, GROQ_MODEL)

from vay.rag.retriever import (
    build_billing_rag_tool,
    build_product_rag_tool,
    build_support_rag_tool,
    build_technical_rag_tool,
)
from vay.tools.billing import build_billing_tools
from vay.tools.complaints import build_complaints_tools
from vay.tools.coverage import build_coverage_tools
from vay.tools.plans import build_plans_tools


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


from vay.graph.nodes.orchestrator import _run_subagent
from vay.graph.state import GraphState


def billing_node(state: GraphState) -> GraphState:
    return _run_subagent(state, "billing", build_billing_tools, build_billing_rag_tool)


def plans_node(state: GraphState) -> GraphState:
    return _run_subagent(state, "plans", build_plans_tools, build_product_rag_tool)


def complaints_node(state: GraphState) -> GraphState:
    return _run_subagent(state, "complaints", build_complaints_tools, build_support_rag_tool)


def coverage_node(state: GraphState) -> GraphState:
    return _run_subagent(state, "coverage", build_coverage_tools, build_technical_rag_tool)
