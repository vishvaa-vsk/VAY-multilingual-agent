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
    python scripts/build_kb.py    # one-time: build the 5 Nexatel RAG collections (idempotent
                                   # upsert into ./chroma_db -- safe to re-run)
    python scripts/manage_db.py   # one-time: create + seed the mock customer database
                                   # (./src/vay/tools/nexatel_customers.db -- only seeds if
                                   # empty; add --reset to wipe and reseed)
    set GROQ_API_KEY=your_key_here   (Windows)  /  export GROQ_API_KEY=...  (bash)

    Neither this script nor run_voice.py builds or seeds these stores themselves -- they
    only read from whatever already exists on disk, so the two commands above are run
    once (or after a deliberate --reset), not on every launch.

USAGE
-----
    python agent_graph.py
    python agent_graph.py --show_debug
    python agent_graph.py --min_similarity 0.3 --max_history_turns 8
"""

from __future__ import annotations

import argparse
import os
import re

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()  # picks up .env in the current/parent directory (GROQ_API_KEY, GROQ_MODEL)

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


from vay.graph.utils import localized, trim_history

from vay.graph.state import GraphState
from vay.graph.workflow import build_graph
from vay.tools.session import SessionContext


def _prompt_phone_number() -> str:
    while True:
        phone = input("Caller phone number (10 digits): ").strip()
        if re.fullmatch(r"\d{10}", phone):
            return phone
        print("  Please enter exactly 10 digits.")


def main():
    parser = argparse.ArgumentParser(
        description="Nexatel orchestrator + sub-agents voice-assistant loop."
    )
    parser.add_argument(
        "--min_similarity",
        type=float,
        default=DEFAULT_MIN_SIMILARITY,
        help="Confidence gate threshold on a sub-agent's best RAG hit.",
    )
    parser.add_argument("--max_history_turns", type=int, default=DEFAULT_MAX_HISTORY_TURNS)
    parser.add_argument(
        "--show_debug", action="store_true", help="Print orchestrator JSON and tool calls."
    )
    parser.add_argument(
        "--language", default=None, help="Skip the language prompt; use this ISO 639-1 code."
    )
    parser.add_argument(
        "--phone", default=None, help="Skip the phone-number prompt; use this 10-digit number."
    )
    args = parser.parse_args()

    if not GROQ_API_KEY:
        print("ERROR: GROQ_API_KEY environment variable is not set.")
        return

    graph = build_graph()

    print(
        "Nexatel Voice Assistant — one continuous call. Type a transcribed customer utterance below."
    )
    print("(dev shortcut: type 'exit' to kill the script without a proper call-ending flow)\n")

    phone_number = (
        args.phone if args.phone and re.fullmatch(r"\d{10}", args.phone) else _prompt_phone_number()
    )
    language = (
        args.language or input("Caller language code (e.g. en, hi, ta): ").strip().lower() or "en"
    )

    # One SessionContext for the whole call -- carries identity/consent state (pending_action,
    # verified, escalation) across turns. verified=True stands in for real identity
    # verification in this mock system (see tools.py's module docstring).
    session = SessionContext(phone_number=phone_number, verified=True, language=language)
    conversation_history: list = []

    while True:
        try:
            user_text = input("User speaks (transcribed text): ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit"):
            break

        state: GraphState = {
            "phone_number": phone_number,
            "language": language,
            "transcript": user_text,
            "conversation_history": conversation_history,
            "show_debug": args.show_debug,
            "min_similarity": args.min_similarity,
            "session": session,
        }

        result = graph.invoke(state)
        reply = result.get("final_reply") or localized(
            HANDOFF_MESSAGE_TEMPLATES, result.get("language", language)
        )
        print(f"Assistant: {reply}\n")

        conversation_history.append(HumanMessage(content=user_text))
        conversation_history.append(AIMessage(content=reply))
        conversation_history = trim_history(conversation_history, args.max_history_turns)

        if result.get("call_end_requested"):
            print("--- Call ended by customer. Session terminated. ---")
            break
        if result.get("handoff"):
            print(
                f"--- Call transferred to a human agent ({result.get('handoff_reason', 'n/a')}). "
                f"Session terminated. See {HANDOFF_LOG_PATH} for the escalation packet. ---"
            )
            break


if __name__ == "__main__":
    main()
