"""
tools.py

Backend "API" tools for the four Nexatel sub-agents, per the mentor doc's
tool lists (section 3.2). Backed by the mock SQLite store in customer_db.py.

Design note — identity/session binding:
Per the Compliance/Policy guardrail rules (kb_docs/compliance_policy.md),
the caller's phone number is session context established once at the start
of a call, not something the LLM should be able to invent or override
mid-conversation. So instead of exposing `phone_number` as an LLM-fillable
tool argument, each `build_*_tools(session)` factory below closes over a
`SessionContext` and returns LangChain @tool-wrapped closures bound to that
one caller — the LLM only supplies the *content* arguments (plan id, ticket
id, pincode, etc.), never the identity.

Sensitive actions (`changePlan`, `sendPaymentLink`) refuse to execute unless
`session.verified` is True (set by the graph's identity-verification guardrail
step) and instead return a message telling the caller/agent to hand off.

Design note — consent is enforced in CODE, not just prompted for:
Live testing showed the LLM cannot be trusted to reliably read back the
mandated consent script (kb_docs/compliance_policy.md) and wait for an
affirmative "yes" before calling a sensitive tool -- a small model will
happily call changePlan()/sendPaymentLink() in the very same turn it was
first asked, with no confirmation at all. So these two tools are two-phase
at the code level: the FIRST call only stages the action on
`session.pending_action` and returns a confirmation script for the LLM to
read back; nothing is written to the database yet. The action only actually
executes when `confirm_pending_action()` is called by the graph after
verifying the *customer's own next-turn transcript* contains an affirmation
-- the LLM's own say-so is never sufficient.
"""

from __future__ import annotations

import random
import string
from dataclasses import dataclass
from datetime import date

import vay.tools.db_queries as customer_db



# ---------------------------------------------------------------------------
# Session context — identity binding + cross-turn state for tool calls.
# One instance is created per CALL (not per turn) so pending_action and
# verified/escalation state survive across the whole conversation.
# ---------------------------------------------------------------------------
@dataclass
class SessionContext:
    phone_number: str
    verified: bool = False
    language: str = "en"
    # Customer's registered language from the DB (the `lang` column set at account creation).
    # Loaded once at call startup via load_customer_preferred_language() below.
    # Used by ASRRouter as a soft hint when Whisper's auto-detection confidence is too low
    # to trust (e.g. Tamil speech mis-detected as Indonesian at 0.43 confidence) — see
    # run_voice.py's on_asr_result for how this feeds into override_language.
    preferred_language: str = "en"
    escalation_requested: bool = False  # set True by escalateToHuman()
    escalation_reason: str = ""
    pending_action: dict | None = (
        None  # {"tool": "changePlan"|"sendPaymentLink", "args": {...}, "summary": str}
    )
    # Consecutive turns the orchestrator failed to route (route="unclear" or low NLU
    # confidence). Reset to 0 the moment a turn is understood. agent_graph.py uses this to
    # give the customer one gentle "could you clarify?" re-prompt before ever handing off to
    # a human for a merely-unclear utterance -- see agent_graph.route_after_orchestrator().
    consecutive_unclear: int = 0
    # Number of turns the caller has used aggressive/abusive language. On the first offence a
    # warning is spoken in their language; on the second the call is terminated.
    aggressive_count: int = 0
    # Route used in the PREVIOUS turn (billing/plans/complaints/coverage/chitchat/unclear) --
    # used by _run_subagent to detect a domain switch and trim stale tool messages. Lives here,
    # not in GraphState, for the same reason aggressive_count and consecutive_unclear do: the
    # caller (scripts/run_assistant.py) builds a FRESH GraphState dict every turn and never
    # threads the previous graph.invoke() result back in -- session is the only object that
    # actually survives across turns. See rag-tts-evaluvation.md for the bug this caused when
    # aggressive_count/previous_route were (incorrectly) read from state instead of session.
    last_route: str = ""


def load_customer_preferred_language(phone_number: str) -> str:
    """Look up the customer's registered language preference from the DB.

    Returns the ISO 639-1 language code stored in the ``lang`` column of the
    customers table (e.g. ``"ta"`` for Tamil, ``"hi"`` for Hindi), or ``"en"``
    if the customer is not found or the lookup fails.

    This is called ONCE at call startup (``run_voice.py``) and stored on
    ``SessionContext.preferred_language``.  The ASR router uses it as a
    fallback hint when Whisper auto-detects a Tier-2 language with low
    confidence (< 0.5), which is a common failure mode for Indian languages
    in short/noisy utterances (e.g. Tamil speech mistakenly classified as
    Indonesian at confidence 0.43).
    """
    try:
        conn = customer_db._connect()
        row = conn.execute(
            "SELECT language_pref FROM customers WHERE phone_number=?", (phone_number,)
        ).fetchone()
        if row and row["language_pref"]:
            return row["language_pref"]
    except Exception:
        pass
    return "en"

SENSITIVE_DENIAL = (
    "This action requires identity verification that hasn't been completed in this "
    "session. Please connect the customer to a human agent to proceed."
)

# Consent-script frame per language, deliberately fixed/hand-written (not LLM-generated —
# see tools.py's module docstring on why consent can't be trusted to the LLM). The customer
# is explicitly told to answer with the literal English word "yes"/"no" regardless of which
# language the rest of the sentence is in, so agent_graph.py's confirmation check only ever
# needs to match those two literal tokens instead of enumerating affirmation phrases across
# every supported language. {summary} is the plan/price description (kept in English/brand
# terms, same convention as kb_docs/product_catalog.md).
CONSENT_TEMPLATES = {
    "en": 'To confirm, I\'ll {summary}. Please say "yes" to confirm, or "no" to cancel.',
    "hi": 'पुष्टि के लिए बता रहा हूँ: मैं {summary}। पुष्टि करने के लिए अंग्रेज़ी में "yes" बोलें, '
    'या रद्द करने के लिए "no" बोलें।',
    "ta": 'உறுதிசெய்ய: நான் {summary}. உறுதிசெய்ய ஆங்கிலத்தில் "yes" என்று சொல்லுங்கள், '
    'அல்லது ரத்து செய்ய "no" என்று சொல்லுங்கள்.',
}


def consent_script(language: str, summary: str) -> str:
    """Build the fixed-template consent question for a sensitive action, in the caller's
    language, always ending with the literal English yes/no instruction."""
    # Deferred import: graph.core_utils has no dependency back on tools.session, so this
    # isn't a real circular import, but importing it at module load time here -- while
    # graph/__init__.py's own import chain (workflow -> nodes.agents -> nodes.orchestrator)
    # is what pulls this very module in in the first place -- would make the import order
    # fragile. Importing inside the function sidesteps that entirely.
    from vay.graph.core_utils import localized

    template = localized(CONSENT_TEMPLATES, language)
    return template.format(summary=summary)


# These three outcomes of confirm_pending_action() bypass the LLM entirely (same rationale
# as CONSENT_TEMPLATES above -- a real DB mutation's confirmation text must be deterministic,
# not model-generated), so they need their own fixed per-language templates rather than
# relying on a sub-agent LLM call to translate them. A language not yet hand-written here
# falls back to an LLM translation of the English entry at call time (see
# graph.core_utils.localized()) rather than silently staying in English.
CONFIRM_DECLINED_TEMPLATES = {
    "en": "Okay, I won't make that change.",
    "hi": "ठीक है, मैं यह बदलाव नहीं करूँगा।",
    "ta": "சரி, நான் அந்த மாற்றத்தைச் செய்ய மாட்டேன்.",
}
CONFIRM_UNAVAILABLE_TEMPLATES = {
    "en": "Sorry, that plan is no longer available — nothing was changed.",
    "hi": "क्षमा करें, वह प्लान अब उपलब्ध नहीं है — कुछ भी नहीं बदला गया।",
    "ta": "மன்னிக்கவும், அந்தத் திட்டம் இப்போது கிடைக்கவில்லை — எதுவும் மாற்றப்படவில்லை.",
}
CONFIRM_CHANGED_TEMPLATES = {
    "en": "Confirmed — plan changed to {plan_name} (Rs {price}), effective immediately.",
    "hi": "पुष्टि हो गई — प्लान बदलकर {plan_name} (Rs {price}) कर दिया गया है, यह तुरंत प्रभावी है।",
    "ta": "உறுதி செய்யப்பட்டது — திட்டம் {plan_name} (Rs {price}) ஆக மாற்றப்பட்டது, உடனடியாக நடைமுறைக்கு வரும்.",
}
# subscriptions.phone_number is a FOREIGN KEY on customers(phone_number) -- used when
# session.phone_number has no matching row, which would otherwise raise a raw sqlite
# IntegrityError from the INSERT below (same failure mode as createComplaint() in
# tools/complaints.py). Shouldn't normally happen since changePlan() requires
# session.verified, but this keeps confirm_pending_action() failing soft either way.
CONFIRM_NO_ACCOUNT_TEMPLATES = {
    "en": "Sorry, I couldn't find an account for this number — nothing was changed. "
    "Connecting you to a human agent.",
    "hi": "क्षमा करें, इस नंबर के लिए कोई खाता नहीं मिला — कुछ भी नहीं बदला गया। "
    "आपको एक मानव एजेंट से जोड़ रहा हूँ।",
    "ta": "மன்னிக்கவும், இந்த எண்ணிற்கு கணக்கு எதுவும் கிடைக்கவில்லை — எதுவும் மாற்றப்படவில்லை. "
    "உங்களை ஒரு மனித முகவருடன் இணைக்கிறேன்.",
}


def confirm_pending_action(session: SessionContext, customer_said_yes: bool) -> str | None:
    """Code-level enforcement of the mandated consent script: called by the graph itself
    (never by the LLM) with a decision computed straight from the CUSTOMER's own transcript
    for the confirmation turn. Only actually writes to the database if customer_said_yes.
    Returns a result string if a pending action was resolved this call, else None (no
    pending action was waiting). The returned string is spoken back verbatim (it bypasses
    the sub-agent LLM, same as the consent script itself), so it's built from the fixed
    per-language templates above rather than left hardcoded in English.
    """
    # Deferred import -- see the matching comment in consent_script() above.
    from vay.graph.core_utils import localized

    pending = session.pending_action
    if not pending:
        return None
    session.pending_action = None  # consume it either way -- one confirmation attempt only
    # Use the language captured when the action was STAGED (see plans.changePlan), not
    # session.language now -- this turn is just the bare word "yes"/"no", which the ASR/
    # orchestrator correctly detects as English, and that must not be mistaken for the
    # call's actual language.
    lang = pending.get("language") or session.language

    if not customer_said_yes:
        return localized(CONFIRM_DECLINED_TEMPLATES, lang)

    conn = customer_db._connect()
    if pending["tool"] == "changePlan":
        new_plan_id = pending["args"]["new_plan_id"]
        plan = conn.execute("SELECT * FROM plans WHERE plan_id=?", (new_plan_id,)).fetchone()
        if not plan:
            return localized(CONFIRM_UNAVAILABLE_TEMPLATES, lang)
        cust = conn.execute(
            "SELECT 1 FROM customers WHERE phone_number=?", (session.phone_number,)
        ).fetchone()
        if not cust:
            return localized(CONFIRM_NO_ACCOUNT_TEMPLATES, lang)
        conn.execute(
            "UPDATE subscriptions SET status='cancelled' WHERE phone_number=? AND status='active'",
            (session.phone_number,),
        )
        conn.execute(
            "INSERT INTO subscriptions (phone_number, plan_id, activated_on, status, addons) "
            "VALUES (?,?,?,?,?)",
            (session.phone_number, new_plan_id, date.today().isoformat(), "active", ""),
        )
        conn.commit()
        template = localized(CONFIRM_CHANGED_TEMPLATES, lang)
        return template.format(plan_name=plan["plan_name"], price=plan["price"])

    return None


def build_escalate_tool(session: "SessionContext"):
    """Shared escalateToHuman tool factory, used by ALL FOUR sub-agents (billing,
    plans, complaints, coverage) -- not just complaints.

    Previously only complaints.py had this tool. The other three sub-agents had no
    tool-based way to signal "this needs a human" -- so when their LLM decided
    escalation was warranted, it could only say so in free text (e.g. "I can connect
    you to a human agent"), which the guardrail's HUMAN_REQUEST_PATTERNS regex then
    mis-matched against the ASSISTANT's own draft reply (see nodes/utils.py bugfix),
    silently discarding an otherwise good, specific answer. Giving every sub-agent
    this tool means an intentional escalation decision is always made explicitly and
    reliably via session.escalation_requested, exactly like complaints.py already did.
    """
    from langchain_core.tools import tool

    @tool
    def escalateToHuman(reason: str) -> str:
        """Escalate this call to a human agent, e.g. for anger/frustration, a repeated
        unresolved issue, an explicit request for a human, or an action (like approving
        an existing ticket, or a sensitive identity-verification step) that this
        assistant is not permitted to complete itself."""
        session.escalation_requested = True
        session.escalation_reason = reason
        return f"Escalating to a human agent: {reason}"

    return escalateToHuman


TROUBLESHOOT_FLOWS = {
    "call_drop": [
        "Confirm the location isn't a known low-coverage area (checkCoverage).",
        "Ask the customer to toggle Airplane Mode on/off to force re-registration.",
        "Confirm the SIM is seated properly and undamaged.",
        "Confirm VoLTE is enabled in phone settings.",
        "If unresolved, log a network complaint ticket for field-team review.",
    ],
    "slow_data": [
        "Confirm a data plan is active and not expired/exhausted.",
        "Check APN settings match Nexatel's standard APN.",
        "Restart the device and toggle mobile data off/on.",
        "Check for an area outage before assuming a device-side issue.",
        "Check whether the daily/monthly FUP data cap has been crossed.",
    ],
    "sms_issue": [
        "Confirm SMS balance/plan is active.",
        "Confirm SMSC number is correctly configured.",
        "Check if the daily SMS FUP cap (100/day) has been hit.",
    ],
    "cannot_call": [
        "Confirm the account isn't suspended for non-payment.",
        "Confirm Do Not Disturb or call-barring isn't enabled.",
        "Confirm the SIM hasn't been reported lost/blocked accidentally.",
        "Escalate to the Network team if all of the above are ruled out.",
    ],
    "recharge_not_reflecting": [
        "Ask for the payment reference/transaction ID and time of payment.",
        "Most payments reflect within 15 minutes; if beyond 30 minutes, log a billing "
        "complaint with the transaction reference.",
        "Never ask the customer to pay again while a trace is pending.",
    ],
}

SLA_DAYS = {
    "network": 2,
    "billing": 5,
    "service_request": 3,
    "technical": 2,
    "other": 3,
}


def _row_to_dict(row) -> dict:
    return dict(row) if row is not None else None


def _gen_ticket_id() -> str:
    return "NXT-" + "".join(random.choices(string.digits, k=6))
