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

import json
import random
import string
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import customer_db
from langchain_core.tools import tool


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
    escalation_requested: bool = False   # set True by escalateToHuman()
    escalation_reason: str = ""
    pending_action: dict | None = None  # {"tool": "changePlan"|"sendPaymentLink", "args": {...}, "summary": str}
    # Consecutive turns the orchestrator failed to route (route="unclear" or low NLU
    # confidence). Reset to 0 the moment a turn is understood. agent_graph.py uses this to
    # give the customer one gentle "could you clarify?" re-prompt before ever handing off to
    # a human for a merely-unclear utterance -- see agent_graph.route_after_orchestrator().
    consecutive_unclear: int = 0


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
    "en": "To confirm, I'll {summary}. Please say \"yes\" to confirm, or \"no\" to cancel.",
    "hi": "पुष्टि के लिए बता रहा हूँ: मैं {summary}। पुष्टि करने के लिए अंग्रेज़ी में \"yes\" बोलें, "
          "या रद्द करने के लिए \"no\" बोलें।",
    "ta": "உறுதிசெய்ய: நான் {summary}. உறுதிசெய்ய ஆங்கிலத்தில் \"yes\" என்று சொல்லுங்கள், "
          "அல்லது ரத்து செய்ய \"no\" என்று சொல்லுங்கள்.",
}


def consent_script(language: str, summary: str) -> str:
    """Build the fixed-template consent question for a sensitive action, in the caller's
    language, always ending with the literal English yes/no instruction."""
    template = CONSENT_TEMPLATES.get(language, CONSENT_TEMPLATES["en"])
    return template.format(summary=summary)


# These three outcomes of confirm_pending_action() bypass the LLM entirely (same rationale
# as CONSENT_TEMPLATES above -- a real DB mutation's confirmation text must be deterministic,
# not model-generated), so they need their own fixed per-language templates rather than
# relying on a sub-agent LLM call to translate them. Falls back to English for any language
# not yet covered here -- add more entries rather than leaving callers silently in English.
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


def confirm_pending_action(session: SessionContext, customer_said_yes: bool) -> str | None:
    """Code-level enforcement of the mandated consent script: called by the graph itself
    (never by the LLM) with a decision computed straight from the CUSTOMER's own transcript
    for the confirmation turn. Only actually writes to the database if customer_said_yes.
    Returns a result string if a pending action was resolved this call, else None (no
    pending action was waiting). The returned string is spoken back verbatim (it bypasses
    the sub-agent LLM, same as the consent script itself), so it's built from the fixed
    per-language templates above rather than left hardcoded in English.
    """
    pending = session.pending_action
    if not pending:
        return None
    session.pending_action = None  # consume it either way -- one confirmation attempt only
    lang = session.language

    if not customer_said_yes:
        return CONFIRM_DECLINED_TEMPLATES.get(lang, CONFIRM_DECLINED_TEMPLATES["en"])

    conn = customer_db._connect()
    if pending["tool"] == "changePlan":
        new_plan_id = pending["args"]["new_plan_id"]
        plan = conn.execute("SELECT * FROM plans WHERE plan_id=?", (new_plan_id,)).fetchone()
        if not plan:
            return CONFIRM_UNAVAILABLE_TEMPLATES.get(lang, CONFIRM_UNAVAILABLE_TEMPLATES["en"])
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
        template = CONFIRM_CHANGED_TEMPLATES.get(lang, CONFIRM_CHANGED_TEMPLATES["en"])
        return template.format(plan_name=plan["plan_name"], price=plan["price"])

    return None

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


# ---------------------------------------------------------------------------
# Billing & Payments Agent tools
# ---------------------------------------------------------------------------
def build_billing_tools(session: SessionContext) -> list:
    conn = customer_db._connect()

    @tool
    def getBalance() -> str:
        """Get the caller's current account balance / plan status: for prepaid,
        remaining validity and data; for postpaid/broadband, outstanding due amount."""
        cust = _row_to_dict(conn.execute(
            "SELECT * FROM customers WHERE phone_number=?", (session.phone_number,)
        ).fetchone())
        if not cust:
            return "No account found for this number."

        sub = _row_to_dict(conn.execute(
            "SELECT s.*, p.plan_name, p.price, p.validity_days, p.data_limit "
            "FROM subscriptions s JOIN plans p ON p.plan_id = s.plan_id "
            "WHERE s.phone_number=? AND s.status='active' "
            "ORDER BY s.subscription_id DESC LIMIT 1",
            (session.phone_number,),
        ).fetchone())
        if not sub:
            return "No active plan found on this account."

        if cust["account_type"] == "prepaid":
            activated = datetime.fromisoformat(sub["activated_on"]).date()
            expiry = activated + timedelta(days=sub["validity_days"] or 0)
            days_left = (expiry - date.today()).days
            return (f"Prepaid plan '{sub['plan_name']}' ({sub['data_limit']}), "
                    f"validity {'expired' if days_left < 0 else f'{days_left} days left'} "
                    f"(expires {expiry.isoformat()}).")

        bill = _row_to_dict(conn.execute(
            "SELECT * FROM bills WHERE phone_number=? AND status!='paid' "
            "ORDER BY due_date ASC LIMIT 1",
            (session.phone_number,),
        ).fetchone())
        if bill:
            return (f"{cust['account_type'].title()} plan '{sub['plan_name']}'. "
                    f"Outstanding: Rs {bill['amount']:.2f}, status={bill['status']}, "
                    f"due {bill['due_date']}.")
        return f"{cust['account_type'].title()} plan '{sub['plan_name']}'. No outstanding dues."

    @tool
    def getBillBreakup(billing_period: str = "") -> str:
        """Get the itemized charge breakup for the caller's bill. Pass billing_period
        as 'YYYY-MM' for a specific month, or leave empty for the latest bill."""
        if billing_period:
            bill = conn.execute(
                "SELECT * FROM bills WHERE phone_number=? AND billing_period=?",
                (session.phone_number, billing_period),
            ).fetchone()
        else:
            bill = conn.execute(
                "SELECT * FROM bills WHERE phone_number=? ORDER BY billing_period DESC LIMIT 1",
                (session.phone_number,),
            ).fetchone()
        if not bill:
            return "No bill found for that period."
        breakup = json.loads(bill["breakup_json"])
        lines = [f"Bill for {bill['billing_period']} (status: {bill['status']}, due {bill['due_date']}):"]
        for k, v in breakup.items():
            if v:
                lines.append(f"  {k.replace('_', ' ')}: Rs {v}")
        return "\n".join(lines)

    @tool
    def getDueDate() -> str:
        """Get the caller's next payment due date and amount, if any."""
        bill = conn.execute(
            "SELECT * FROM bills WHERE phone_number=? AND status!='paid' "
            "ORDER BY due_date ASC LIMIT 1",
            (session.phone_number,),
        ).fetchone()
        if not bill:
            return "No pending dues — account is fully paid up."
        return f"Rs {bill['amount']:.2f} due on {bill['due_date']} (status: {bill['status']})."

    @tool
    def sendPaymentLink() -> str:
        """Send a payment link (via SMS) to the caller's registered number for their
        current outstanding balance. Sensitive action — requires identity verification."""
        if not session.verified:
            return SENSITIVE_DENIAL
        bill = conn.execute(
            "SELECT * FROM bills WHERE phone_number=? AND status!='paid' "
            "ORDER BY due_date ASC LIMIT 1",
            (session.phone_number,),
        ).fetchone()
        if not bill:
            return "No outstanding balance — no payment link needed."
        token = "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        link = f"https://pay.nexatel.in/{token}"
        print(f"  [MOCK SMS to {session.phone_number}] Pay Rs {bill['amount']:.2f}: {link}")
        return f"Payment link for Rs {bill['amount']:.2f} sent via SMS to the registered number: {link} (valid 24 hours)."

    @tool
    def explainCharge(charge_name: str) -> str:
        """Explain a specific charge line item (e.g. 'roaming_surcharge', 'gst_18pct',
        'late_payment_fee') found on the caller's latest bill."""
        bill = conn.execute(
            "SELECT * FROM bills WHERE phone_number=? ORDER BY billing_period DESC LIMIT 1",
            (session.phone_number,),
        ).fetchone()
        if not bill:
            return "No bill on file to explain charges from."
        breakup = json.loads(bill["breakup_json"])
        key = charge_name.strip().lower().replace(" ", "_")
        matches = {k: v for k, v in breakup.items() if key in k}
        if not matches:
            return f"'{charge_name}' was not found as a line item on the latest bill."
        return "; ".join(f"{k.replace('_', ' ')}: Rs {v}" for k, v in matches.items() if v)

    return [getBalance, getBillBreakup, getDueDate, sendPaymentLink, explainCharge]


# ---------------------------------------------------------------------------
# Plans & Offers Agent tools
# ---------------------------------------------------------------------------
def build_plans_tools(session: SessionContext) -> list:
    conn = customer_db._connect()

    @tool
    def listPlans(plan_type: str = "") -> str:
        """List Nexatel plans, optionally filtered by plan_type
        ('prepaid', 'postpaid', or 'broadband')."""
        if plan_type:
            rows = conn.execute("SELECT * FROM plans WHERE plan_type=?", (plan_type,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM plans").fetchall()
        return "\n".join(
            f"{r['plan_id']}: {r['plan_name']} — Rs {r['price']}/{r['validity_days']}d, "
            f"data={r['data_limit']}, voice={r['voice_minutes']}, sms={r['sms']}"
            for r in rows
        ) or "No plans found."

    @tool
    def comparePlans(plan_ids: list[str]) -> str:
        """Compare two or more Nexatel plans by their plan_id (see listPlans)."""
        rows = []
        for pid in plan_ids:
            r = conn.execute("SELECT * FROM plans WHERE plan_id=?", (pid,)).fetchone()
            if r:
                rows.append(r)
        if not rows:
            return "None of the given plan_ids were found."
        return "\n".join(
            f"{r['plan_id']}: {r['plan_name']} — Rs {r['price']}, data={r['data_limit']}, "
            f"voice={r['voice_minutes']}, sms={r['sms']}, desc={r['description']}"
            for r in rows
        )

    @tool
    def checkEligibility(plan_id: str) -> str:
        """Check whether the caller is eligible for a given plan_id (e.g. Youth Unlimited
        requires age 18-25)."""
        plan = conn.execute("SELECT * FROM plans WHERE plan_id=?", (plan_id,)).fetchone()
        if not plan:
            return f"Unknown plan_id '{plan_id}'."
        cust = conn.execute(
            "SELECT * FROM customers WHERE phone_number=?", (session.phone_number,)
        ).fetchone()
        if plan_id == "YOUTH_UNL" and cust and cust["dob"]:
            age = (date.today() - datetime.fromisoformat(cust["dob"]).date()).days // 365
            if not (18 <= age <= 25):
                return f"Not eligible for {plan['plan_name']}: requires age 18-25 (caller is {age})."
        if cust and plan["plan_type"] != "prepaid" and not cust["kyc_verified"]:
            return f"Not eligible for {plan['plan_name']}: KYC verification is pending."
        return f"Eligible for {plan['plan_name']}."

    @tool
    def changePlan(new_plan_id: str) -> str:
        """Propose changing the caller's active plan to new_plan_id (see listPlans for valid
        ids). Sensitive action — this does NOT change anything yet. It stages the change and
        returns a consent script; you must read that back to the customer verbatim and wait
        for their next reply. The change only actually applies once the customer confirms in
        a following turn — do not tell the customer it's done from this call alone."""
        if not session.verified:
            return SENSITIVE_DENIAL
        plan = conn.execute("SELECT * FROM plans WHERE plan_id=?", (new_plan_id,)).fetchone()
        if not plan:
            return f"Unknown plan_id '{new_plan_id}' — cannot change plan."
        summary = f"change your plan to {plan['plan_name']} at Rs {plan['price']}, effective immediately"
        session.pending_action = {"tool": "changePlan", "args": {"new_plan_id": new_plan_id}, "summary": summary}
        # STOP_AND_SAY: is a code-level sentinel -- run_tool_agent() in agent_graph.py
        # recognizes this prefix and returns the rest verbatim as the turn's final reply,
        # WITHOUT letting the LLM see/paraphrase it. Live testing showed the LLM cannot be
        # trusted to relay a "wait for confirmation" instruction faithfully -- it will
        # confidently claim the change is "done" in the same turn if given the chance.
        return "STOP_AND_SAY: " + consent_script(session.language, summary)

    @tool
    def activateAddOn(addon_name: str) -> str:
        """Activate an add-on (e.g. 'OTT Super Bundle', 'Extra Data 5GB') on the caller's
        active subscription."""
        sub = conn.execute(
            "SELECT * FROM subscriptions WHERE phone_number=? AND status='active' "
            "ORDER BY subscription_id DESC LIMIT 1",
            (session.phone_number,),
        ).fetchone()
        if not sub:
            return "No active subscription found to attach the add-on to."
        existing = [a.strip() for a in (sub["addons"] or "").split(",") if a.strip()]
        if addon_name not in existing:
            existing.append(addon_name)
        conn.execute(
            "UPDATE subscriptions SET addons=? WHERE subscription_id=?",
            (",".join(existing), sub["subscription_id"]),
        )
        conn.commit()
        return f"Add-on '{addon_name}' activated (may take up to 2 hours to reflect)."

    return [listPlans, comparePlans, changePlan, activateAddOn, checkEligibility]


# ---------------------------------------------------------------------------
# Complaints & Service-Request Agent tools
# ---------------------------------------------------------------------------
def build_complaints_tools(session: SessionContext) -> list:
    conn = customer_db._connect()

    @tool
    def createComplaint(category: str, description: str) -> str:
        """Log a new complaint/service-request ticket. category must be one of
        'network', 'billing', 'service_request', 'technical', 'other'."""
        category = category.strip().lower()
        if category not in SLA_DAYS:
            category = "other"
        ticket_id = _gen_ticket_id()
        created_at = date.today().isoformat()
        sla_due = (date.today() + timedelta(days=SLA_DAYS[category])).isoformat()
        conn.execute(
            "INSERT INTO tickets VALUES (?,?,?,?,?,?,?,?)",
            (ticket_id, session.phone_number, category, description, "open", created_at, sla_due, ""),
        )
        conn.commit()
        return f"Ticket {ticket_id} logged (category={category}), target resolution by {sla_due}."

    @tool
    def getTicketStatus(ticket_id: str = "") -> str:
        """Get the status of a ticket by ticket_id, or the caller's most recent
        tickets if ticket_id is left empty."""
        if ticket_id:
            row = conn.execute("SELECT * FROM tickets WHERE ticket_id=?", (ticket_id,)).fetchone()
            rows = [row] if row else []
        else:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE phone_number=? ORDER BY created_at DESC LIMIT 3",
                (session.phone_number,),
            ).fetchall()
        if not rows:
            return "No matching ticket found."
        return "\n".join(
            f"{r['ticket_id']} [{r['category']}] status={r['status']}, sla_due={r['sla_due']}: "
            f"{r['description']}" + (f" — notes: {r['resolution_notes']}" if r["resolution_notes"] else "")
            for r in rows
        )

    @tool
    def runTroubleshootFlow(issue_type: str) -> str:
        """Get the standard troubleshooting steps for issue_type: one of
        'call_drop', 'slow_data', 'sms_issue', 'cannot_call', 'recharge_not_reflecting'."""
        steps = TROUBLESHOOT_FLOWS.get(issue_type.strip().lower())
        if not steps:
            return f"No standard troubleshooting flow for '{issue_type}'."
        return "\n".join(f"{i+1}. {s}" for i, s in enumerate(steps))

    @tool
    def escalateToHuman(reason: str) -> str:
        """Escalate this call to a human agent, e.g. for anger/frustration, a repeated
        unresolved issue, or an explicit request for a human."""
        session.escalation_requested = True
        session.escalation_reason = reason
        return f"Escalating to a human agent: {reason}"

    return [createComplaint, getTicketStatus, runTroubleshootFlow, escalateToHuman]


# ---------------------------------------------------------------------------
# Coverage & Technical Agent tools
# ---------------------------------------------------------------------------
DEVICE_SETTINGS = {
    "android": (
        "APN: Settings > Network & Internet > Mobile Network > Access Point Names > "
        "Add new: APN name 'Nexatel Internet', APN 'nexatel.data'. "
        "VoLTE: Settings > Network & Internet > Mobile Network > toggle 'VoLTE calls'."
    ),
    "iphone": (
        "APN: Settings > Mobile Data > Mobile Data Network > enter 'nexatel.data' under "
        "Mobile Data. Check Settings > General > About for a Carrier Settings Update. "
        "VoLTE: Settings > Mobile Data > Mobile Data Options > Voice & Data > select '4G' or '5G Auto'."
    ),
}


def build_coverage_tools(session: SessionContext) -> list:
    conn = customer_db._connect()

    @tool
    def checkCoverage(pincode: str) -> str:
        """Check Nexatel network coverage (signal strength, technology) for a pincode."""
        row = conn.execute("SELECT * FROM coverage WHERE pincode=?", (pincode,)).fetchone()
        if not row:
            return f"No coverage data on file for pincode {pincode} — treat as unverified/rural area."
        return f"{row['area']} ({pincode}): signal={row['signal_strength']}, technology={row['technology']}."

    @tool
    def getOutageStatus(pincode: str) -> str:
        """Check whether there is a known network outage in a pincode."""
        row = conn.execute("SELECT * FROM coverage WHERE pincode=?", (pincode,)).fetchone()
        if not row:
            return f"No outage data on file for pincode {pincode}."
        status = row["outage_status"]
        if status == "none":
            return f"No known outage in {row['area']} ({pincode})."
        return f"Outage status in {row['area']} ({pincode}): {status}."

    @tool
    def getDeviceSettings(device_type: str) -> str:
        """Get APN/VoLTE configuration steps for device_type ('android' or 'iphone')."""
        return DEVICE_SETTINGS.get(device_type.strip().lower(), "Unknown device_type — use 'android' or 'iphone'.")

    @tool
    def guideSimSwap() -> str:
        """Get the standard SIM/eSIM replacement process steps."""
        return (
            "1. Verify identity (name, address, last bill/recharge amount, or OTP to an alternate contact). "
            "2. Log a service-request ticket. "
            "3. Visit a retail outlet with ID proof, or request doorstep delivery (Rs 49 fee). "
            "4. New SIM activates within 2-4 hours; old SIM stops working immediately upon activation. "
            "5. Number, balance, and plan carry over automatically."
        )

    return [checkCoverage, getOutageStatus, getDeviceSettings, guideSimSwap]
