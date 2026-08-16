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

import vay.tools.db_queries as customer_db

from langchain_core.tools import tool

from vay.tools.session import SessionContext, build_escalate_tool


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
            return (
                f"No coverage data on file for pincode {pincode} — treat as unverified/rural area."
            )
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
        return DEVICE_SETTINGS.get(
            device_type.strip().lower(), "Unknown device_type — use 'android' or 'iphone'."
        )

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

    @tool
    def getTicketStatus(ticket_id: str = "") -> str:
        """Get the status of a previously-logged network/technical ticket by ticket_id, or
        the caller's most recent tickets if ticket_id is left empty. Use this BEFORE asking
        for a pincode or re-running troubleshooting when the customer is asking about an
        issue they already reported (e.g. "is my 5G issue fixed", "any update on my
        ticket") -- the account context above may already show it, but this also finds
        RESOLVED tickets that the account context omits."""
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
            f"{r['description']}"
            + (f" — notes: {r['resolution_notes']}" if r["resolution_notes"] else "")
            for r in rows
        )

    return [
        checkCoverage,
        getOutageStatus,
        getDeviceSettings,
        guideSimSwap,
        getTicketStatus,
        build_escalate_tool(session),
    ]
