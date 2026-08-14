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

from datetime import date, timedelta

import vay.tools.db_queries as customer_db

from langchain_core.tools import tool

from vay.tools.session import (
    TROUBLESHOOT_FLOWS,
    SessionContext,
    _gen_ticket_id,
)


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
            (
                ticket_id,
                session.phone_number,
                category,
                description,
                "open",
                created_at,
                sla_due,
                "",
            ),
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
            f"{r['description']}"
            + (f" — notes: {r['resolution_notes']}" if r["resolution_notes"] else "")
            for r in rows
        )

    @tool
    def runTroubleshootFlow(issue_type: str) -> str:
        """Get the standard troubleshooting steps for issue_type: one of
        'call_drop', 'slow_data', 'sms_issue', 'cannot_call', 'recharge_not_reflecting'."""
        steps = TROUBLESHOOT_FLOWS.get(issue_type.strip().lower())
        if not steps:
            return f"No standard troubleshooting flow for '{issue_type}'."
        return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))

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
