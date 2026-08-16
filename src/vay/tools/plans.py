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

from datetime import date, datetime

import vay.tools.db_queries as customer_db

from langchain_core.tools import tool

from vay.tools.session import (
    SENSITIVE_DENIAL,
    SessionContext,
    build_escalate_tool,
    consent_script,
)


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
        return (
            "\n".join(
                f"{r['plan_id']}: {r['plan_name']} — Rs {r['price']}/{r['validity_days']}d, "
                f"data={r['data_limit']}, voice={r['voice_minutes']}, sms={r['sms']}"
                for r in rows
            )
            or "No plans found."
        )

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
                return (
                    f"Not eligible for {plan['plan_name']}: requires age 18-25 (caller is {age})."
                )
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
        summary = (
            f"change your plan to {plan['plan_name']} at Rs {plan['price']}, effective immediately"
        )
        session.pending_action = {
            "tool": "changePlan",
            "args": {"new_plan_id": new_plan_id},
            "summary": summary,
        }
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

    return [
        listPlans,
        comparePlans,
        changePlan,
        activateAddOn,
        checkEligibility,
        build_escalate_tool(session),
    ]


# ---------------------------------------------------------------------------
# Complaints & Service-Request Agent tools
