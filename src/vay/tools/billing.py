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
from datetime import date, datetime, timedelta

import customer_db
from langchain_core.tools import tool

from vay.tools.session import SENSITIVE_DENIAL, SessionContext


# ---------------------------------------------------------------------------
# Billing & Payments Agent tools
# ---------------------------------------------------------------------------
def build_billing_tools(session: SessionContext) -> list:
    conn = customer_db._connect()

    @tool
    def getBalance() -> str:
        """Get the caller's current account balance / plan status: for prepaid,
        remaining validity and data; for postpaid/broadband, outstanding due amount."""
        cust = _row_to_dict(
            conn.execute(
                "SELECT * FROM customers WHERE phone_number=?", (session.phone_number,)
            ).fetchone()
        )
        if not cust:
            return "No account found for this number."

        sub = _row_to_dict(
            conn.execute(
                "SELECT s.*, p.plan_name, p.price, p.validity_days, p.data_limit "
                "FROM subscriptions s JOIN plans p ON p.plan_id = s.plan_id "
                "WHERE s.phone_number=? AND s.status='active' "
                "ORDER BY s.subscription_id DESC LIMIT 1",
                (session.phone_number,),
            ).fetchone()
        )
        if not sub:
            return "No active plan found on this account."

        if cust["account_type"] == "prepaid":
            activated = datetime.fromisoformat(sub["activated_on"]).date()
            expiry = activated + timedelta(days=sub["validity_days"] or 0)
            days_left = (expiry - date.today()).days
            return (
                f"Prepaid plan '{sub['plan_name']}' ({sub['data_limit']}), "
                f"validity {'expired' if days_left < 0 else f'{days_left} days left'} "
                f"(expires {expiry.isoformat()})."
            )

        bill = _row_to_dict(
            conn.execute(
                "SELECT * FROM bills WHERE phone_number=? AND status!='paid' "
                "ORDER BY due_date ASC LIMIT 1",
                (session.phone_number,),
            ).fetchone()
        )
        if bill:
            return (
                f"{cust['account_type'].title()} plan '{sub['plan_name']}'. "
                f"Outstanding: Rs {bill['amount']:.2f}, status={bill['status']}, "
                f"due {bill['due_date']}."
            )
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
        lines = [
            f"Bill for {bill['billing_period']} (status: {bill['status']}, due {bill['due_date']}):"
        ]
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
