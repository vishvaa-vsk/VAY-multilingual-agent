"""Smoke test: invoke every read-only tool from every sub-agent once against the
seeded mock DB and assert it doesn't raise.

Added Aug 16 session: two tools (`billing.getBalance`, `complaints.createComplaint`)
were crashing on every single call with a NameError (an undefined name that was
never imported) -- and neither `test_rag.py`, `test_routing.py`, nor `test_types.py`
would ever have caught it, because none of them actually invoke a tool against the
DB. See rag-tts-evaluvation.md §2.1/§2.2 for the full story. This test exists so
that class of bug fails CI instead of a live call.

Deliberately only exercises READ-ONLY tools (no changePlan/createComplaint/
sendPaymentLink/activateAddOn) so this test never mutates the seeded DB as a side
effect of running the suite.
"""

from __future__ import annotations

import pytest

from vay.tools.billing import build_billing_tools
from vay.tools.complaints import build_complaints_tools
from vay.tools.coverage import build_coverage_tools
from vay.tools.plans import build_plans_tools
from vay.tools.session import SessionContext

# A seeded phone number for each account type covered by db_seed_data.py
PREPAID_PHONE = "9876500001"
POSTPAID_PHONE = "9876500002"


def _session(phone: str = PREPAID_PHONE) -> SessionContext:
    return SessionContext(phone_number=phone, verified=True, language="en")


@pytest.mark.parametrize("phone", [PREPAID_PHONE, POSTPAID_PHONE])
def test_billing_read_tools_do_not_crash(phone: str) -> None:
    session = _session(phone)
    tools = {t.name: t for t in build_billing_tools(session)}
    assert tools["getBalance"].invoke({})
    assert tools["getBillBreakup"].invoke({})
    assert tools["getDueDate"].invoke({})


def test_plans_read_tools_do_not_crash() -> None:
    session = _session()
    tools = {t.name: t for t in build_plans_tools(session)}
    assert tools["listPlans"].invoke({"plan_type": "prepaid"})
    assert tools["checkEligibility"].invoke({"plan_id": "PPD_VALUE"})


def test_complaints_read_tools_do_not_crash() -> None:
    session = _session()
    tools = {t.name: t for t in build_complaints_tools(session)}
    assert tools["getTicketStatus"].invoke({})
    assert tools["runTroubleshootFlow"].invoke({"issue_type": "slow_data"})


def test_coverage_read_tools_do_not_crash() -> None:
    session = _session()
    tools = {t.name: t for t in build_coverage_tools(session)}
    assert tools["checkCoverage"].invoke({"pincode": "600001"})
    assert tools["getOutageStatus"].invoke({"pincode": "600001"})
    assert tools["getDeviceSettings"].invoke({"device_type": "android"})
    assert tools["guideSimSwap"].invoke({})
    # coverage's getTicketStatus was added Aug 16 -- confirm it's wired and callable
    assert tools["getTicketStatus"].invoke({})


def test_all_four_agents_have_escalate_to_human() -> None:
    """Aug 16 fix: previously only complaints had this tool."""
    session = _session()
    for builder in (
        build_billing_tools,
        build_plans_tools,
        build_complaints_tools,
        build_coverage_tools,
    ):
        names = [t.name for t in builder(session)]
        assert "escalateToHuman" in names
