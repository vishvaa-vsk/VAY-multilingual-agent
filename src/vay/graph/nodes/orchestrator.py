"""
orchestrator.py
---------------
Nexatel orchestrator node + sub-agent runner for the LangGraph voice assistant.

Orchestrator responsibilities:
- NLU: detect language, intent, route, entities, confidence
- Aggressive/abusive intent: warn on 1st offence, cut the call on 2nd
- Sensitive intent (fraud/dispute/cancellation): direct handoff
- Low-confidence / unclear: clarify re-prompt, then handoff after threshold
- Route to one of 4 sub-agents: billing / plans / complaints / coverage

Sub-agent runner (_run_subagent):
- Pre-fetches customer account context (balance, active plan, open tickets)
  from the mock CustomerDB and injects it into the system prompt so the LLM
  doesn't need a round-trip tool call for basic account info.
- Trims history to remove stale ToolMessages from a previous domain when the
  route switches between turns (e.g. billing → plans).
- Runs a bounded tool-calling loop and returns the draft reply.
"""

from __future__ import annotations

import json
import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

load_dotenv()

from vay.rag.retriever import RetrievalTracker
from vay.tools.session import SessionContext, confirm_pending_action

from vay.graph.state import GraphState
from vay.graph.utils import (
    AFFIRMATION_PATTERN,
    AGENT_NAMES,
    AGGRESSIVE_WARNING_TEMPLATES,
    CALL_CUT_TEMPLATES,
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


# ---------------------------------------------------------------------------
# Account context pre-fetcher
# ---------------------------------------------------------------------------

def _fetch_account_context(phone_number: str) -> str:
    """Fetch key account details from the mock CustomerDB and format them as a
    concise block to prepend to the sub-agent system prompt.

    This avoids a redundant tool-call round-trip just to get balance / active
    plan at the start of every sub-agent turn.  Returns an empty string on any
    DB error so the sub-agent degrades gracefully (it can still call tools).
    """
    try:
        import vay.tools.db_queries as db

        conn = db._connect()

        # Customer profile
        customer = conn.execute(
            "SELECT name FROM customers WHERE phone_number=?", (phone_number,)
        ).fetchone()
        if not customer:
            return ""

        # Active subscription
        sub = conn.execute(
            """
            SELECT p.plan_name, p.price, p.data_limit, p.voice_minutes, p.validity_days,
                   s.activated_on, s.addons
            FROM subscriptions s
            JOIN plans p ON s.plan_id = p.plan_id
            WHERE s.phone_number=? AND s.status='active'
            ORDER BY s.activated_on DESC LIMIT 1
            """,
            (phone_number,),
        ).fetchone()

        # Outstanding balance
        balance_row = conn.execute(
            "SELECT SUM(amount) as total FROM bills WHERE phone_number=? AND status='unpaid'",
            (phone_number,),
        ).fetchone()
        outstanding = balance_row["total"] if balance_row and balance_row["total"] else 0.0

        # Open tickets
        tickets = conn.execute(
            "SELECT ticket_id, category, status FROM tickets "
            "WHERE phone_number=? AND status NOT IN ('resolved','closed') "
            "ORDER BY created_at DESC LIMIT 3",
            (phone_number,),
        ).fetchall()

        lines = [
            "--- Customer Account Context (pre-fetched — use this directly) ---",
            f"Name: {customer['name']}",
        ]
        if sub:
            lines.append(
                f"Active Plan: {sub['plan_name']} | Rs {sub['price']}/month | "
                f"{sub['data_limit']} data | {sub['voice_minutes']} voice | "
                f"Valid: {sub['validity_days']} days | Since: {sub['activated_on']}"
            )
            if sub["addons"]:
                lines.append(f"Add-ons: {sub['addons']}")
        else:
            lines.append("Active Plan: No active subscription found.")

        lines.append(
            f"Outstanding Balance: Rs {outstanding:.2f}"
            if outstanding > 0
            else "Outstanding Balance: None (all bills paid)"
        )

        if tickets:
            ticket_strs = [f"{t['ticket_id']} ({t['category']}: {t['status']})" for t in tickets]
            lines.append(f"Open Tickets: {', '.join(ticket_strs)}")
        else:
            lines.append("Open Tickets: None")

        lines.append("--- End Account Context ---")
        return "\n".join(lines)

    except Exception as e:
        # Non-fatal: sub-agent can still call tools to get this data
        return f"(Account context unavailable: {e})"


# ---------------------------------------------------------------------------
# Orchestrator node
# ---------------------------------------------------------------------------

def orchestrator_node(state: GraphState) -> GraphState:
    """NLU + routing node.  Returns updates to GraphState.

    Handles:
    - Aggressive/abusive language: warning (1st offence) → call cut (2nd)
    - Sensitive intent: sets handoff=True
    - Unclear/low-confidence: clarify re-prompt, escalate after threshold
    - Valid routes: delegates to sub-agents
    """
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
        parsed = {
            "language": state["language"],
            "intent": "unclear",
            "route": "unclear",
            "normalized_query": state["transcript"],
            "entities": {},
            "confidence": 0.0,
            "sensitive": False,
            "aggressive": False,
            "call_end_requested": False,
        }

    route = parsed.get("route") if parsed.get("route") in VALID_ROUTES else "unclear"
    confidence = (
        float(parsed.get("confidence", 0.0))
        if isinstance(parsed.get("confidence"), (int, float))
        else 0.0
    )
    sensitive = bool(parsed.get("sensitive", False))
    aggressive = bool(parsed.get("aggressive", False))
    detected_lang = parsed.get("language") or state["language"]

    # --- Pending-action overrides routing back to the owning sub-agent ---
    session = state.get("session")
    forced_by_pending_action = bool(session and session.pending_action)
    if forced_by_pending_action:
        route = PENDING_ACTION_ROUTE.get(session.pending_action.get("tool"), route)
        sensitive = False
        aggressive = False
        confidence = 1.0

    # --- Aggressive / abusive caller handling ---
    # This is SEPARATE from sensitive: aggressive callers get a warning first,
    # then a call-cut — NOT a handoff to a human agent.
    warning_reply = ""
    cut_call = False
    if aggressive and not forced_by_pending_action:
        current_agg_count = state.get("aggressive_count", 0) + 1
        if session:
            session.aggressive_count = current_agg_count
        if current_agg_count == 1:
            # First offence: issue a firm warning, skip routing
            warning_reply = localized(AGGRESSIVE_WARNING_TEMPLATES, detected_lang)
        else:
            # Second or more: terminate the call
            warning_reply = localized(CALL_CUT_TEMPLATES, detected_lang)
            cut_call = True
    else:
        current_agg_count = state.get("aggressive_count", 0)

    # --- Unclear / low-confidence escalation tracking ---
    unclear_escalate = False
    is_unclear = route == "unclear" or confidence < DEFAULT_NLU_CONFIDENCE
    normalized_q = parsed.get("normalized_query", "")
    explicit_human_request = bool(
        HUMAN_REQUEST_PATTERNS.search(state["transcript"]) or
        HUMAN_REQUEST_PATTERNS.search(normalized_q) or
        parsed.get("intent") in ("escalate", "request_human_agent", "human_handoff", "speak_to_agent")
    )

    if session and not forced_by_pending_action and not aggressive:
        if sensitive or explicit_human_request:
            unclear_escalate = True
            session.consecutive_unclear = 0
        elif is_unclear:
            session.consecutive_unclear += 1
            unclear_escalate = session.consecutive_unclear >= UNCLEAR_ESCALATION_THRESHOLD
        else:
            session.consecutive_unclear = 0

    handoff_reason = ""
    if sensitive:
        handoff_reason = "Sensitive intent detected (billing dispute, cancellation, fraud/security)."
    elif explicit_human_request:
        handoff_reason = "Customer explicitly asked for a human agent."
    elif unclear_escalate:
        handoff_reason = (
            f"Unclear/low-confidence for "
            f"{session.consecutive_unclear if session else '?'} consecutive turns despite a clarify re-prompt."
        )

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
        "call_end_requested": cut_call or bool(parsed.get("call_end_requested", False)),
        "language": detected_lang,
        # Aggressive tracking
        "aggressive_count": current_agg_count,
        "warning_reply": warning_reply,
        # Remember previous route for domain-switch history trimming in _run_subagent
        "previous_route": state.get("route", ""),
    }


# ---------------------------------------------------------------------------
# Sub-agent runner (shared by billing_node / plans_node / complaints_node / coverage_node)
# ---------------------------------------------------------------------------

def _run_subagent(state: GraphState, route: str, tools_builder, rag_tool_builder) -> GraphState:
    """Run a domain sub-agent tool-calling loop and return the draft reply.

    Steps:
    1. Resolve / create the SessionContext for this call.
    2. If a pending consent action is waiting, resolve it in code (not LLM).
    3. Pre-fetch account context from the CustomerDB.
    4. Trim history: on a domain switch (previous_route != route), strip stale
       ToolMessages so the new agent isn't confused by the prior domain's data.
    5. Run the bounded tool-calling loop.
    6. Return draft_reply + retrieval_score.
    """
    # session is created ONCE per call in main() and threaded through state every turn
    session = state.get("session") or SessionContext(
        phone_number=state["phone_number"], verified=True
    )
    session.language = state["language"]

    # Code-enforced consent gate: if a sensitive action is awaiting confirmation
    # from a PRIOR turn, resolve it now from the customer's own transcript.
    if session.pending_action:
        said_yes = bool(AFFIRMATION_PATTERN.search(state["transcript"])) and not bool(
            NEGATION_PATTERN.search(state["transcript"])
        )
        outcome = confirm_pending_action(session, customer_said_yes=said_yes)
        if outcome is not None:
            return {"draft_reply": outcome, "retrieval_score": 1.0}

    # Pre-fetch account context for injection into the system prompt
    account_context = _fetch_account_context(state["phone_number"])

    # Domain-switch history trimming: drop ToolMessages from a prior agent's
    # domain so they don't pollute the new sub-agent's context window.
    raw_history: list = state.get("conversation_history", [])
    previous_route = state.get("previous_route", "")
    domain_switched = bool(previous_route) and previous_route != route and previous_route in VALID_ROUTES
    if domain_switched:
        # Keep only HumanMessage and AIMessage (without tool_calls) — drop ToolMessages
        # and AIMessages that are just tool-call stubs from the previous domain.
        clean_history = [
            msg for msg in raw_history
            if isinstance(msg, HumanMessage)
            or (isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None))
        ]
        if state.get("show_debug"):
            print(
                f"  [domain switch {previous_route} → {route}: "
                f"trimmed {len(raw_history) - len(clean_history)} tool messages from history]"
            )
    else:
        clean_history = raw_history

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
            account_context=account_context,
        ),
        state["normalized_query"],
        clean_history,
        language=state["language"],
        show_debug=state.get("show_debug", False),
    )

    result: GraphState = {
        "draft_reply": reply,
        "retrieval_score": tracker.last_score if tracker.called else 1.0,
        # no RAG call needed → don't penalize the confidence gate
    }
    if session.escalation_requested:
        result["handoff"] = True
        result["handoff_reason"] = session.escalation_reason or "Escalation requested by sub-agent."
    return result
