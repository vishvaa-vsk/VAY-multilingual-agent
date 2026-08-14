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


def run_tool_agent(
    llm: ChatGroq,
    tools: list,
    system_prompt: str,
    user_text: str,
    history: list,
    language: str = "en",
    show_debug: bool = False,
) -> str:
    """Minimal bounded tool-calling loop: LLM <-> tools until it stops calling
    tools or MAX_TOOL_ITERATIONS is hit."""
    bound_llm = llm.bind_tools(tools)
    tools_by_name = {t.name: t for t in tools}

    messages = (
        [SystemMessage(content=system_prompt)] + list(history) + [HumanMessage(content=user_text)]
    )

    for _ in range(MAX_TOOL_ITERATIONS):
        try:
            ai_msg: AIMessage = bound_llm.invoke(messages)
        except Exception as e:
            # The model hallucinated something the Groq API rejected outright before we
            # ever got a normal response back (e.g. calling an unregistered tool name) --
            # degrade to a safe, guardrail-recognized reply instead of crashing the call.
            if show_debug:
                print(f"  [tool-calling LLM call failed, degrading to handoff: {e}]")
            return localized(TOOL_LOOP_FAILURE_TEMPLATES, language)
        messages.append(ai_msg)

        if not getattr(ai_msg, "tool_calls", None):
            reply = (ai_msg.content or "").strip() or localized(HANDOFF_MESSAGE_TEMPLATES, language)
            if show_debug:
                print(f"  [LLM final reply] {reply}")
            return reply

        if show_debug and (ai_msg.content or "").strip():
            # Some models emit reasoning/commentary text alongside tool_calls -- show it so a
            # garbled/off-topic final reply is traceable to what the LLM actually said.
            print(f"  [LLM message] {ai_msg.content.strip()[:500]}")

        for call in ai_msg.tool_calls:
            tool_fn = tools_by_name.get(call["name"])
            if tool_fn is None:
                result = f"Unknown tool: {call['name']}"
            else:
                try:
                    result = tool_fn.invoke(call["args"])
                except Exception as e:
                    result = f"Tool error: {e}"
            if show_debug:
                print(f"  [tool call] {call['name']}({call['args']}) -> {str(result)[:200]}")

            # STOP_AND_SAY: sentinel (see tools.changePlan) -- a sensitive action just staged
            # a pending confirmation. Return its consent script VERBATIM as the final reply
            # instead of letting the LLM see and potentially paraphrase/misreport it -- live
            # testing showed a small model will happily claim "done" here if given the chance.
            result_str = str(result)
            if result_str.startswith("STOP_AND_SAY:"):
                return result_str[len("STOP_AND_SAY:") :].strip()

            messages.append(ToolMessage(content=result_str, tool_call_id=call["id"]))

    # Ran out of iterations without a final answer -- force a concrete wrap-up grounded in
    # whatever tool/search results are already in the transcript, rather than a vague prompt
    # that a small model tends to answer with unrelated filler.
    if show_debug:
        print(f"  [ran out of {MAX_TOOL_ITERATIONS} tool iterations -- forcing a grounded wrap-up]")
    messages.append(
        HumanMessage(
            content="You're out of tool calls for this turn. Using ONLY the concrete facts already "
            "returned by your tool calls and knowledge-base search above, give your final "
            "spoken-language reply to the customer's actual question now -- state the "
            "relevant amounts/dates/status directly. Do not call any more tools, do not add "
            "unrelated remarks, and do not claim to have done something no tool actually did."
        )
    )
    final = llm.invoke(messages)
    reply = (final.content or "").strip() or localized(HANDOFF_MESSAGE_TEMPLATES, language)
    if show_debug:
        print(f"  [LLM wrap-up reply] {reply}")
    return reply


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
