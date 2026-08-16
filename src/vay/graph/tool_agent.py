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

import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from vay.graph.utils import localized

MAX_TOOL_ITERATIONS = 6

# Repetition-loop guard on the SUB-AGENT's generated reply.
#
# llama-3.1-8b-instant is prone to phrase/sentence repetition when generating
# non-English (Tamil, Hindi) replies about factual/numeric content.  Two guards:
#
# Guard 1 — _REPEAT_RE: catches any 12–350 char span that repeats 2+ times
#   back-to-back.  Upper bound raised from 80→350 to catch full Tamil/Hindi
#   sentences (e.g. "நாங்கள் உங்களுக்கு உதவ முடியாத சில விஷயங்கள் உள்ளன."
#   is ~70 UTF-8 bytes but >80 chars in codepoints on some builds — cap at 350
#   to be safe).
#
# Guard 2 — _dedup_sentences: splits on sentence boundaries and removes any
#   sentence that has already appeared earlier in the reply (case-insensitive,
#   whitespace-normalised).  This catches the paragraph-level repeat pattern
#   where the *same full sentence* appears 8+ times spread across multiple
#   paragraphs, which _REPEAT_RE misses because the sentences aren't immediately
#   adjacent (there may be punctuation/newlines between occurrences).
_REPEAT_RE = re.compile(r"(.{12,350}?)\1{1,}", re.DOTALL)

# Sentence splitter: split on . ! ? followed by whitespace or end-of-string,
# keeping the delimiter attached to the preceding token.
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?।।])\s+")


def _dedup_sentences(text: str) -> str:
    """Remove duplicate sentences (case/whitespace-insensitive) from a reply.

    Preserves the first occurrence of each unique sentence; strips any that
    appear again later.  Paragraph breaks (double newlines) are preserved
    around the surviving sentences.
    """
    if not text:
        return text
    paragraphs = text.split("\n\n")
    seen: set[str] = set()
    out_paragraphs: list[str] = []
    for para in paragraphs:
        sentences = _SENT_SPLIT_RE.split(para.strip())
        kept: list[str] = []
        for sent in sentences:
            key = " ".join(sent.lower().split())
            if key and key not in seen:
                seen.add(key)
                kept.append(sent.strip())
        if kept:
            out_paragraphs.append(" ".join(kept))
    return "\n\n".join(out_paragraphs)


# Sentence-terminal punctuation: full stop, !, ?, Devanagari danda,
# Tamil/Telugu punctuation, ellipsis.  Used to detect truncated fragments.
_TERMINAL_PUNCT_RE = re.compile(r"[.!?\u0964\u0965\u0be6-\u0bf2\u2026]\s*$")
# Non-terminal characters that signal the reply was cut mid-sentence.
_FRAGMENT_END_RE = re.compile(r"[,;:\-–—\/]\s*$")


def _is_complete_reply(text: str) -> bool:
    """Return True when *text* looks like at least one complete sentence.

    A reply is considered incomplete (fragment) when it:
    - ends with a non-terminal character (comma, semicolon, dash, colon)
    - contains no terminal punctuation at all AND is shorter than 80 chars
      (a very short non-punctuated phrase is almost certainly a fragment)

    This is used by _detoxify_repetition to decide whether a truncated result
    should be returned or discarded (→ caller uses fallback template).
    """
    if not text:
        return False
    # Ends with a non-terminal character → definitely a fragment
    if _FRAGMENT_END_RE.search(text):
        return False
    # Ends with terminal punctuation → complete
    if _TERMINAL_PUNCT_RE.search(text):
        return True
    # No terminal punctuation but long enough to be a standalone phrase
    return len(text) >= 80


def _detoxify_repetition(text: str) -> str:
    """Two-stage repetition filter for LLM output.

    Stage 1 (_REPEAT_RE): catches short-to-medium span back-to-back repeats
      (12–350 chars).  Truncates right before the first repeated block.  This
      handles word/phrase loops like \"...299 ரூபாய் வரையிலான \" × 30.

    Stage 2 (_dedup_sentences): removes any sentence that appears more than
      once across the whole reply, regardless of adjacency.  This is the
      primary defence for the paragraph-spread pattern seen in Tamil/Hindi
      where the LLM repeats a full sentence 8+ times across separate paragraphs
      — Stage 1 may not catch these if the repeated sentences aren't strictly
      adjacent (there may be a period or newline between them).

    Fragment guard: if the result after truncation is an incomplete sentence
      (ends with comma/colon/dash or has no terminal punctuation and is very
      short), returns "" so the caller falls back to the localized handoff
      template rather than speaking a grammatically broken fragment.
    """
    if not text:
        return text

    # Stage 1: truncate on back-to-back short/medium span repeats
    match = _REPEAT_RE.search(text)
    if match:
        truncated = text[: match.start()].rstrip()
        # If truncation cuts too early (< 40 chars), keep first occurrence instead
        candidate = truncated if len(truncated) >= 40 else text[: match.end(1)].rstrip()
        # Fragment guard: if the candidate doesn't look like a complete sentence,
        # discard it entirely so the caller's fallback template is used.
        if not _is_complete_reply(candidate):
            return ""
        text = candidate

    # Stage 2: sentence-level dedup (primary defence for Tamil/Hindi loops)
    text = _dedup_sentences(text)

    return text.strip()

HANDOFF_MESSAGE_TEMPLATES = {
    "en": (
        "I want to make sure I get this right for you, and I'm not fully confident I can "
        "help with that myself right now. Let me connect you with a live Nexatel agent who "
        "can take it from here."
    ),
    "hi": (
        "मैं चाहता हूँ कि आपकी सही तरीके से मदद हो, और अभी मुझे पूरा भरोसा नहीं है कि मैं इसे "
        "खुद संभाल पाऊँगा। मैं आपको Nexatel के एक लाइव एजेंट से जोड़ रहा हूँ जो आगे मदद करेंगे।"
    ),
    "ta": (
        "உங்களுக்குச் சரியாக உதவ விரும்புகிறேன், ஆனால் இதை என்னால் இப்போது சரியாகக் கையாள "
        "முடியுமா என்பதில் முழு நம்பிக்கை இல்லை. இதைத் தொடர்ந்து கவனிக்க ஒரு நேரடி Nexatel "
        "முகவரிடம் உங்களை இணைக்கிறேன்."
    ),
}

TOOL_LOOP_FAILURE_TEMPLATES = {
    "en": "I'm not fully sure I can complete that here -- let me connect you with a human agent.",
    "hi": "मुझे पूरा यकीन नहीं है कि मैं इसे यहाँ पूरा कर पाऊँगा — मैं आपको एक मानव एजेंट से जोड़ता हूँ।",
    "ta": "இதை என்னால் இங்கு முழுமையாக முடிக்க முடியுமா என்று உறுதியாக இல்லை — நான் உங்களை ஒரு மனித முகவரிடம் இணைக்கிறேன்.",
}


def run_tool_agent(
    llm: Any,

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

    # Dedup guard: a small model will sometimes retry the exact same (or a near-identical)
    # tool call several times in a row when the first result wasn't what it wanted --
    # e.g. re-issuing search_product_catalog with trivially reworded queries, burning
    # the whole MAX_TOOL_ITERATIONS budget (and real latency/token cost) without ever
    # finding a different answer. Track exact (name, args) signatures seen this turn; a
    # repeat is answered locally without a further tool invocation or LLM round-trip,
    # and nudges the model to try something different or wrap up instead of looping.
    seen_calls: dict[str, int] = {}

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
            reply = _detoxify_repetition(reply) or localized(HANDOFF_MESSAGE_TEMPLATES, language)
            print(f"  [SubAgent] Final reply (len={len(reply)}): {reply[:200]}")
            return reply

        if show_debug and (ai_msg.content or "").strip():
            # Some models emit reasoning/commentary text alongside tool_calls -- show it so a
            # garbled/off-topic final reply is traceable to what the LLM actually said.
            print(f"  [LLM message] {ai_msg.content.strip()[:500]}")

        for call in ai_msg.tool_calls:
            # json.dumps (not a tuple of dict items) so unhashable arg values -- e.g.
            # comparePlans(plan_ids=[...]) -- don't crash the signature computation.
            signature = call["name"] + "|" + json.dumps(call["args"], sort_keys=True, default=str)
            seen_calls[signature] = seen_calls.get(signature, 0) + 1

            if seen_calls[signature] > 1:
                result = (
                    "You already called this exact tool with these exact arguments and got "
                    "a result -- calling it again will return the same thing. Either use a "
                    "meaningfully different query/argument, or stop searching and answer with "
                    "what you already have (say plainly if the exact answer wasn't found)."
                )
                if show_debug:
                    print(f"  [tool call SKIPPED (duplicate #{seen_calls[signature]})] {call['name']}({call['args']})")
            else:
                tool_fn = tools_by_name.get(call["name"])
                if tool_fn is None:
                    result = f"Unknown tool: {call['name']}"
                else:
                    try:
                        result = tool_fn.invoke(call["args"])
                    except Exception as e:
                        result = f"Tool error: {e}"
                print(f"  [SubAgent tool] {call['name']}({call['args']}) -> {str(result)[:300]}")

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
    reply = _detoxify_repetition(reply) or localized(HANDOFF_MESSAGE_TEMPLATES, language)
    if show_debug:
        print(f"  [LLM wrap-up reply] {reply}")
    return reply


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
