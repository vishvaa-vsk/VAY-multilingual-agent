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
import logging
import re
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from vay.graph.utils import localized

MAX_TOOL_ITERATIONS = 6

# ---------------------------------------------------------------------------
# Groq HTTP/retry visibility
# ---------------------------------------------------------------------------
# ChatGroq is configured with max_retries=3 (core_utils._llm). That retry logic
# lives INSIDE the groq SDK's HTTP client and is otherwise completely silent --
# a 429 (rate limit) or 5xx there just makes the surrounding llm.invoke() call
# take longer with exponential backoff, and nothing in our own logs explains
# why (e.g. "[SubAgent] Took 31.54s" with nothing printed in between). Turn on
# the SDK's own request/retry logging so a slow call shows its actual cause
# (HTTP status code + "Retrying request in N seconds") instead of looking like
# an unexplained stall.
_groq_http_logger = logging.getLogger("httpx")
_groq_sdk_logger = logging.getLogger("groq")
if not _groq_http_logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("  [Groq HTTP] %(message)s"))
    _groq_http_logger.addHandler(_handler)
    _groq_http_logger.setLevel(logging.INFO)
    _groq_http_logger.propagate = False
    _groq_sdk_logger.addHandler(_handler)
    # INFO, not DEBUG: DEBUG on the groq SDK logger dumps the full raw request
    # body -- including the raw audio bytes on STT calls -- which floods the
    # console with megabytes of binary per request. INFO still surfaces
    # "Retrying request ... in N seconds" backoff lines, which is all we need.
    _groq_sdk_logger.setLevel(logging.INFO)
    _groq_sdk_logger.propagate = False


def _timed_invoke(llm: Any, messages: list, label: str) -> Any:
    """llm.invoke(messages) with an elapsed-time print, so a slow sub-agent
    turn can be pinned on a specific LLM round-trip (and, via the httpx/groq
    logging above, on a specific 429/backoff) instead of only showing up as
    one big unexplained total at the end."""
    start = time.monotonic()
    try:
        result = llm.invoke(messages)
        print(f"  [SubAgent LLM call: {label}] took {time.monotonic() - start:.2f}s")
        return result
    except Exception:
        print(f"  [SubAgent LLM call: {label}] FAILED after {time.monotonic() - start:.2f}s")
        raise

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

    # Post-processing: force English telecom jargon that small models stubbornly translate
    text = text.replace("தரவு", "data").replace("அழைப்புகள்", "calls").replace("வாலிடிடி", "validity").replace("அலகுகள்", "packs")

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


# ---------------------------------------------------------------------------
# Language-conformance guard
# ---------------------------------------------------------------------------
#
# SUBAGENT_SYSTEM_PROMPT_TEMPLATE (core_utils.py) instructs the LLM to "write your ENTIRE
# final reply strictly in {language}", but a small model (llama-3.1-8b-instant) does not
# reliably obey that -- it will silently answer in English even when {language} is "hi"/"ta"
# and the customer's own turn was correctly detected in that language. Nothing previously
# checked the OUTPUT actually landed in the right script before it went to TTS. This is a
# deterministic, code-level backstop: for languages with a distinct Unicode script, verify
# the reply actually contains at least one character from that script; if not, force one
# translation-only retry (no tools, no rephrasing of content) before returning.
_SCRIPT_RANGES: dict[str, tuple[int, int]] = {
    "hi": (0x0900, 0x097F),  # Devanagari (Hindi)
    "mr": (0x0900, 0x097F),  # Devanagari (Marathi)
    "ta": (0x0B80, 0x0BFF),  # Tamil
    "te": (0x0C00, 0x0C7F),  # Telugu
    "kn": (0x0C80, 0x0CFF),  # Kannada
    "ml": (0x0D00, 0x0D7F),  # Malayalam
    "gu": (0x0A80, 0x0AFF),  # Gujarati
    "pa": (0x0A00, 0x0A7F),  # Gurmukhi (Punjabi)
    "bn": (0x0980, 0x09FF),  # Bengali
    "ur": (0x0600, 0x06FF),  # Arabic script (Urdu)
    "ar": (0x0600, 0x06FF),  # Arabic
    "ja": (0x3040, 0x30FF),  # Hiragana/Katakana (kanji overlaps CJK, this is enough of a signal)
    "ko": (0xAC00, 0xD7A3),  # Hangul
    "zh": (0x4E00, 0x9FFF),  # CJK Unified Ideographs
}


def _script_conforms(text: str, language: str) -> bool:
    """True if *text* contains at least one character in *language*'s script, or if
    *language* has no script mapping here (e.g. "en", or a Latin-script language not
    listed -- nothing to check, assume compliant)."""
    rng = _SCRIPT_RANGES.get(language)
    if not rng or not text:
        return True
    lo, hi = rng
    return any(lo <= ord(ch) <= hi for ch in text)


def _enforce_language(reply: str, language: str, llm: Any, show_debug: bool = False) -> str:
    """If *reply* doesn't conform to *language*'s script, force a translation-only retry."""
    if not reply or language == "en" or _script_conforms(reply, language):
        return reply
    print(f"  [LanguageGuard] reply not in expected script for '{language}' -- forcing translation")
    try:
        translated = _timed_invoke(
            llm,
            [
                SystemMessage(
                    content=(
                        f"Translate the following customer-support reply into {language}. "
                        "Keep telecom terms in English as customers naturally hear them "
                        "(data, plan, validity, recharge, balance, calls, SMS, OTP, SIM, "
                        "eSIM, VoLTE, 4G, 5G, brand/plan names). Output ONLY the translated "
                        "reply, nothing else -- no preamble, no quotes."
                    )
                ),
                HumanMessage(content=reply),
            ],
            "language-guard translation retry",
        ).content.strip()
    except Exception as e:
        print(f"  [LanguageGuard] translation retry failed: {e}")
        return reply
    if show_debug:
        print(f"  [LanguageGuard] translated -> {translated[:200]}")
    return _detoxify_repetition(translated) or reply


# ---------------------------------------------------------------------------
# Near-duplicate query guard
# ---------------------------------------------------------------------------
#
# The exact-signature dedup below only catches byte-identical repeats. In
# practice, a small model more often retries a FAILED RAG search with a
# barely-reworded query instead of a materially different one -- e.g.
# 'travel plan recharge requirement postpaid' -> '...postpaid travel add-on'
# -> 'Travel Pack recharge requirement' -- three near-identical searches, none
# scoring well, that each cost a full LLM+tool round-trip (Groq latency +
# growing context) without ever finding a better answer. This guard catches
# that pattern via token-overlap similarity on any free-text "query" argument,
# scoped per tool name, so the model gets nudged to diversify or wrap up
# instead of burning the rest of MAX_TOOL_ITERATIONS on rewordings.
_QUERY_WORD_RE = re.compile(r"[\w']+")


def _query_tokens(text: str) -> set[str]:
    return set(_QUERY_WORD_RE.findall(text.lower()))


def _is_near_duplicate_query(a: str, b: str, threshold: float = 0.5) -> bool:
    """True if two free-text queries are close reworks of each other (Jaccard
    token overlap), even when not byte-identical."""
    tokens_a, tokens_b = _query_tokens(a), _query_tokens(b)
    if not tokens_a or not tokens_b:
        return False
    overlap = len(tokens_a & tokens_b) / len(tokens_a | tokens_b)
    return overlap >= threshold


def run_tool_agent(
    llm: Any,

    tools: list,
    system_prompt: str,
    user_text: str,
    history: list,
    language: str = "en",
    show_debug: bool = False,
) -> tuple[str, bool]:
    """Minimal bounded tool-calling loop: LLM <-> tools until it stops calling
    tools or MAX_TOOL_ITERATIONS is hit.

    Returns (reply, degraded) -- `degraded` is True whenever `reply` is one of the
    generic HANDOFF_MESSAGE_TEMPLATES/TOOL_LOOP_FAILURE_TEMPLATES fallbacks (LLM call
    failed outright, or came back with nothing usable) rather than a real answer. The
    fallback TEXT alone used to be the only signal of this -- callers had no reliable,
    language-independent way to tell "sub-agent genuinely answered" from "sub-agent gave
    up and is punting to a human", so graph state's `handoff` flag never got set for
    this path and the front end never learned the call needs to hand off (see caller).
    """
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
    # Tool name -> list of raw "query" strings already tried this turn, for the
    # near-duplicate guard below (separate from the exact-signature dedup above).
    seen_queries: dict[str, list[str]] = {}

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            ai_msg: AIMessage = _timed_invoke(
                bound_llm, messages, f"tool-loop iter {iteration + 1}/{MAX_TOOL_ITERATIONS}"
            )
        except Exception as e:
            # The model hallucinated something the Groq API rejected outright before we
            # ever got a normal response back (e.g. calling an unregistered tool name) --
            # degrade to a safe, guardrail-recognized reply instead of crashing the call.
            print(f"  [ERROR] tool-calling LLM call failed, degrading to handoff: {e}")
            return localized(TOOL_LOOP_FAILURE_TEMPLATES, language), True
        messages.append(ai_msg)

        if not getattr(ai_msg, "tool_calls", None):
            raw_content = (ai_msg.content or "").strip()
            degraded = not raw_content
            reply = raw_content or localized(HANDOFF_MESSAGE_TEMPLATES, language)
            detoxified = _detoxify_repetition(reply)
            if not detoxified:
                degraded = True
            reply = detoxified or localized(HANDOFF_MESSAGE_TEMPLATES, language)
            reply = _enforce_language(reply, language, llm, show_debug)
            print(f"  [SubAgent] Final reply (len={len(reply)}): {reply[:200]}")
            return reply, degraded

        if show_debug and (ai_msg.content or "").strip():
            # Some models emit reasoning/commentary text alongside tool_calls -- show it so a
            # garbled/off-topic final reply is traceable to what the LLM actually said.
            print(f"  [LLM message] {ai_msg.content.strip()[:500]}")

        for call in ai_msg.tool_calls:
            # json.dumps (not a tuple of dict items) so unhashable arg values -- e.g.
            # comparePlans(plan_ids=[...]) -- don't crash the signature computation.
            signature = call["name"] + "|" + json.dumps(call["args"], sort_keys=True, default=str)
            seen_calls[signature] = seen_calls.get(signature, 0) + 1
            is_exact_repeat = seen_calls[signature] > 1

            # Near-duplicate check: only meaningful the first time this exact
            # signature is seen (an exact repeat is already caught above), and
            # only for calls carrying a free-text "query" argument (the RAG
            # search tools).
            is_near_repeat = False
            query_text = call["args"].get("query") if isinstance(call["args"], dict) else None
            if not is_exact_repeat and isinstance(query_text, str) and query_text.strip():
                prior_queries = seen_queries.setdefault(call["name"], [])
                is_near_repeat = any(
                    _is_near_duplicate_query(query_text, prior) for prior in prior_queries
                )
                prior_queries.append(query_text)

            if is_exact_repeat or is_near_repeat:
                result = (
                    "You already searched for this or something very similar and got a "
                    "result -- calling it again with slightly different wording will return "
                    "the same thing. Either use a meaningfully different query/argument, or "
                    "stop searching and answer with what you already have (say plainly if the "
                    "exact answer wasn't found)."
                )
                if show_debug:
                    reason = "exact duplicate" if is_exact_repeat else "near-duplicate query"
                    print(f"  [tool call SKIPPED ({reason})] {call['name']}({call['args']})")
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
                return result_str[len("STOP_AND_SAY:") :].strip(), False

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
    final = _timed_invoke(llm, messages, "forced wrap-up")
    raw_content = (final.content or "").strip()
    degraded = not raw_content
    reply = raw_content or localized(HANDOFF_MESSAGE_TEMPLATES, language)
    detoxified = _detoxify_repetition(reply)
    if not detoxified:
        degraded = True
    reply = detoxified or localized(HANDOFF_MESSAGE_TEMPLATES, language)
    reply = _enforce_language(reply, language, llm, show_debug)
    if show_debug:
        print(f"  [LLM wrap-up reply] {reply}")
    return reply, degraded


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------
