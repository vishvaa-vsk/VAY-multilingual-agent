"""
pass_llm.py

LLM-based transcript normalization, code-switching cleanup (Tanglish/Hinglish),
intent detection, and entity extraction.

The normalizer uses the Groq LLM (same model as the rest of the pipeline) to
clean up raw ASR output and produce a StructuredTranscript.  Previous-turn
intent and normalized text are injected into the system prompt so the LLM can
resolve cross-turn coreferences ("it", "that plan", "same issue") before the
orchestrator ever sees the query.

Per project_context.md §5.4: this normalization pass is NOT optional polish —
it is the core fix for code-switched Tanglish/Hinglish ASR output.
"""

from __future__ import annotations

import json
import os
import re

from dotenv import load_dotenv

load_dotenv(override=True)

from vay.types import StructuredTranscript

_GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
_GROQ_MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-20b")

# Cached LLM instance (lazy-initialized)
_llm_instance = None


def _get_llm():
    """Lazy-initialise a ChatGroq instance (avoids import-time error when the
    key isn't set yet, e.g. in unit tests that mock this module)."""
    global _llm_instance
    if _llm_instance is None:
        if not _GROQ_API_KEY:
            return None
        try:
            from langchain_groq import ChatGroq

            _llm_instance = ChatGroq(
                model=_GROQ_MODEL,
                api_key=_GROQ_API_KEY,
                temperature=0.1,
                max_retries=2,
                reasoning_effort="low",
            )
        except Exception:
            return None
    return _llm_instance


def _extract_json(text: str) -> dict | None:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


def _fallback_normalize(raw_transcript: str, language: str) -> StructuredTranscript:
    """Simple keyword-based fallback used when the LLM is unavailable."""
    cleaned = raw_transcript.strip()
    intent = "bill_query"
    text_lower = cleaned.lower()
    if "plan" in text_lower or "பிளான்" in cleaned or "प्लान" in cleaned:
        intent = "plan_change"
    elif any(w in text_lower for w in ("complaint", "problem", "issue", "not working")):
        intent = "complaint"
    elif any(w in text_lower for w in ("coverage", "signal", "network", "4g", "5g")):
        intent = "coverage_query"
    return StructuredTranscript(
        original_text=raw_transcript,
        normalized_text=cleaned,
        detected_language=language,
        intent=intent,
        entities={"category": "general"},
        confidence=0.5,
    )


# System prompt for the normalization LLM call.
# Previous turn context is injected at runtime via .format().
_NORMALIZATION_SYSTEM_PROMPT = """\
You are an ASR transcript normalizer for a multilingual telecom customer care system.

Your job is to clean up raw ASR output (which may contain code-switching errors, Tanglish,
Hinglish, repetitions, hallucinations, or mis-transcribed words) and extract structured intent.

{previous_context}

Output STRICT JSON ONLY (no prose, no markdown fences):
{{
  "normalized_text": "<Clean, grammatical version of what the customer said. Fix obvious ASR \
errors. Resolve code-switches to the dominant language. Keep technical telecom terms (1GB, 4G, \
VoLTE, SIM) in English regardless of base language.>",
  "intent": "<Short snake_case label, e.g.: bill_query, plan_change, complaint, coverage_query, \
payment_query, plan_info, add_on_query, technical_support, account_query, call_end, unclear>",
  "entities": {{"<entity>": "<value>"}},
  "confidence": <float 0.0-1.0 for how confident you are in the normalization>,
  "detected_language": "<ISO 639-1 code of the language the customer is primarily speaking>"
}}

Rules:
- Fix ASR hallucinations: remove repetitions, gibberish words that break the sentence.
- For Tanglish (Tamil+English), produce a clean Tamil sentence with English technical terms.
- For Hinglish (Hindi+English), produce a clean Hindi sentence with English technical terms.
- Use the previous intent and normalized text (if provided) to resolve references like
  "it", "that plan", "same issue", "the one you mentioned" before writing normalized_text.
- Output ONLY the JSON object, nothing else.
"""

_PREVIOUS_CONTEXT_BLOCK = """\
Previous turn context (use this to resolve references in the current utterance):
  - Previous intent: {previous_intent}
  - Previous customer utterance (normalized): {previous_normalized}
"""

_NO_PREVIOUS_CONTEXT = "This is the first turn of the call — no previous context."


class LLMTranscriptNormalizer:
    """Normalizes raw ASR transcript using a Groq LLM call.

    Falls back to a simple keyword-based normalizer if the LLM is unavailable
    (no API key, network error, etc.) so the pipeline never hard-fails.
    """

    def normalize(
        self,
        raw_transcript: str,
        language: str,
        previous_intent: str | None = None,
        previous_normalized: str | None = None,
    ) -> StructuredTranscript:
        """Process raw ASR output and generate a StructuredTranscript.

        Args:
            raw_transcript:     Raw text output from the ASR model.
            language:           Detected ISO language code from the ASR stage.
            previous_intent:    Intent label from the previous turn (or None on
                                the first turn).  Used to resolve coreferences.
            previous_normalized: The normalized text from the previous turn (or
                                None on the first turn).
        """
        llm = _get_llm()
        if llm is None:
            return _fallback_normalize(raw_transcript, language)

        # Build the system prompt with/without previous turn context
        if previous_intent and previous_normalized:
            prev_block = _PREVIOUS_CONTEXT_BLOCK.format(
                previous_intent=previous_intent,
                previous_normalized=previous_normalized,
            )
        else:
            prev_block = _NO_PREVIOUS_CONTEXT

        system_prompt = _NORMALIZATION_SYSTEM_PROMPT.format(previous_context=prev_block)

        user_message = (
            f"Language hint from ASR: {language}\n"
            f"Raw ASR transcript:\n{raw_transcript}"
        )

        try:
            from langchain_core.messages import HumanMessage, SystemMessage

            response = llm.invoke(
                [SystemMessage(content=system_prompt), HumanMessage(content=user_message)]
            )
            parsed = _extract_json(response.content or "")
        except Exception as e:
            print(f"  [Normalization LLM call failed: {e}]")
            return _fallback_normalize(raw_transcript, language)

        if not parsed:
            return _fallback_normalize(raw_transcript, language)

        return StructuredTranscript(
            original_text=raw_transcript,
            normalized_text=parsed.get("normalized_text", raw_transcript).strip(),
            detected_language=parsed.get("detected_language", language),
            intent=parsed.get("intent", "unclear"),
            entities=parsed.get("entities") or {},
            confidence=float(parsed.get("confidence", 0.7)),
        )
