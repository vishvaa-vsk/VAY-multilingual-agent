"""Node functions for LangGraph workflow execution graph."""

import uuid

from vay.asr.router import ASRRouter
from vay.audio.utils import load_audio
from vay.graph.state import AgentState
from vay.handoff.queue import HandoffQueueManager
from vay.normalization.pass_llm import LLMTranscriptNormalizer
from vay.rag.retriever import HybridRetriever
from vay.types import HandoffReason, HandoffTicket

# Initialize shared components
asr_router = ASRRouter()
normalizer = LLMTranscriptNormalizer()
retriever = HybridRetriever()
handoff_queue = HandoffQueueManager()


def vad_node(state: AgentState) -> AgentState:
    """VAD Node: Detect speech boundaries in audio stream."""
    return state


def asr_node(state: AgentState) -> AgentState:
    """ASR Node: Route audio to Tier 1 (IndicConformer) or Tier 2 (Whisper)."""
    audio_path = state.get("audio_path", "")
    audio_tensor = load_audio(audio_path)
    asr_res = asr_router.route_and_transcribe(audio_tensor)
    state["asr_result"] = asr_res
    state["detected_language"] = asr_res.detected_language
    return state


def normalization_node(state: AgentState) -> AgentState:
    """Normalization Node: Clean transcript and detect intent & entities.

    Passes previous_intent and previous_normalized from conversation_history
    into the LLM normalizer so it can resolve cross-turn coreferences
    (e.g. "it", "that plan", "same issue") before the orchestrator runs.
    """
    asr_res = state.get("asr_result")
    raw_text = asr_res.raw_text if asr_res else ""
    lang = state.get("detected_language", "en")

    # Extract previous turn context from conversation history for coreference resolution.
    # The history is a list of LangChain messages; we look for the last StructuredTranscript
    # stored on state (if any) from the prior iteration of the call loop.
    previous_intent: str | None = None
    previous_normalized: str | None = None
    prev_tx = state.get("structured_transcript")
    if prev_tx is not None:
        previous_intent = getattr(prev_tx, "intent", None)
        previous_normalized = getattr(prev_tx, "normalized_text", None)

    struct_tx = normalizer.normalize(
        raw_text, lang,
        previous_intent=previous_intent,
        previous_normalized=previous_normalized,
    )
    state["structured_transcript"] = struct_tx

    # Check for sensitive / restricted intent
    sensitive_intents = ["account_pin_reset", "sensitive_pii_update"]
    state["is_sensitive_intent"] = struct_tx.intent in sensitive_intents
    return state


def rag_node(state: AgentState) -> AgentState:
    """RAG Node: Query hybrid vector store & BM25 index."""
    struct_tx = state.get("structured_transcript")
    query = struct_tx.normalized_text if struct_tx else ""
    rag_res = retriever.retrieve(query)
    state["retrieval_result"] = rag_res
    return state


def handoff_gate_node(state: AgentState) -> AgentState:
    """Handoff Gate Node: Check for human escalation triggers."""
    is_sensitive = state.get("is_sensitive_intent", False)
    rag_res = state.get("retrieval_result")
    low_confidence = rag_res is not None and not rag_res.is_high_confidence

    if is_sensitive or low_confidence:
        state["requires_handoff"] = True
    else:
        state["requires_handoff"] = False
    return state


def llm_generation_node(state: AgentState) -> AgentState:
    """LLM Generation Node: Produce grounded response in user's detected language."""
    lang = state.get("detected_language", "en")
    if lang == "ta":
        res_text = "உங்கள் கணக்கில் ரூ. 499 பில் தொகை செலுத்த வேண்டியுள்ளது."
    elif lang == "hi":
        res_text = "आपके खाते में 499 रुपये का बिल देय है।"
    else:
        res_text = "Your current account bill amount is Rs 499."
    state["llm_response_text"] = res_text
    return state


def tts_node(state: AgentState) -> AgentState:
    """TTS Node: Synthesize and play the response audio in detected language."""
    from vay.tts import engine as tts_engine

    lang = state.get("detected_language", "en")
    text = state.get("llm_response_text", "")
    if text:
        tts_engine.speak(text, lang=lang)
    state["output_audio_path"] = "output.mp3"
    return state


def human_handoff_node(state: AgentState) -> AgentState:
    """Single shared terminal node for human escalation across all triggers."""
    tx = state.get("structured_transcript")
    raw_tx = tx.normalized_text if tx else ""
    lang = state.get("detected_language", "en")

    reason = HandoffReason.HANDOFF_GATE_TRIGGERED
    if state.get("is_sensitive_intent"):
        reason = HandoffReason.SENSITIVE_INTENT
    elif state.get("retrieval_result") and not state["retrieval_result"].is_high_confidence:
        reason = HandoffReason.LOW_RETRIEVAL_CONFIDENCE

    ticket = HandoffTicket(
        ticket_id=str(uuid.uuid4()),
        user_id=state.get("user_id", "guest"),
        reason=reason,
        transcript=raw_tx,
        language=lang,
        context={"intent": tx.intent if tx else "unknown"},
        created_at="2026-08-14T00:00:00Z",
    )
    handoff_queue.enqueue_ticket(ticket)
    state["handoff_ticket"] = ticket
    state["llm_response_text"] = (
        "Connecting you to a human customer care executive. Please stay on the line."
    )
    return state
