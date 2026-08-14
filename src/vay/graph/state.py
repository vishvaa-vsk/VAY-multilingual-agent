"""TypedDict state schema for LangGraph voice assistant workflow."""

from typing import Any, TypedDict

from vay.types import ASRResult, HandoffTicket, RetrievalResult, StructuredTranscript


class AgentState(TypedDict, total=False):
    """Complete conversation state for LangGraph pipeline execution."""

    user_id: str
    audio_path: str
    detected_language: str
    asr_result: ASRResult
    structured_transcript: StructuredTranscript
    is_sensitive_intent: bool
    retrieval_result: RetrievalResult
    requires_handoff: bool
    handoff_ticket: HandoffTicket
    llm_response_text: str
    output_audio_path: str
    error: str
    context: dict[str, Any]
