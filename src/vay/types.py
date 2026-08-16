"""Core domain data models and type definitions."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class LanguageTier(StrEnum):
    """Supported language processing tiers."""

    TIER_1 = "tier_1"  # Tamil, Hindi (IndicConformer)
    TIER_2 = "tier_2"  # English and fallback (Whisper)


class ASRResult(BaseModel):
    """Raw transcription output from an ASR engine."""

    raw_text: str
    detected_language: str
    language_tier: LanguageTier
    confidence: float = 1.0
    model_used: str


class StructuredTranscript(BaseModel):
    """Normalized transcript after LLM cleanup."""

    original_text: str
    normalized_text: str
    detected_language: str
    intent: str
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 1.0


class Document(BaseModel):
    """Retrieved knowledge base document."""

    id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class RetrievalResult(BaseModel):
    """RAG retrieval result containing top matches and aggregate confidence score."""

    query: str
    documents: list[Document]
    confidence_score: float
    is_high_confidence: bool


class HandoffReason(StrEnum):
    """Triggers for escalating to human agent."""

    SENSITIVE_INTENT = "sensitive_intent"
    LOW_RETRIEVAL_CONFIDENCE = "low_retrieval_confidence"
    HANDOFF_GATE_TRIGGERED = "handoff_gate_triggered"


class HandoffTicket(BaseModel):
    """Escalation ticket for human customer care dashboard."""

    ticket_id: str
    user_id: str
    reason: HandoffReason
    transcript: str
    language: str
    context: dict[str, Any] = Field(default_factory=dict)
    created_at: str
