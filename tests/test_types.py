"""Tests for domain data types and validation schemas."""

from vay.types import ASRResult, LanguageTier, StructuredTranscript


def test_asr_result_creation() -> None:
    result = ASRResult(
        raw_text="வணக்கம்",
        detected_language="ta",
        language_tier=LanguageTier.TIER_1,
        model_used="ai4bharat/indic-conformer-600m-multilingual",
    )
    assert result.detected_language == "ta"
    assert result.language_tier == LanguageTier.TIER_1


def test_structured_transcript_creation() -> None:
    st = StructuredTranscript(
        original_text="வணக்கம் பில்",
        normalized_text="வணக்கம் பில்",
        detected_language="ta",
        intent="bill_query",
    )
    assert st.intent == "bill_query"
    assert st.detected_language == "ta"
