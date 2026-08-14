"""LLM-based transcript normalization, code-switching cleanup (Tanglish/Hinglish),
and intent parsing.
"""

from vay.types import StructuredTranscript


class LLMTranscriptNormalizer:
    """Normalizes raw ASR transcript, performs code-switch resolution,
    intent detection, and entity extraction.
    """

    def normalize(self, raw_transcript: str, language: str) -> StructuredTranscript:
        """Process raw ASR output and generate structured output.

        Args:
            raw_transcript: Raw text output from ASR model.
            language: Detected ISO language code.

        Returns:
            StructuredTranscript object.
        """
        # Clean text
        cleaned_text = raw_transcript.strip()

        # Simple mock intent classification
        intent = "bill_query"
        if "plan" in cleaned_text.lower() or "பிளான்" in cleaned_text:
            intent = "plan_change"
        elif "complaint" in cleaned_text.lower() or "புகார்" in cleaned_text:
            intent = "complaint"

        return StructuredTranscript(
            original_text=raw_transcript,
            normalized_text=cleaned_text,
            detected_language=language,
            intent=intent,
            entities={"category": "billing"},
            confidence=0.9,
        )
