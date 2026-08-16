"""Two-tier language routing mechanism for ASR execution.

Routing strategy (single-pass, low-latency)
--------------------------------------------
1. Call ``whisper_asr.transcribe_auto()`` — ONE Groq API call that returns
   BOTH the detected language code AND the transcribed text from Whisper's
   verbose_json response.

2. If the detected language is a Tier-1 Indic language (hi, ta, bn, …):
   - Re-transcribe with IndicConformer for higher accuracy on Indian speech.
   - The Whisper transcription from step 1 is available as a fallback if
     IndicConformer fails.

3. If the detected language is Tier-2 (en + 90 other languages):
   - Return the Whisper result directly — NO second API call needed.

Previous design (two-call pattern, removed)
-------------------------------------------
The old router called ``detect_language()`` (one Groq call) and then called
``transcribe()`` again with the detected code (a second Groq call).  For all
Tier-2 languages this doubled per-turn latency for no benefit.  The new
single-pass approach cuts that to one call and uses the full-utterance audio
for both detection and transcription, which also improves detection accuracy.

Language lock (removed)
-----------------------
The original ``locked_language`` field permanently locked the router to the
first detected language after 2 utterances and never cleared it — causing every
subsequent utterance to be routed to the wrong ASR model when the caller
switched languages.  The per-utterance state reset in ``_reset_utterance_state``
ensures each call gets a fresh detection pass.
"""

from __future__ import annotations

import torch

from vay.asr.indic import IndicConformerASR
from vay.asr.whisper import WhisperASR
from vay.config import settings
from vay.types import ASRResult


class ASRRouter:
    """Routes incoming audio to the appropriate ASR model based on detected language.

    Each call to ``route_and_transcribe`` is independent — language is detected
    fresh from the full utterance audio on every call.
    """

    def __init__(self) -> None:
        self.indic_asr = IndicConformerASR(model_id=settings.indic_asr_model)
        self.whisper_asr = WhisperASR(model_id=settings.whisper_asr_model)

        # Exposed for logging / external inspection only
        self.last_detected_language: str | None = None
        self.last_detected_confidence: float = 0.0

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------

    def reset_session(self) -> None:
        """Reset all per-session state.  Call when starting a new call."""
        self.last_detected_language = None
        self.last_detected_confidence = 0.0

    # ------------------------------------------------------------------
    # Core routing
    # ------------------------------------------------------------------

    def route_and_transcribe(
        self, audio_tensor: torch.Tensor, override_language: str | None = None
    ) -> ASRResult:
        """Detect language and transcribe in the fewest possible API calls.

        **Tier-2 (en + other languages):** Single Groq API call — Whisper's
        ``verbose_json`` response includes both the detected language code and
        the transcribed text, so no second call is needed.

        **Tier-1 (Indic languages):** One Groq call for detection/text, then
        IndicConformer (local model, no extra API call) for higher-accuracy
        transcription.  The Whisper text is used as a fallback if IndicConformer
        returns an empty result.

        Args:
            audio_tensor: 1-D float32 audio tensor at 16 kHz (mono).
            override_language: Hard-code the language (skips auto-detection).

        Returns:
            ASRResult from the selected ASR model.
        """
        if override_language:
            print(f"[Router] Language overridden to '{override_language}'.")
            return self._transcribe_with_lang(audio_tensor, override_language)

        # ------------------------------------------------------------------
        # SINGLE-PASS: one API call → language code + transcription text
        # ------------------------------------------------------------------
        whisper_result = self.whisper_asr.transcribe_auto(audio_tensor)
        detected_lang = whisper_result.detected_language
        confidence = whisper_result.confidence

        self.last_detected_language = detected_lang
        self.last_detected_confidence = confidence

        print(
            f"[Router] Detected language '{detected_lang}' "
            f"(confidence: {confidence:.2f}) from full utterance."
        )

        if detected_lang in settings.tier1_languages:
            # Re-transcribe with IndicConformer for Indic-language accuracy
            print(f"[Router] Language: {detected_lang} → IndicConformer (Tier 1)")
            indic_result = self.indic_asr.transcribe(audio_tensor, language=detected_lang)

            # Fallback: if IndicConformer returns nothing, use the Whisper text
            if not indic_result.raw_text.strip() and whisper_result.raw_text.strip():
                print(
                    "[Router] IndicConformer returned empty — "
                    "falling back to Whisper transcription."
                )
                return ASRResult(
                    raw_text=whisper_result.raw_text,
                    detected_language=detected_lang,
                    language_tier=whisper_result.language_tier,
                    confidence=whisper_result.confidence,
                    model_used=f"{whisper_result.model_used} (fallback)",
                )
            return indic_result

        else:
            # Tier-2: reuse the Whisper result from the detection pass
            # (no second API call)
            print(f"[Router] Language: {detected_lang} → Whisper (Tier 2, single-pass)")
            return whisper_result

    def _transcribe_with_lang(
        self, audio_tensor: torch.Tensor, language: str
    ) -> ASRResult:
        """Internal helper for override_language path."""
        if language in settings.tier1_languages:
            return self.indic_asr.transcribe(audio_tensor, language=language)
        return self.whisper_asr.transcribe(audio_tensor, language=language)
