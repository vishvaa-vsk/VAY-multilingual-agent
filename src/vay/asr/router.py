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

import torch

from vay.asr.indic import IndicConformerASR
from vay.asr.whisper import WhisperASR
from vay.config import settings
from vay.types import ASRResult


class ASRRouter:
    """Routes incoming audio to the appropriate ASR model based on detected language."""

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
        """Detect language and transcribe.

        **Language Detection:** Uses Whisper API for auto-detection.

        **Tier-1 (Indic languages):** Calls IndicConformer (local model). 
        Falls back to Whisper if IndicConformer returns an empty result.

        **Tier-2 (en + other languages):** Uses Groq Whisper API result directly.
        """
        if override_language:
            print(f"[Router] Language overridden to '{override_language}'.")
            return self._transcribe_with_lang(audio_tensor, override_language)

        # ------------------------------------------------------------------
        # STEP 1: Acoustic Language Detection & Transcription via Whisper
        # ------------------------------------------------------------------
        whisper_result = self.whisper_asr.transcribe_auto(audio_tensor)
        detected_lang = whisper_result.detected_language

        self.last_detected_language = detected_lang
        self.last_detected_confidence = whisper_result.confidence

        print(
            f"[Router] Detected language '{detected_lang}' "
            f"(confidence: {whisper_result.confidence:.2f}) via Whisper Auto-Detect."
        )

        # ------------------------------------------------------------------
        # STEP 2: Routing to ASR Model
        # ------------------------------------------------------------------
        if detected_lang in settings.tier1_languages:
            # Transcribe with IndicConformer for Indic-language accuracy
            print(f"[Router] Language: {detected_lang} → IndicConformer (Tier 1)")
            indic_result = self.indic_asr.transcribe(audio_tensor, language=detected_lang)

            # Fallback: if IndicConformer returns nothing, return Whisper result
            if not indic_result.raw_text.strip():
                print(
                    "[Router] IndicConformer returned empty — "
                    "falling back to Whisper API."
                )
                return whisper_result
            return indic_result

        else:
            # Tier-2: Return the Whisper API result we already got in Step 1
            print(f"[Router] Language: {detected_lang} → Whisper API (Tier 2)")
            return whisper_result

    def _transcribe_with_lang(
        self, audio_tensor: torch.Tensor, language: str
    ) -> ASRResult:
        """Internal helper for override_language path."""
        if language in settings.tier1_languages:
            return self.indic_asr.transcribe(audio_tensor, language=language)
        return self.whisper_asr.transcribe(audio_tensor, language=language)
