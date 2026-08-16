"""Whisper ASR wrapper for English and general fallback (openai/whisper-large-v3-turbo).

Key design: ``transcribe_auto()`` performs language detection AND transcription in a
SINGLE Groq API call (``verbose_json`` response includes the detected language code).
The router uses this to avoid the previous two-call pattern (detect → transcribe),
cutting Tier-2 latency roughly in half.
"""

from __future__ import annotations

import io
import math
import os
from typing import Any

import numpy as np
import soundfile as sf
import torch
from dotenv import load_dotenv
from groq import Groq

from vay.asr.base import BaseASR
from vay.config import settings
from vay.types import ASRResult, LanguageTier


# ---------------------------------------------------------------------------
# Full English language name → ISO 639-1 code
# Groq's Whisper endpoint returns the full English name in response.language;
# we map it back to the short ISO code before routing.
# ---------------------------------------------------------------------------
_LANGUAGE_NAME_TO_CODE: dict[str, str] = {
    # Indian languages
    "assamese": "as", "bengali": "bn", "bodo": "brx", "dogri": "doi",
    "gujarati": "gu", "hindi": "hi", "kannada": "kn", "konkani": "kok",
    "kashmiri": "ks", "maithili": "mai", "malayalam": "ml", "manipuri": "mni",
    "marathi": "mr", "nepali": "ne", "odia": "or", "oriya": "or",
    "punjabi": "pa", "panjabi": "pa", "sanskrit": "sa", "santali": "sat",
    "sindhi": "sd", "tamil": "ta", "telugu": "te", "urdu": "ur",
    # Major world languages
    "english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
    "arabic": "ar", "french": "fr", "german": "de", "spanish": "es",
    "portuguese": "pt", "russian": "ru", "italian": "it", "dutch": "nl",
    "polish": "pl", "turkish": "tr", "vietnamese": "vi", "thai": "th",
    "indonesian": "id", "malay": "ms", "swahili": "sw", "hausa": "ha",
    "yoruba": "yo", "afrikaans": "af", "albanian": "sq", "armenian": "hy",
    "azerbaijani": "az", "basque": "eu", "belarusian": "be", "bosnian": "bs",
    "breton": "br", "bulgarian": "bg", "catalan": "ca", "croatian": "hr",
    "czech": "cs", "danish": "da", "estonian": "et", "faroese": "fo",
    "finnish": "fi", "galician": "gl", "georgian": "ka", "greek": "el",
    "hebrew": "he", "hungarian": "hu", "icelandic": "is", "javanese": "jv",
    "kazakh": "kk", "khmer": "km", "latin": "la", "latvian": "lv",
    "lingala": "ln", "lithuanian": "lt", "luxembourgish": "lb",
    "macedonian": "mk", "malagasy": "mg", "maltese": "mt", "maori": "mi",
    "mongolian": "mn", "myanmar": "my", "burmese": "my", "norwegian": "no",
    "occitan": "oc", "pashto": "ps", "persian": "fa", "romanian": "ro",
    "serbian": "sr", "shona": "sn", "sinhalese": "si", "sinhala": "si",
    "slovak": "sk", "slovenian": "sl", "somali": "so", "sundanese": "su",
    "swedish": "sv", "tagalog": "tl", "tajik": "tg", "tatar": "tt",
    "tibetan": "bo", "turkmen": "tk", "ukrainian": "uk", "uzbek": "uz",
    "welsh": "cy", "yiddish": "yi", "yoruba": "yo", "cantonese": "yue",
    "haitian creole": "ht", "hawaiian": "haw", "lao": "lo",
    "amharic": "am", "bashkir": "ba", "brezhoneg": "br",
}

# Set of ISO codes the Groq Whisper endpoint actually accepts as a ``language``
# parameter.  Any code not in this set is treated as unrecognised and we fall
# back to auto-detection (no language hint) to avoid a 400 error.
_GROQ_VALID_LANGUAGE_CODES: frozenset[str] = frozenset(
    "gl mr sn lo my pt tr ms te bn ka am tt uk ml br pa lb es da af ht mt "
    "sa mg ba fa zh de ko pl la mi sk et ru hi hu az sl si yo tg vi el th "
    "lt kn oc sd bo en sv id cs ur ps tl haw fr nl he be as su ne ha jv hr "
    "sr hy tk ln yue ar it ro bg sq uz fo ja fi no is bs sw so yi mk mn kk "
    "km ca cy eu gu nn ta lv".split()
)


def _normalise_language(raw: str) -> str | None:
    """Map Groq's raw language string to a validated ISO 639-1 code.

    Returns *None* when the code is unrecognised so callers can fall back to
    auto-detection instead of passing an invalid value to the API.
    """
    if not raw:
        return None
    lower = raw.strip().lower()
    # Already a valid short code (e.g. "en", "hi", "ta")
    if lower in _GROQ_VALID_LANGUAGE_CODES:
        return lower
    # Full English name (e.g. "hindi", "thai")
    mapped = _LANGUAGE_NAME_TO_CODE.get(lower)
    if mapped and mapped in _GROQ_VALID_LANGUAGE_CODES:
        return mapped
    return None


class WhisperASR(BaseASR):
    """Tier-2 ASR wrapper using Groq's whisper-large-v3-turbo endpoint.

    Primary entry point: ``transcribe_auto()`` — performs language detection
    AND transcription in a single API call.  The router calls this instead of
    the old detect_language + transcribe two-call pattern.
    """

    def __init__(self, model_id: str = "whisper-large-v3-turbo") -> None:
        self.model_id = model_id
        load_dotenv()
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _tensor_to_wav_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        """Convert a float32 audio tensor to WAV bytes (16-bit PCM, 16 kHz)."""
        if audio_tensor.is_cuda:
            audio_np = audio_tensor.detach().cpu().numpy()
        else:
            audio_np = audio_tensor.detach().numpy()
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_np, settings.sample_rate, format="WAV", subtype="PCM_16")
        wav_io.seek(0)
        return wav_io.read()

    def _extract_confidence(self, segments: list[Any]) -> float:
        """Derive an overall confidence score from segment log-probabilities."""
        if not segments:
            return 0.5
        total_logprob = sum(
            seg.get("avg_logprob", -1.0) if isinstance(seg, dict)
            else getattr(seg, "avg_logprob", -1.0)
            for seg in segments
        )
        return float(math.exp(total_logprob / len(segments)))

    def _is_silence(self, segments: list[Any]) -> bool:
        """Return True when the first segment's no_speech_prob > 0.6."""
        if not segments:
            return False
        first = segments[0]
        no_speech = (
            first.get("no_speech_prob", 0.0) if isinstance(first, dict)
            else getattr(first, "no_speech_prob", 0.0)
        )
        return no_speech > 0.6

    def filter_hallucinations(self, text: str) -> str:
        """Remove consecutive duplicate words (Whisper repetition hallucination)."""
        words = text.split()
        if not words:
            return ""
        filtered: list[str] = []
        for word in words:
            if not filtered or filtered[-1].lower() != word.lower():
                filtered.append(word)
        return " ".join(filtered)

    # ------------------------------------------------------------------
    # PRIMARY: single-pass transcription + language detection
    # ------------------------------------------------------------------

    def transcribe_auto(self, audio_tensor: torch.Tensor) -> ASRResult:
        """Transcribe audio WITHOUT a language hint, letting Whisper auto-detect.

        This is the main entry point used by the router.  A single Groq API
        call with ``response_format="verbose_json"`` returns both the
        transcribed text AND the detected language code — eliminating the
        previous two-call pattern (detect_language → transcribe) that was
        responsible for most of the per-turn latency on Tier-2 paths.

        Returns:
            ASRResult with detected_language populated from Whisper's output.
        """
        wav_bytes = self._tensor_to_wav_bytes(audio_tensor)

        response = self.client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=self.model_id,
            # No ``language`` param → Whisper auto-detects from the audio
            response_format="verbose_json",
        )

        segments = getattr(response, "segments", []) or []

        # Silence / hallucination guard
        if self._is_silence(segments):
            return ASRResult(
                raw_text="",
                detected_language="en",
                language_tier=LanguageTier.TIER_2,
                confidence=0.0,
                model_used=self.model_id,
            )

        raw_lang = getattr(response, "language", "") or ""
        detected_lang = _normalise_language(raw_lang) or "en"

        confidence = self._extract_confidence(segments)
        # Reduce confidence when speech probability is low
        if segments:
            first = segments[0]
            no_speech = (
                first.get("no_speech_prob", 0.0) if isinstance(first, dict)
                else getattr(first, "no_speech_prob", 0.0)
            )
            if no_speech > 0.3:
                confidence *= 1.0 - no_speech

        cleaned_text = self.filter_hallucinations(getattr(response, "text", "") or "")

        return ASRResult(
            raw_text=cleaned_text,
            detected_language=detected_lang,
            language_tier=LanguageTier.TIER_2,
            confidence=confidence,
            model_used=self.model_id,
        )

    # ------------------------------------------------------------------
    # SECONDARY: forced-language transcription (called by router for
    # Tier-1 fallback path and backward-compat callers)
    # ------------------------------------------------------------------

    def transcribe(self, audio_tensor: torch.Tensor, language: str = "en") -> ASRResult:
        """Transcribe with an explicit language hint.

        Args:
            audio_tensor: 1-D float32 audio tensor (16 kHz mono).
            language: Validated ISO 639-1 code (e.g. ``"en"``, ``"ta"``).
                      Must be in ``_GROQ_VALID_LANGUAGE_CODES`` — call
                      ``_normalise_language()`` before passing if unsure.

        Returns:
            ASRResult containing the transcribed text.
        """
        # Safety: if the code is invalid, fall back to auto
        validated = _normalise_language(language)
        if validated is None:
            print(
                f"[WhisperASR] Unknown language code '{language}' — "
                "falling back to auto-detection."
            )
            return self.transcribe_auto(audio_tensor)

        wav_bytes = self._tensor_to_wav_bytes(audio_tensor)
        response = self.client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=self.model_id,
            language=validated,
            response_format="verbose_json",
        )

        segments = getattr(response, "segments", []) or []
        if self._is_silence(segments):
            return ASRResult(
                raw_text="",
                detected_language=validated,
                language_tier=LanguageTier.TIER_2,
                confidence=0.0,
                model_used=self.model_id,
            )

        cleaned_text = self.filter_hallucinations(getattr(response, "text", "") or "")
        return ASRResult(
            raw_text=cleaned_text,
            detected_language=validated,
            language_tier=LanguageTier.TIER_2,
            confidence=self._extract_confidence(segments),
            model_used=self.model_id,
        )

    # ------------------------------------------------------------------
    # LEGACY: kept for backward compatibility with any callers that still
    # call detect_language() directly (e.g. tests).
    # Now implemented as a thin wrapper over transcribe_auto().
    # ------------------------------------------------------------------

    def detect_language(self, audio_tensor: torch.Tensor) -> tuple[str, float]:
        """Detect language from audio.  Now a wrapper over ``transcribe_auto()``.

        Returns:
            Tuple of (iso_language_code, confidence_float).
        """
        result = self.transcribe_auto(audio_tensor)
        return result.detected_language, result.confidence
