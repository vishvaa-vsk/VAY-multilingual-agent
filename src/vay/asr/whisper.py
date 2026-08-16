"""Whisper ASR wrapper for English and general fallback (openai/whisper-large-v3-turbo)."""

import io
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


class WhisperASR(BaseASR):
    """Tier 2 ASR model wrapper for English and general language fallback using Groq."""

    LANGUAGE_MAP = {
        "assamese": "as", "bengali": "bn", "bodo": "brx", "dogri": "doi",
        "gujarati": "gu", "hindi": "hi", "kannada": "kn", "konkani": "kok",
        "kashmiri": "ks", "maithili": "mai", "malayalam": "ml", "manipuri": "mni",
        "marathi": "mr", "nepali": "ne", "odia": "or", "punjabi": "pa",
        "sanskrit": "sa", "santali": "sat", "sindhi": "sd", "tamil": "ta",
        "telugu": "te", "urdu": "ur", "english": "en"
    }

    def __init__(self, model_id: str = "whisper-large-v3-turbo") -> None:
        self.model_id = model_id
        load_dotenv()
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

    def _tensor_to_wav_bytes(self, audio_tensor: torch.Tensor) -> bytes:
        """Convert float32 audio tensor to WAV bytes."""
        # Convert to numpy
        if audio_tensor.is_cuda:
            audio_np = audio_tensor.detach().cpu().numpy()
        else:
            audio_np = audio_tensor.detach().numpy()
            
        wav_io = io.BytesIO()
        sf.write(wav_io, audio_np, settings.sample_rate, format='WAV', subtype='PCM_16')
        wav_io.seek(0)
        return wav_io.read()

    def filter_hallucinations(self, text: str) -> str:
        """Apply hallucination & repetition filtering on Whisper path outputs."""
        # Simple deduplication filter for repeating phrases
        words = text.split()
        if not words:
            return ""
        filtered: list[str] = []
        for word in words:
            if not filtered or filtered[-1].lower() != word.lower():
                filtered.append(word)
        return " ".join(filtered)

    def detect_language(self, audio_tensor: torch.Tensor) -> tuple[str, float]:
        """Run Whisper to detect language from audio chunk.
        
        Returns:
            Tuple of (iso_language_code, probability).
        """
        wav_bytes = self._tensor_to_wav_bytes(audio_tensor)
        
        # We don't necessarily need the transcription here, but Whisper detects language during transcription
        response = self.client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=self.model_id,
            response_format="verbose_json",
        )
        
        lang = getattr(response, "language", "unknown").lower()
        lang = self.LANGUAGE_MAP.get(lang, lang)
        
        prob = 0.0
        segments = getattr(response, "segments", [])
        if segments:
            import math
            
            total_logprob = 0.0
            for seg in segments:
                # Handle both dict and object access depending on Groq client version
                seg_logprob = seg.get("avg_logprob", -1.0) if isinstance(seg, dict) else getattr(seg, "avg_logprob", -1.0)
                total_logprob += seg_logprob
                
            avg_logprob = total_logprob / len(segments)
            # Convert the log probability (e.g. -0.09) to a standard probability between 0 and 1
            prob = math.exp(avg_logprob)
            
            # If the audio contains mostly silence, reduce the confidence
            first_seg = segments[0]
            no_speech = first_seg.get("no_speech_prob", 0.0) if isinstance(first_seg, dict) else getattr(first_seg, "no_speech_prob", 0.0)
            if no_speech > 0.5:
                prob = prob * (1.0 - no_speech)
                
        return lang, float(prob)

    def transcribe(self, audio_tensor: torch.Tensor, language: str = "en") -> ASRResult:
        """Transcribe audio using Whisper.

        Args:
            audio_tensor: 1D torch float32 audio tensor (16kHz mono).
            language: Detected ISO language code (e.g. 'en', 'es', 'fr').

        Returns:
            ASRResult containing raw transcribed text.
        """
        wav_bytes = self._tensor_to_wav_bytes(audio_tensor)
        
        response = self.client.audio.transcriptions.create(
            file=("audio.wav", wav_bytes),
            model=self.model_id,
            language=language,
            response_format="verbose_json",
        )
        
        # Whisper tends to hallucinate "Thank you" on silence. We use no_speech_prob to filter it.
        segments = getattr(response, "segments", [])
        if segments:
            first_seg = segments[0]
            no_speech = first_seg.get("no_speech_prob", 0.0) if isinstance(first_seg, dict) else getattr(first_seg, "no_speech_prob", 0.0)
            if no_speech > 0.6:
                # Highly likely to be silence/noise hallucination
                return ASRResult(
                    raw_text="",
                    detected_language=language,
                    language_tier=LanguageTier.TIER_2,
                    confidence=0.0,
                    model_used=self.model_id,
                )
                
        cleaned_text = self.filter_hallucinations(getattr(response, "text", ""))

        return ASRResult(
            raw_text=cleaned_text,
            detected_language=language,
            language_tier=LanguageTier.TIER_2,
            confidence=0.92,  # Mocked confidence, Whisper doesn't always return word-level conf nicely
            model_used=self.model_id,
        )
