"""Two-tier language routing mechanism for ASR execution."""

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

    def detect_language(self, audio_tensor: torch.Tensor) -> str:
        """Language ID pass using Whisper encoder pass.

        Returns ISO language code (e.g. 'ta', 'hi', 'en').
        """
        # Skeleton LID pass defaults to English if unassigned
        return "ta"

    def route_and_transcribe(
        self, audio_tensor: torch.Tensor, override_language: str | None = None
    ) -> ASRResult:
        """Route audio sample to appropriate ASR model.

        Args:
            audio_tensor: 1D float32 audio sample (16kHz mono).
            override_language: Optional language code override.

        Returns:
            ASRResult object from executed ASR model.
        """
        lang = override_language or self.detect_language(audio_tensor)

        if lang in settings.tier1_languages:
            return self.indic_asr.transcribe(audio_tensor, language=lang)
        else:
            return self.whisper_asr.transcribe(audio_tensor, language=lang)
