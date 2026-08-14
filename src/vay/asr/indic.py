"""IndicConformer ASR wrapper for Tamil & Hindi (ai4bharat/indic-conformer-600m-multilingual)."""

from typing import Any

import torch

from vay.asr.base import BaseASR
from vay.types import ASRResult, LanguageTier


class IndicConformerASR(BaseASR):
    """Tier 1 ASR model wrapper for Tamil and Hindi using IndicConformer.

    Note: Loaded using AutoModel.from_pretrained (transformers pipeline() is NOT supported).
    Does NOT support English.
    """

    def __init__(self, model_id: str = "ai4bharat/indic-conformer-600m-multilingual") -> None:
        self.model_id = model_id
        self.model: Any = None

    def transcribe(self, audio_tensor: torch.Tensor, language: str) -> ASRResult:
        """Transcribe Tamil ('ta') or Hindi ('hi') speech audio.

        Args:
            audio_tensor: 1D torch float32 audio tensor (16kHz mono).
            language: Language code ('ta' or 'hi').

        Returns:
            ASRResult containing raw transcribed text.
        """
        if language not in ("ta", "hi"):
            raise ValueError(
                f"IndicConformer only supports Tier 1 languages ('ta', 'hi'), got: '{language}'"
            )

        # Ensure wav_tensor has shape [1, num_samples]
        if audio_tensor.ndim == 1:
            audio_tensor.unsqueeze(0)

        # Skeleton transcript simulation
        simulated_text = (
            "என் பில் தொகையை சரிபார்க்க வேண்டும்"
            if language == "ta"
            else "मेरा बिल विवरण देखना है"
        )

        return ASRResult(
            raw_text=simulated_text,
            detected_language=language,
            language_tier=LanguageTier.TIER_1,
            confidence=0.95,
            model_used=self.model_id,
        )
