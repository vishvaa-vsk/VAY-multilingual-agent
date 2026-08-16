"""Abstract base class interface for ASR engines."""

from abc import ABC, abstractmethod

import torch

from vay.types import ASRResult


class BaseASR(ABC):
    """Abstract Base Class for ASR models."""

    @abstractmethod
    def transcribe(self, audio_tensor: torch.Tensor, language: str) -> ASRResult:
        """Transcribe audio tensor for a specified language.

        Args:
            audio_tensor: 1D torch float32 audio tensor (16kHz mono).
            language: ISO language code (e.g. 'ta', 'hi', 'en').

        Returns:
            ASRResult object containing transcribed raw text.
        """
        pass
