"""Two-tier language routing mechanism for ASR execution."""

import numpy as np
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
        
        # State for continuous language ID
        self.locked_language: str | None = None
        self.utterances_count = 0
        self.audio_accumulator: list[np.ndarray] = []

    def reset_session(self) -> None:
        """Reset the language locking state for a new session."""
        self.locked_language = None
        self.utterances_count = 0
        self.audio_accumulator.clear()

    def lock_language(self, language: str) -> None:
        """Manually force lock the language."""
        self.locked_language = language

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
        lang = override_language or self.locked_language
        
        # If we haven't locked a language yet, we need to accumulate and detect
        if not lang:
            self.utterances_count += 1
            
            # Add to accumulator
            if audio_tensor.is_cuda:
                self.audio_accumulator.append(audio_tensor.detach().cpu().numpy())
            else:
                self.audio_accumulator.append(audio_tensor.detach().numpy())
                
            # Combine accumulated audio
            combined_audio_np = np.concatenate(self.audio_accumulator)
            combined_tensor = torch.from_numpy(combined_audio_np)
            
            # Detect language using Whisper on the accumulated audio
            detected_lang, confidence = self.whisper_asr.detect_language(combined_tensor)
            
            # Check if we should lock it
            if self.utterances_count >= settings.min_utterances_for_lock and confidence >= settings.language_confidence_threshold:
                print(f"[Router] Locking language to '{detected_lang}' (confidence: {confidence:.2f})")
                self.locked_language = detected_lang
            else:
                print(f"[Router] Detected language '{detected_lang}' (confidence: {confidence:.2f}). Not locked yet (utterance {self.utterances_count}/{settings.min_utterances_for_lock}).")
                
            lang = detected_lang

        # Routing Logic based on the selected language
        if lang in settings.tier1_languages:
            return self.indic_asr.transcribe(audio_tensor, language=lang)
        else:
            return self.whisper_asr.transcribe(audio_tensor, language=lang)
