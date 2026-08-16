"""IndicConformer ASR wrapper for Tamil & Hindi (ai4bharat/indic-conformer-600m-multilingual)."""

from typing import Any

import torch

from vay.asr.base import BaseASR
from vay.config import settings
from vay.types import ASRResult, LanguageTier


class IndicConformerASR(BaseASR):
    """Tier 1 ASR model wrapper for Tamil and Hindi using IndicConformer.

    Note: Loaded using AutoModel.from_pretrained (transformers pipeline() is NOT supported).
    Does NOT support English.
    """

    def __init__(self, model_id: str = "ai4bharat/indic-conformer-600m-multilingual") -> None:
        self.model_id = model_id
        print(f"[IndicConformer] Loading model {model_id}. Ensure you are authenticated with HuggingFace (HF_TOKEN) as this is a gated model.")
        try:
            import os
            from dotenv import load_dotenv
            from transformers import AutoModel
            
            # Ensure environment variables are loaded
            load_dotenv(override=True)
            hf_token = os.environ.get("HF_TOKEN")
            
            self.model = AutoModel.from_pretrained(
                self.model_id, 
                trust_remote_code=True,
                token=hf_token
            )
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.to('cuda')
        except Exception as e:
            print(f"[IndicConformer] Failed to load model: {e}")
            self.model = None

    def transcribe(self, audio_tensor: torch.Tensor, language: str) -> ASRResult:
        """Transcribe speech audio using the actual IndicConformer model."""
        if language not in settings.tier1_languages:
            raise ValueError(
                f"IndicConformer only supports Tier 1 languages, got: '{language}'"
            )

        if self.model is None:
            return ASRResult(
                raw_text="[IndicConformer Model Not Loaded - Check HF_TOKEN]",
                detected_language=language,
                language_tier=LanguageTier.TIER_1,
                confidence=0.0,
                model_used=self.model_id,
            )

        # Ensure audio_tensor has shape [1, num_samples]
        # VAD might return [num_samples, 1], so we flatten and add batch dim
        audio_tensor = audio_tensor.view(1, -1)
            
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.to('cuda')

        try:
            with torch.no_grad():
                # Inference call per AI4Bharat documentation (via project_context.md)
                output = self.model(audio_tensor, language, "rnnt")
                
            # Parse output flexibly (could be dict, list, or string depending on version)
            if isinstance(output, dict):
                transcribed_text = output.get("text", str(output))
            elif isinstance(output, list) and len(output) > 0:
                if isinstance(output[0], dict):
                    transcribed_text = output[0].get("text", str(output[0]))
                elif isinstance(output[0], str):
                    transcribed_text = output[0]
                else:
                    transcribed_text = str(output[0])
            else:
                transcribed_text = str(output)
                
        except Exception as e:
            print(f"[IndicConformer] Inference error: {e}")
            transcribed_text = f"[Inference Error: {e}]"

        return ASRResult(
            raw_text=transcribed_text,
            detected_language=language,
            language_tier=LanguageTier.TIER_1,
            confidence=0.95,  # IndicConformer doesn't reliably return word confidences in the base call
            model_used=self.model_id,
        )
