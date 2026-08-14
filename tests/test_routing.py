"""Tests for ASR language routing mechanism."""

import torch

from vay.asr.router import ASRRouter
from vay.types import LanguageTier


def test_routing_tamil_to_indic_conformer() -> None:
    router = ASRRouter()
    dummy_audio = torch.zeros(16000, dtype=torch.float32)
    res = router.route_and_transcribe(dummy_audio, override_language="ta")
    assert res.language_tier == LanguageTier.TIER_1
    assert res.detected_language == "ta"


def test_routing_english_to_whisper() -> None:
    router = ASRRouter()
    dummy_audio = torch.zeros(16000, dtype=torch.float32)
    res = router.route_and_transcribe(dummy_audio, override_language="en")
    assert res.language_tier == LanguageTier.TIER_2
    assert res.detected_language == "en"
