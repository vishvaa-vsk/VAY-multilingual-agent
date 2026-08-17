"""Tests for the TTS sentence-chunking used to pipeline synthesis+playback."""

from vay.tts.engine import _split_into_speech_chunks


def test_short_text_is_not_split() -> None:
    text = "Your balance is Rs 299."
    assert _split_into_speech_chunks(text) == [text]


def test_single_long_sentence_without_boundaries_is_not_split() -> None:
    # No sentence-ending punctuation at all, even though it's long.
    text = "a" * 200
    assert _split_into_speech_chunks(text) == [text]


def test_multi_sentence_english_text_is_split() -> None:
    text = (
        "Your current plan is Prepaid Value at Rs 299 per month. "
        "It includes 2 GB of data per day and unlimited voice calls. "
        "Your plan is valid for 28 days from the activation date."
    )
    chunks = _split_into_speech_chunks(text)
    assert len(chunks) == 3
    assert chunks[0].startswith("Your current plan")
    assert chunks[1].startswith("It includes")
    assert chunks[2].startswith("Your plan is valid")


def test_hindi_danda_boundary_is_split() -> None:
    text = (
        "आपका मौजूदा प्लान 299 रुपये प्रति माह का है। "
        "इसमें रोज़ाना 2 जीबी डेटा और अनलिमिटेड कॉल शामिल हैं। "
        "यह प्लान सक्रियण तिथि से 28 दिनों के लिए वैध है।"
    )
    chunks = _split_into_speech_chunks(text)
    assert len(chunks) == 3


def test_no_empty_chunks_produced() -> None:
    text = (
        "This is sentence one. "
        "This is sentence two!   "
        "Is this sentence three?"
    )
    chunks = _split_into_speech_chunks(text)
    assert all(c.strip() for c in chunks)
