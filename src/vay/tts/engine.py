"""
tts/engine.py

Language-aware TTS for the VAY / Nexatel voice assistant, via edge-tts
(Microsoft Edge Neural voices). Exposes a speak() function and a TTSEngine
class — graph/nodes/utils.py calls tts.speak() at the end of every turn.

Supported languages (ISO 639-1 → Microsoft Neural Voice):
    ta  hi  en  fr  de  es  ja  ko  zh  it  ru  ar
    te  kn  ml  mr  gu  ur
Any language not in this map falls back to en-IN-NeerjaNeural.

CLI (manual testing only):
    python -m vay.tts.engine
"""

from __future__ import annotations

import asyncio
import os
import re
import tempfile
from pathlib import Path

try:
    import edge_tts
except ImportError:
    edge_tts = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Default neural voice per language (ISO 639-1 code)
# ---------------------------------------------------------------------------
VOICES: dict[str, str] = {
    "ta": "ta-IN-PallaviNeural",
    "hi": "hi-IN-SwaraNeural",
    "en": "en-IN-NeerjaNeural",
    "fr": "fr-FR-DeniseNeural",
    "de": "de-DE-KatjaNeural",
    "es": "es-ES-ElviraNeural",
    "ja": "ja-JP-NanamiNeural",
    "ko": "ko-KR-SunHiNeural",
    "zh": "zh-CN-XiaoxiaoNeural",
    "it": "it-IT-ElsaNeural",
    "ru": "ru-RU-SvetlanaNeural",
    "ar": "ar-AE-FatimaNeural",
    "te": "te-IN-ShrutiNeural",
    "kn": "kn-IN-SapnaNeural",
    "ml": "ml-IN-SobhanaNeural",
    "mr": "mr-IN-AarohiNeural",
    "gu": "gu-IN-DhwaniNeural",
    "ur": "ur-IN-GulNeural",
}

FALLBACK_VOICE = VOICES["en"]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_text_for_speech(text: str) -> str:
    """Strip markdown formatting for clean voice synthesis."""
    cleaned = re.sub(r"\*\*|\*|#+", "", text)
    cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n+", " ", cleaned)
    return cleaned.strip()


# Sentence-ending punctuation across every language this TTS engine supports:
# Latin '.', '!', '?'; Devanagari danda '।' / double danda '॥' (Hindi, Marathi);
# fullwidth CJK punctuation used by the zh/ja/ko voices.
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?।॥。！？])\s+")

# Chunking only pays off once there's more than one sentence to pipeline —
# below this length, splitting just adds an extra edge-tts connection for
# no latency benefit.
_MIN_CHARS_FOR_CHUNKING = 120


def _split_into_speech_chunks(text: str) -> list[str]:
    """Split *text* into sentence-level chunks for pipelined synthesis.

    Returns a single-item list (the whole text) when the text is short or
    has no detectable sentence boundaries — chunking a one-liner only adds
    network overhead with no latency win.
    """
    if len(text) < _MIN_CHARS_FOR_CHUNKING:
        return [text]

    parts = [p.strip() for p in _SENTENCE_SPLIT_RE.split(text) if p.strip()]
    if len(parts) <= 1:
        return [text]
    return parts


async def _generate_speech(text: str, voice: str, output_path: str) -> None:
    """Async helper: synthesize text with edge-tts and save to output_path."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


async def _synthesize_chunk(text: str, voice: str) -> str:
    """Synthesize one chunk to a fresh temp mp3 file, return its path."""
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
    os.close(tmp_fd)
    await _generate_speech(text, voice, tmp_path)
    return tmp_path


def _play_file(path: str) -> None:
    """Blocking playback of a single audio file. Logs and swallows errors."""
    try:
        from playsound3 import playsound  # type: ignore[import]
        playsound(path)
    except Exception as e:
        print(f"  [TTS playback error (continuing without audio): {e}]")
    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass


def _run_async(coro) -> None:
    """Run *coro* to completion, handling the case where we're already
    inside a running event loop (e.g. Gradio / Streamlit / Jupyter) by
    falling back to a dedicated thread. Errors are logged, not raised.
    """
    try:
        asyncio.run(coro)
    except RuntimeError:
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, coro)
            try:
                future.result(timeout=60)
            except Exception as e:
                print(f"  [TTS error (thread): {e}]")
    except Exception as e:
        print(f"  [TTS error: {e}]")


async def _speak_pipelined(chunks: list[str], voice: str) -> None:
    """Synthesize and play *chunks* in order, overlapping synthesis of the
    next chunk with playback of the current one.

    Time-to-first-audio is bounded by the synthesis time of chunk 0 only —
    every subsequent chunk is already being synthesized in the background
    while the previous one plays, instead of the whole reply having to
    finish synthesizing before any audio starts.
    """
    loop = asyncio.get_running_loop()

    next_chunk_task = asyncio.ensure_future(_synthesize_chunk(chunks[0], voice))
    for i in range(len(chunks)):
        current_path = await next_chunk_task

        # Kick off synthesis of the following chunk (if any) before we start
        # blocking on playback of the current one, so it runs concurrently.
        if i + 1 < len(chunks):
            next_chunk_task = asyncio.ensure_future(
                _synthesize_chunk(chunks[i + 1], voice)
            )

        await loop.run_in_executor(None, _play_file, current_path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def speak(
    text: str,
    lang: str = "en",
    language: str | None = None,
    output_path: str | None = None,
    play: bool = True,
    **kwargs,
) -> str:
    """Synthesize *text* in the neural voice for *lang* and optionally play it.

    Args:
        text:        The spoken-language reply to synthesize.
        lang:        ISO 639-1 language code (e.g. ``"ta"``, ``"hi"``, ``"en"``).
        language:    Alias for *lang* — whichever is non-empty wins (lang takes
                     priority if both are set).
        output_path: Where to write the MP3.  If ``None`` a temp file is used
                     and deleted after playback.
        play:        Whether to play the audio immediately after synthesis.

    Returns:
        The path to the MP3 file (may already be deleted if *play* was True
        and cleanup succeeded).

    Never raises — synthesis/playback failures are logged and swallowed so a
    headless environment or missing audio device doesn't kill the call loop.
    """
    effective_lang = (lang or language or "en").lower().strip()
    if not text:
        return output_path or ""

    # Script-aware safeguard: If the LLM returned Tamil/Hindi/Indic Unicode text,
    # ensure we pick the correct native neural voice rather than an English voice
    # which would otherwise pronounce the Unicode characters as numbers/gibberish.
    if re.search(r"[\u0b80-\u0bff]", text):
        effective_lang = "ta"
    elif re.search(r"[\u0900-\u097f]", text):
        effective_lang = "hi"
    elif re.search(r"[\u0c00-\u0c7f]", text):
        effective_lang = "te"
    elif re.search(r"[\u0c80-\u0cff]", text):
        effective_lang = "kn"
    elif re.search(r"[\u0d00-\u0d7f]", text):
        effective_lang = "ml"

    print(f"\n[TTS Audio Output ({effective_lang})]: {text}\n")

    if edge_tts is None:
        print("  [TTS Warning: edge-tts package is not installed. Run: uv add edge-tts]")
        return output_path or ""

    voice = VOICES.get(effective_lang, FALLBACK_VOICE)
    speech_text = _clean_text_for_speech(text)

    # --- Pipelined path: synthesize-and-play-immediately, no caller-supplied
    # output_path to preserve. This is the hot path used by tts_node on every
    # turn. Sentence chunks are synthesized and played in an overlapping
    # pipeline so playback of chunk 1 starts as soon as it's ready instead of
    # waiting for the whole reply to finish synthesizing first.
    if play and output_path is None:
        chunks = _split_into_speech_chunks(speech_text)
        _run_async(_speak_pipelined(chunks, voice))
        return ""

    # --- Single-file path: explicit output_path given, and/or play=False
    # (e.g. TTSEngine.synthesize(), tests, or any caller that wants one
    # complete file on disk). Behavior unchanged from before.
    using_temp = output_path is None
    if using_temp:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(tmp_fd)
        output_path = tmp_path

    try:
        asyncio.run(_generate_speech(speech_text, voice, output_path))
    except RuntimeError:
        # Already inside a running event loop (e.g. Gradio / Jupyter)
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, _generate_speech(speech_text, voice, output_path))
            try:
                future.result(timeout=30)
            except Exception as e:
                print(f"  [TTS synthesis error (thread): {e}]")
                return output_path
    except Exception as e:
        print(f"  [TTS synthesis error: {e}]")
        return output_path

    # --- Playback ---
    if play:
        try:
            from playsound3 import playsound  # type: ignore[import]
            playsound(output_path)
        except Exception as e:
            print(f"  [TTS playback error (continuing without audio): {e}]")
        finally:
            if using_temp:
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                except OSError:
                    pass

    return output_path


# ---------------------------------------------------------------------------
# TTSEngine class (used by graph/nodes/utils.py and tests)
# ---------------------------------------------------------------------------

class TTSEngine:
    """Language-aware TTS synthesis engine wrapping edge-tts."""

    def synthesize(self, text: str, language: str, output_path: str | Path) -> Path:
        """Synthesize *text* in *language* and write to *output_path*.

        Returns the output path (creates parent dirs as needed).
        Does not play audio — use :py:meth:`speak` for that.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        speak(text, lang=language, output_path=str(out_p), play=False)
        return out_p

    def speak(self, text: str, lang: str = "en", **kwargs) -> None:
        """Synthesize and play *text* in *lang*."""
        speak(text, lang=lang, **kwargs)


# ---------------------------------------------------------------------------
# CLI — manual testing only
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    text_in = input("Enter text: ")
    lang_in = input("Enter language code: ").lower().strip()

    if lang_in not in VOICES:
        print(f"Language '{lang_in}' not configured. Using English fallback.")
        print("Available languages:", ", ".join(sorted(VOICES.keys())))
        lang_in = "en"

    print(f"Using voice: {VOICES[lang_in]}")
    speak(text_in, lang=lang_in)
