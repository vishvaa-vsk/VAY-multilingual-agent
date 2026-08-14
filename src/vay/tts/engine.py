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


async def _generate_speech(text: str, voice: str, output_path: str) -> None:
    """Async helper: synthesize text with edge-tts and save to output_path."""
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


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

    print(f"\n[TTS Audio Output ({effective_lang})]: {text}\n")

    if edge_tts is None:
        print("  [TTS Warning: edge-tts package is not installed. Run: uv add edge-tts]")
        return output_path or ""

    voice = VOICES.get(effective_lang, FALLBACK_VOICE)
    speech_text = _clean_text_for_speech(text)

    # Use a temp file when no explicit path is given
    using_temp = output_path is None
    if using_temp:
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp3")
        os.close(tmp_fd)
        output_path = tmp_path

    # --- Synthesis ---
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
