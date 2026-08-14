"""
tts.py

Language-aware TTS for the Nexatel voice assistant, via edge-tts (Microsoft
Edge neural voices). Exposes an importable speak() function — agent_graph.py
calls this at the end of every turn to speak the final grounded reply.

CLI (manual testing only):
    python tts.py
"""

import asyncio
import os

import edge_tts

# Default voice for each language
VOICES = {
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


async def _generate_speech(text: str, voice: str, output_path: str) -> None:
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)


def speak(text: str, lang: str = "en", output_path: str = "output.mp3", play: bool = True) -> str:
    """Synthesize `text` in the voice for `lang` (falls back to English if the
    language isn't configured), optionally play it, then clean up the file.

    Never raises — playback/synthesis failures are logged and swallowed so a
    headless environment or missing audio device doesn't kill the call loop.
    Returns the output_path used (even if playback failed).
    """
    voice = VOICES.get(lang, VOICES["en"])
    try:
        asyncio.run(_generate_speech(text, voice, output_path))
    except Exception as e:
        print(f"  [TTS synthesis error: {e}]")
        return output_path

    if play:
        try:
            from playsound3 import playsound
            playsound(output_path)
        except Exception as e:
            print(f"  [TTS playback error (continuing without audio): {e}]")
        finally:
            try:
                if os.path.exists(output_path):
                    os.remove(output_path)
            except OSError:
                pass

    return output_path


if __name__ == "__main__":
    text = input("Enter text: ")
    lang = input("Enter language code: ").lower()

    if lang not in VOICES:
        print("Language not configured.")
        print("Available:", ", ".join(VOICES.keys()))
    else:
        print("Using voice:", VOICES[lang])
        speak(text, lang=lang)
