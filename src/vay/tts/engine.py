"""Text-to-Speech (TTS) dispatcher for Indic (Tamil/Hindi) and English voice synthesis."""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from gtts import gTTS
except ImportError:
    gTTS = None


def _clean_text_for_speech(text: str) -> str:
    """Strip markdown formatting (asterisks, bullet points, headers) for clean voice synthesis."""
    cleaned = re.sub(r"\*\*|\*|#+", "", text)
    cleaned = re.sub(r"^\s*[-•]\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n+", " ", cleaned)
    return cleaned.strip()


def speak(text: str, lang: str = "en", language: str | None = None, **kwargs) -> None:
    """Language-aware TTS synthesis and speaker audio playback."""
    l = (language or lang or "en").lower().strip()
    if not text:
        return

    print(f"\n[TTS Audio Output ({l})]: {text}\n")

    if not gTTS:
        print("  [TTS Warning: gtts package is not installed for audio synthesis.]")
        return

    # Map language code to gTTS supported code
    supported_langs = {"ta": "ta", "hi": "hi", "en": "en", "te": "te", "bn": "bn", "mr": "mr"}
    speech_lang = supported_langs.get(l, "en")

    speech_text = _clean_text_for_speech(text)

    # For very long responses, summarize first 350 chars for prompt speech playback
    if len(speech_text) > 400:
        sentences = speech_text.split(".")
        short_speech = ". ".join(sentences[:2]) + "."
        if len(short_speech) < 50:
            short_speech = speech_text[:350]
        speech_text = short_speech

    try:
        tts = gTTS(text=speech_text, lang=speech_lang, slow=False)
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            tmp_path = tmp.name

        tts.save(tmp_path)

        # Audio playback per OS
        if sys.platform == "win32":
            ps_script = f"""
            Add-Type -AssemblyName PresentationCore
            $player = New-Object System.Windows.Media.MediaPlayer
            $player.Open('{tmp_path}')
            $player.Play()
            $duration = 0
            while ($player.NaturalDuration.HasTimeSpan -eq $false -and $duration -lt 30) {{
                Start-Sleep -Milliseconds 100
                $duration += 1
            }}
            if ($player.NaturalDuration.HasTimeSpan) {{
                $sleepTime = [math]::Ceiling($player.NaturalDuration.TimeSpan.TotalSeconds)
                Start-Sleep -Seconds $sleepTime
            }} else {{
                Start-Sleep -Seconds 5
            }}
            $player.Close()
            """
            subprocess.run(
                ["powershell", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        elif sys.platform == "darwin":
            subprocess.run(["afplay", tmp_path], check=False)
        else:
            subprocess.run(["aplay", tmp_path], check=False)

        try:
            os.remove(tmp_path)
        except Exception:
            pass
    except Exception as e:
        print(f"  [TTS Audio Playback Notice: {e}]")


class TTSEngine:
    """Language-aware TTS synthesis engine."""

    def synthesize(self, text: str, language: str, output_path: str | Path) -> Path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        l = (language or "en").lower().strip()
        speech_text = _clean_text_for_speech(text)
        if gTTS:
            try:
                tts = gTTS(text=speech_text, lang=l, slow=False)
                tts.save(str(out_p))
                return out_p
            except Exception:
                pass
        if not out_p.exists():
            out_p.touch()
        return out_p

    def speak(self, text: str, lang: str = "en", **kwargs) -> None:
        speak(text, lang=lang, **kwargs)
