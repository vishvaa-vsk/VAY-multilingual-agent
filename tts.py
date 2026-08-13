import asyncio
import edge_tts
from playsound3 import playsound
import os

# Default voice for each language
voices = {
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
    "ur": "ur-IN-GulNeural"
}

async def generate_speech(text, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save("output.mp3")

text = input("Enter text: ")
lang = input("Enter language code: ").lower()

if lang not in voices:
    print("Language not configured.")
    print("Available:", ", ".join(voices.keys()))
else:
    voice = voices[lang]

    print("Using voice:", voice)

    asyncio.run(generate_speech(text, voice))

    playsound("output.mp3")

    os.remove("output.mp3")