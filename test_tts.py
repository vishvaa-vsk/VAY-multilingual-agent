import sys
import os
from vay.tts.engine import speak
speak("Hello world", lang="en", play=False, output_path="out.mp3")
print("Size:", os.path.getsize("out.mp3"))
