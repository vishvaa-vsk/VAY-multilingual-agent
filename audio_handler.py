import base64

def decode_audio(base64_str: str) -> bytes:
    """
    Decodes a base64 string sent from the browser into raw WAV bytes.
    """
    return base64.b64decode(base64_str)

def encode_audio_bytes(audio_bytes: bytes) -> str:
    """
    Encodes raw WAV bytes into a base64 string to send to the browser.
    """
    return base64.b64encode(audio_bytes).decode("utf-8")

def encode_audio_file(file_path: str) -> str:
    """
    Reads a WAV file from disk and encodes it into a base64 string.
    """
    with open(file_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")
