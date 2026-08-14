"""Text-to-Speech (TTS) dispatcher for Indic (Tamil/Hindi) and English fallback."""

from pathlib import Path


class TTSEngine:
    """Language-aware TTS synthesis engine."""

    def synthesize(self, text: str, language: str, output_path: str | Path) -> Path:
        """Synthesize text into speech file.

        Args:
            text: Grounded response text to speak.
            language: ISO language code ('ta', 'hi', 'en').
            output_path: Path to write synthesized audio file.

        Returns:
            Path object pointing to synthesized audio.
        """
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Mock synthesis output
        if not out_p.exists():
            out_p.touch()
        return out_p
