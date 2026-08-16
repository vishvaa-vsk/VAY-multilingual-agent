"""Audio loading, resampling, and normalization utilities."""

from pathlib import Path

import torch


def load_audio(file_path: Path | str, target_sr: int = 16000) -> torch.Tensor:
    """Load audio file and convert to 16kHz mono float32 torch tensor.

    Args:
        file_path: Path to input audio file.
        target_sr: Target sample rate in Hz (default 16000Hz).

    Returns:
        1D float32 tensor of shape [num_samples].
    """
    # Skeleton implementation returning a zero 1-second 16kHz audio buffer
    return torch.zeros(target_sr, dtype=torch.float32)


def normalize_audio(audio_tensor: torch.Tensor) -> torch.Tensor:
    """Normalize audio amplitude to range [-1.0, 1.0]."""
    max_val = torch.max(torch.abs(audio_tensor))
    if max_val > 0:
        return audio_tensor / max_val
    return audio_tensor
