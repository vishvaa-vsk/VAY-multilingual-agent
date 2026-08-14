"""Application settings and configuration parameters."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings class."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Project paths
    project_root: Path = Path(__file__).parent.parent.parent
    data_dir: Path = project_root / "data"

    # Models configuration
    indic_asr_model: str = "ai4bharat/indic-conformer-600m-multilingual"
    whisper_asr_model: str = "openai/whisper-large-v3-turbo"

    # RAG settings
    retrieval_confidence_threshold: float = 0.80  # tau threshold (0.75 - 0.85)
    top_k_results: int = 5

    # VAD Settings
    sample_rate: int = 16000
    silence_duration_ms: int = 650

    # Language tiers
    tier1_languages: list[str] = ["ta", "hi"]  # Tamil, Hindi


settings = Settings()
