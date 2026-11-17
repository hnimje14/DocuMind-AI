import os
from pathlib import Path
from typing import Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    """Application settings loaded from environment variables or .env file."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    APP_NAME: str = "Intelligent Document Q&A System"
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # Server
    BACKEND_HOST: str = "0.0.0.0"
    BACKEND_PORT: int = 8000
    BACKEND_URL: str = "http://127.0.0.1:8000"

    # LLM Provider Configuration
    # Supported values: "mock", "openai", "groq", "openrouter", "gemini"
    LLM_PROVIDER: str = Field(default="mock")
    LLM_API_KEY: Optional[str] = Field(default=None)
    LLM_BASE_URL: Optional[str] = Field(default=None)
    LLM_MODEL: str = Field(default="gpt-4o-mini")
    LLM_TEMPERATURE: float = Field(default=0.1, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(default=1024, ge=64, le=8192)

    # Embedding Model Configuration
    EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBEDDING_DEVICE: str = "cpu"
    NORMALIZE_EMBEDDINGS: bool = True
    USE_HASH_EMBEDDINGS_FALLBACK: bool = False

    # Ingestion & Chunking Defaults
    MAX_FILE_SIZE_MB: int = 25
    DEFAULT_CHUNK_SIZE: int = 350
    DEFAULT_CHUNK_OVERLAP: int = 50

    # Retrieval Defaults
    DEFAULT_TOP_K: int = 5
    DEFAULT_SIMILARITY_THRESHOLD: float = 0.05

    # Storage Paths
    DATA_DIR: str = str(PROJECT_ROOT / "data")
    UPLOAD_DIR: str = str(PROJECT_ROOT / "data" / "uploads")
    VECTORSTORE_DIR: str = str(PROJECT_ROOT / "data" / "vectorstore")
    SQLITE_DB_PATH: str = str(PROJECT_ROOT / "data" / "rag_metadata.db")
    EVALUATION_DIR: str = str(PROJECT_ROOT / "evaluation")

    # Security & Rate Limiting
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 120

    def ensure_directories(self) -> None:
        """Ensure all required data and evaluation directories exist."""
        for dir_path in [
            self.DATA_DIR,
            self.UPLOAD_DIR,
            self.VECTORSTORE_DIR,
            self.EVALUATION_DIR,
            os.path.join(self.EVALUATION_DIR, "datasets"),
            os.path.join(self.EVALUATION_DIR, "results"),
        ]:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


settings = Settings()
settings.ensure_directories()
