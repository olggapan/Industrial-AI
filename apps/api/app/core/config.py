from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


API_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = API_ROOT / ".env"


class Settings(BaseSettings):
    app_name: str = "Industrial AI Troubleshooting Copilot API"
    app_version: str = "0.1.0"
    database_url: str = (
        "postgresql+psycopg://industrial_ai:industrial_ai@localhost:5432/industrial_ai"
    )
    cors_origins: list[str] = ["*"]

    rag_embedding_provider: str = "fake"
    local_embedding_model: str = "intfloat/multilingual-e5-small"
    local_embedding_batch_size: int = 32

    model_config = SettingsConfigDict(env_file=ENV_FILE, extra="ignore")


settings = Settings()