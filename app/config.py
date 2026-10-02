"""Small, dependency-free application settings."""
from dataclasses import dataclass
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    root_dir: Path
    seed_dir: Path
    upload_dir: Path
    max_retrieval_loops: int
    cors_origins: list[str]
    ollama_enabled: bool
    ollama_base_url: str
    ollama_model: str
    ollama_timeout_seconds: int


def get_settings() -> Settings:
    root = Path(__file__).resolve().parent.parent
    configured_upload = os.getenv("ACADEMIA_UPLOAD_DIR", "data/uploads")
    upload_dir = Path(configured_upload)
    if not upload_dir.is_absolute():
        upload_dir = root / upload_dir
    origins = [origin.strip() for origin in os.getenv("ACADEMIA_CORS_ORIGINS", "").split(",") if origin.strip()]
    ollama_enabled = os.getenv("ACADEMIA_OLLAMA_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
    return Settings(
        root_dir=root,
        seed_dir=root / "data" / "seed",
        upload_dir=upload_dir,
        max_retrieval_loops=max(1, int(os.getenv("ACADEMIA_MAX_RETRIEVAL_LOOPS", "3"))),
        cors_origins=origins,
        ollama_enabled=ollama_enabled,
        ollama_base_url=os.getenv("ACADEMIA_OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/"),
        ollama_model=os.getenv("ACADEMIA_OLLAMA_MODEL", "gemma4:12b"),
        ollama_timeout_seconds=max(5, int(os.getenv("ACADEMIA_OLLAMA_TIMEOUT_SECONDS", "120"))),
    )
