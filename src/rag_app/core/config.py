"""Configuración centralizada — toda variable de entorno pasa por acá,
ningún otro módulo lee `os.environ` directamente."""

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# El modelo de embeddings ya está cacheado localmente; sin esto,
# sentence-transformers hace un HEAD request a HF Hub en cada arranque
# solo para chequear versión, añadiendo latencia de red gratuita.
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Entorno de ejecución ---
    environment: Literal["development", "testing", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    # --- Base de datos ---
    database_url: str = (
        "postgresql://raguser:ragpass@localhost:5432/rag_scientific"
    )

    # --- Qdrant ---
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection_name: str = "scientific_papers"

    # --- Embeddings ---
    embedding_model_name: str = "intfloat/multilingual-e5-large"
    embedding_dimension: int = 1024

    # --- Cache y almacenamiento ---
    cache_dir: str = "data/cache"
    audit_dir: str = "data/audit"
    uploads_dir: str = "data/uploads"
    # No regenerable como cache_dir: un rename o carpeta puesta a mano no
    # debe perderse si se limpia el cache.
    document_metadata_path: str = "data/document_metadata.json"
    # Contiene secretos (API keys propias de usuario) — ver .gitignore.
    app_settings_path: str = "data/app_settings.json"

    # --- LLM (Gemini) ---
    gemini_api_key: str = ""
    # Alias que Google mantiene apuntando al Flash vigente; el nombre
    # pinneado "gemini-2.5-flash" fue retirado para cuentas nuevas.
    gemini_model_name: str = "gemini-flash-latest"


@lru_cache
def get_settings() -> Settings:
    return Settings()