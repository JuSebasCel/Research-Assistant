"""
Configuración centralizada de la aplicación.

Todas las variables de entorno del sistema pasan por aquí. Ningún otro
módulo debe leer `os.environ` directamente: eso dispersa el conocimiento
de qué configuración existe y elimina la validación de tipos.
"""

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# Evita que sentence-transformers/transformers hagan HEAD requests a
# Hugging Face Hub en cada arranque solo para chequear si hay una versión
# más nueva del modelo (el modelo ya está cacheado localmente). Sin esto,
# cada reinicio del proceso (ej. --reload) paga latencia de red extra
# antes de responder la primera pregunta. setdefault() respeta si el
# usuario ya seteó esto explícitamente (ej. para forzar descarga de un
# modelo nuevo la primera vez).
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class Settings(BaseSettings):
    """
    Define el *contrato* de configuración de la aplicación.

    Pydantic valida automáticamente tipos y presencia de cada campo al
    arrancar la app. Si falta una variable requerida o tiene el tipo
    incorrecto, la aplicación falla inmediatamente al iniciar (fail-fast).
    """

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

    # --- LLM (Gemini) ---
    gemini_api_key: str = ""
    # "gemini-2.5-flash" fue retirado para cuentas nuevas (404 real, no una
    # suposición) — usamos el alias que Google mantiene apuntando siempre
    # al Flash vigente, para no volver a romper cuando roten el nombre.
    gemini_model_name: str = "gemini-flash-latest"


@lru_cache
def get_settings() -> Settings:
    """
    Devuelve una instancia única (singleton) de Settings.

    Sin `lru_cache`, cada vez que alguien llame a `get_settings()` se
    reconstruiría el objeto y se releería el .env desde disco. Con
    `lru_cache`, la primera llamada construye el objeto; las siguientes
    reciben la misma instancia ya creada. Esto importa porque vamos a
    inyectar Settings en varios servicios y no queremos ese costo repetido
    en cada petición HTTP.
    """
    return Settings()