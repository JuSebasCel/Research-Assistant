"""
Punto de entrada de la aplicación FastAPI.

En esta etapa (fundación) este archivo solo prueba que las piezas base
encajan: configuración, logging, y la app misma arrancan correctamente.
La lógica de negocio (ingesta, retrieval, generación) vivirá en los
routers de `api/`, que se irán registrando aquí a medida que existan.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from rag_app.core.config import get_settings
from rag_app.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Hook de ciclo de vida de FastAPI: código antes del `yield` corre al
    arrancar la app; código después corre al apagarla. Es el lugar
    correcto para inicializar recursos compartidos (logging, y más
    adelante el pool de conexiones a la base de datos) una sola vez,
    en vez de recrearlos en cada request.
    """
    configure_logging()
    settings = get_settings()
    logger.info("Iniciando aplicación en entorno '%s'", settings.environment)
    yield
    logger.info("Apagando aplicación")


app = FastAPI(
    title="RAG Scientific Papers",
    description="Sistema RAG para análisis y síntesis de papers científicos",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Endpoint mínimo de verificación. No consulta la base de datos todavía
    (eso se agrega cuando exista la capa de repositorios) — por ahora
    solo confirma que el servidor HTTP y la configuración responden.
    """
    return {"status": "ok"}
