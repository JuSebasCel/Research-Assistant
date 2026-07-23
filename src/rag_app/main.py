"""
Punto de entrada de la aplicación FastAPI.

En esta etapa (fundación) este archivo solo prueba que las piezas base
encajan: configuración, logging, y la app misma arrancan correctamente.
La lógica de negocio (ingesta, retrieval, generación) vivirá en los
routers de `api/`, que se irán registrando aquí a medida que existan.
"""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from rag_app.api.chat import router as chat_router
from rag_app.api.documents import router as documents_router
from rag_app.api.ingest import router as ingest_router
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

# CORS para el frontend (Vite dev server, puerto propio distinto al backend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Sirve las figuras extraídas (data/cache/{doc}/figures/*.png) para que el
# frontend pueda mostrarlas en el panel de fuentes.
_cache_dir = Path(get_settings().cache_dir)
_cache_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/cache", StaticFiles(directory=_cache_dir), name="cache")

app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(ingest_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    """
    Endpoint mínimo de verificación. No consulta la base de datos todavía
    (eso se agrega cuando exista la capa de repositorios) — por ahora
    solo confirma que el servidor HTTP y la configuración responden.
    """
    return {"status": "ok"}
