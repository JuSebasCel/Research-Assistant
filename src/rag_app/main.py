"""Punto de entrada de la aplicación FastAPI: configuración, middleware,
archivos estáticos y registro de routers."""

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from rag_app.api.chat import router as chat_router
from rag_app.api.documents import router as documents_router
from rag_app.api.ingest import router as ingest_router
from rag_app.api.settings import router as settings_router
from rag_app.core.config import get_settings
from rag_app.core.logging_config import configure_logging

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_cache_dir = Path(get_settings().cache_dir)
_cache_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/cache", StaticFiles(directory=_cache_dir), name="cache")

_uploads_dir = Path(get_settings().uploads_dir)
_uploads_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static/uploads", StaticFiles(directory=_uploads_dir), name="uploads")

app.include_router(chat_router)
app.include_router(documents_router)
app.include_router(ingest_router)
app.include_router(settings_router)


@app.get("/health")
async def health_check() -> dict[str, str]:
    return {"status": "ok"}
