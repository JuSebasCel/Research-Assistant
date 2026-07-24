"""Cliente Qdrant configurado a partir de Settings."""

from qdrant_client import QdrantClient

from rag_app.core.config import Settings


def create_qdrant_client(settings: Settings) -> QdrantClient:
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
