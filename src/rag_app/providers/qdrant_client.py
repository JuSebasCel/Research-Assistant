"""
Provider del cliente Qdrant.

Este módulo proporciona una instancia configurada del cliente Qdrant
que se usará en los repositories para realizar operaciones vectoriales.
"""

from qdrant_client import QdrantClient

from rag_app.core.config import Settings


def create_qdrant_client(settings: Settings) -> QdrantClient:
    """
    Crea y retorna un cliente Qdrant configurado.

    Args:
        settings: Configuración de la aplicación con host/port de Qdrant

    Returns:
        Cliente Qdrant conectado y listo para usar
    """
    return QdrantClient(
        host=settings.qdrant_host,
        port=settings.qdrant_port,
    )
