"""
Provider del cliente Qdrant.

Este módulo proporciona una instancia configurada del cliente Qdrant
que se usará en los repositories para realizar operaciones vectoriales.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

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


async def ensure_collection_exists(
    client: QdrantClient,
    collection_name: str,
    vector_size: int,
) -> None:
    """
    Verifica que la colección existe, si no la crea.
    
    Args:
        client: Cliente Qdrant
        collection_name: Nombre de la colección a verificar/crear
        vector_size: Dimensión de los vectores (debe coincidir con el modelo de embeddings)
    
    Explicación:
        - Distance.COSINE: Usa similitud coseno para buscar vectores cercanos
        - VectorParams: Define la estructura de los vectores (tamaño y métrica)
        - recreate_collection si no existe crea una nueva, si existe no hace nada
    """
    collections = client.get_collections().collections
    collection_exists = any(col.name == collection_name for col in collections)
    
    if not collection_exists:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE,
            ),
        )
