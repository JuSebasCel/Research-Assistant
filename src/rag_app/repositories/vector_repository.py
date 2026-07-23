"""
Repository para operaciones con Qdrant (vector database).

Encapsula todas las operaciones CRUD sobre la base de datos vectorial,
aislando la lógica de negocio de los detalles de implementación de Qdrant.

Responsabilidades:
- Crear/verificar colecciones
- Upsert de puntos (con batch)
- Búsqueda vectorial
- Eliminación de documentos
- Estadísticas de la colección
"""

import logging
import uuid
from typing import List, Dict, Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
)

logger = logging.getLogger(__name__)


class VectorRepository:
    """
    Repository para operaciones vectoriales con Qdrant.
    
    Maneja la persistencia de embeddings y metadata de chunks en Qdrant.
    Todos los métodos son idempotentes donde sea posible.
    """
    
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
    ):
        """
        Inicializa el repository.
        
        Args:
            client: Cliente Qdrant configurado
            collection_name: Nombre de la colección a usar
            vector_size: Dimensión de los vectores (debe coincidir con modelo de embeddings)
        """
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
    
    def ensure_collection_exists(self) -> None:
        """
        Verifica que la colección existe, si no la crea.
        
        Operación idempotente: si la colección ya existe, no hace nada.
        Usa distancia COSINE para similitud (apropiado para embeddings normalizados).
        """
        try:
            collections = self.client.get_collections().collections
            collection_exists = any(
                col.name == self.collection_name for col in collections
            )
            
            if not collection_exists:
                logger.info(f"Creando colección '{self.collection_name}'")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(f"✓ Colección '{self.collection_name}' creada")
            else:
                logger.debug(f"Colección '{self.collection_name}' ya existe")
        
        except Exception as e:
            logger.error(f"Error al verificar/crear colección: {e}")
            raise
    
    def upsert_chunks(
        self,
        document_name: str,
        chunks_data: List[Dict[str, Any]],
        embeddings: List[List[float]],
    ) -> int:
        """
        Inserta o actualiza chunks con sus embeddings en Qdrant.
        
        Usa UUIDs determinísticos basados en document_name + chunk_id,
        lo que hace la operación idempotente: si vuelves a indexar el
        mismo documento, los puntos se actualizan en lugar de duplicarse.
        
        Args:
            document_name: Nombre del documento (para namespace de UUIDs)
            chunks_data: Lista de dicts con datos de chunks (de chunks.json)
            embeddings: Lista de vectores (misma longitud que chunks_data)
        
        Returns:
            Número de puntos insertados
        
        Raises:
            ValueError: Si las longitudes no coinciden
        """
        if len(chunks_data) != len(embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks_data)} chunks pero {len(embeddings)} embeddings"
            )
        
        if not chunks_data:
            logger.warning("No hay chunks para insertar")
            return 0
        
        points = []
        
        for chunk, embedding in zip(chunks_data, embeddings):
            # UUID determinístico: mismo documento + mismo chunk = mismo UUID
            point_id = str(
                uuid.uuid5(
                    uuid.NAMESPACE_DNS,
                    f"{document_name}:{chunk['chunk_id']}"
                )
            )
            
            # Extraer metadata relevante (NO todo docling_meta crudo)
            meta = chunk["metadata"]["docling_meta"]
            
            # Extraer páginas de provenance
            pages = sorted({
                prov["page_no"]
                for doc_item in meta.get("doc_items", [])
                for prov in doc_item.get("prov", [])
                if "page_no" in prov
            })
            
            # Payload limpio: solo lo necesario para filtrado y display
            payload = {
                "document_name": document_name,
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "contextualized_text": chunk["contextualized_text"],
                "headings": meta.get("headings", []),
                "pages": pages,
                "image_paths": chunk.get("image_paths", []),
                "token_count": chunk["token_count"],
            }
            
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )
        
        # Upsert en Qdrant
        try:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )
            logger.info(
                f"✓ {len(points)} chunks indexados para '{document_name}'"
            )
            return len(points)
        
        except Exception as e:
            logger.error(f"Error al insertar puntos en Qdrant: {e}")
            raise
    
    def search(
        self,
        query_vector: List[float],
        limit: int = 5,
        score_threshold: Optional[float] = None,
        document_filter: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda vectorial por similitud coseno.
        
        Args:
            query_vector: Vector de la consulta (ya embedido con prefijo "query:")
            limit: Número máximo de resultados
            score_threshold: Umbral mínimo de score (0-1 para cosine)
            document_filter: Opcional, filtrar por document_name específico
        
        Returns:
            Lista de resultados con score y payload
        """
        try:
            # Construir filtro si es necesario
            query_filter = None
            if document_filter:
                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="document_name",
                            match=MatchValue(value=document_filter),
                        )
                    ]
                )

            # client.search() fue removido en qdrant-client >=1.14; la API
            # vigente es query_points(), que devuelve un QueryResponse (.points)
            # en vez de una lista directa de ScoredPoint.
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                query_filter=query_filter,
            )

            # Formatear resultados
            formatted_results = []
            for result in response.points:
                formatted_results.append({
                    "id": result.id,
                    "score": result.score,
                    "chunk_id": result.payload["chunk_id"],
                    "document_name": result.payload["document_name"],
                    "content": result.payload["content"],
                    "headings": result.payload.get("headings", []),
                    "pages": result.payload.get("pages", []),
                    "image_paths": result.payload.get("image_paths", []),
                    "token_count": result.payload.get("token_count", 0),
                })
            
            return formatted_results
        
        except Exception as e:
            logger.error(f"Error en búsqueda vectorial: {e}")
            raise
    
    def delete_document(self, document_name: str) -> int:
        """
        Elimina todos los chunks de un documento.
        
        Args:
            document_name: Nombre del documento a eliminar
        
        Returns:
            Número de puntos eliminados (aproximado)
        """
        try:
            # Contar primero cuántos puntos tiene el documento
            count_filter = Filter(
                must=[
                    FieldCondition(
                        key="document_name",
                        match=MatchValue(value=document_name),
                    )
                ]
            )
            
            # Qdrant no tiene count directo, hacemos scroll para contar
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=count_filter,
                limit=10000,  # Suficiente para documentos grandes
                with_payload=False,
                with_vectors=False,
            )
            
            count = len(points)
            
            if count == 0:
                logger.warning(f"No se encontraron chunks para '{document_name}'")
                return 0
            
            # Eliminar con filtro
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=count_filter,
            )
            
            logger.info(f"✓ Eliminados ~{count} chunks de '{document_name}'")
            return count
        
        except Exception as e:
            logger.error(f"Error al eliminar documento: {e}")
            raise
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la colección.
        
        Returns:
            Dict con información de la colección
        """
        try:
            info = self.client.get_collection(self.collection_name)

            # info.indexed_vectors_count cuenta solo vectores ya incorporados
            # al índice HNSW (0 por debajo del indexing_threshold, ~20k
            # puntos), no el total almacenado. Cada punto tiene un único
            # vector denso, así que points_count ya es el conteo real.
            return {
                "collection_name": self.collection_name,
                "points_count": info.points_count,
                "status": info.status,
            }
        
        except Exception as e:
            logger.error(f"Error al obtener stats de colección: {e}")
            raise
    
    def list_documents(self) -> List[str]:
        """
        Lista todos los documentos únicos en la colección.
        
        Returns:
            Lista de nombres de documentos
        """
        try:
            # Scroll sobre toda la colección (solo payload, no vectores)
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )
            
            # Extraer document_names únicos
            document_names = {
                point.payload.get("document_name")
                for point in points
                if point.payload.get("document_name")
            }
            
            return sorted(document_names)
        
        except Exception as e:
            logger.error(f"Error al listar documentos: {e}")
            raise
