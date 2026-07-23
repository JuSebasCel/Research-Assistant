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
    SparseVectorParams,
    SparseVector,
    Modifier,
    PointStruct,
    Prefetch,
    FusionQuery,
    Fusion,
    Filter,
    FieldCondition,
    MatchValue,
    MatchAny,
    MatchText,
)

logger = logging.getLogger(__name__)

# Nombres de los named vectors dentro de cada punto. Qdrant requiere
# nombrar los vectores cuando una colección combina denso + sparse.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


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
        Crea dos named vectors por punto:
        - "dense": embedding semántico (e5), distancia COSINE.
        - "sparse": embedding léxico (BM25 vía fastembed), con
          Modifier.IDF para que Qdrant calcule el IDF real sobre el
          corpus indexado al momento de buscar (fastembed solo aporta
          la frecuencia de término al indexar).
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
                    vectors_config={
                        DENSE_VECTOR_NAME: VectorParams(
                            size=self.vector_size,
                            distance=Distance.COSINE,
                        ),
                    },
                    sparse_vectors_config={
                        SPARSE_VECTOR_NAME: SparseVectorParams(
                            modifier=Modifier.IDF,
                        ),
                    },
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
        dense_embeddings: List[List[float]],
        sparse_embeddings: List[SparseVector],
    ) -> int:
        """
        Inserta o actualiza chunks con sus embeddings (dense + sparse) en Qdrant.

        Usa UUIDs determinísticos basados en document_name + chunk_id,
        lo que hace la operación idempotente: si vuelves a indexar el
        mismo documento, los puntos se actualizan en lugar de duplicarse.

        Args:
            document_name: Nombre del documento (para namespace de UUIDs)
            chunks_data: Lista de dicts con datos de chunks (de chunks.json)
            dense_embeddings: Vectores semánticos (misma longitud que chunks_data)
            sparse_embeddings: Vectores léxicos BM25 (misma longitud que chunks_data)

        Returns:
            Número de puntos insertados

        Raises:
            ValueError: Si las longitudes no coinciden
        """
        if len(chunks_data) != len(dense_embeddings) or len(chunks_data) != len(sparse_embeddings):
            raise ValueError(
                f"Mismatch: {len(chunks_data)} chunks, {len(dense_embeddings)} dense "
                f"embeddings, {len(sparse_embeddings)} sparse embeddings"
            )

        if not chunks_data:
            logger.warning("No hay chunks para insertar")
            return 0

        points = []

        for chunk, dense_vec, sparse_vec in zip(chunks_data, dense_embeddings, sparse_embeddings):
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
                    vector={
                        DENSE_VECTOR_NAME: dense_vec,
                        SPARSE_VECTOR_NAME: sparse_vec,
                    },
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
    
    def _build_filter(
        self,
        document_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
        heading_contains: Optional[str] = None,
    ) -> Optional[Filter]:
        """
        Arma un Filter de Qdrant a partir de condiciones opcionales sobre
        la metadata ya presente en el payload (document_name, pages,
        headings). Devuelve None si no se pidió ningún filtro.
        """
        conditions = []

        if document_filter:
            conditions.append(
                FieldCondition(key="document_name", match=MatchValue(value=document_filter))
            )

        if page_filter is not None:
            # "pages" es una lista por punto; MatchAny hace match si el
            # valor está contenido en esa lista.
            conditions.append(
                FieldCondition(key="pages", match=MatchAny(any=[page_filter]))
            )

        if heading_contains:
            # Búsqueda de texto libre sobre "headings" (complementaria al
            # BM25 sobre "content").
            conditions.append(
                FieldCondition(key="headings", match=MatchText(text=heading_contains))
            )

        if not conditions:
            return None

        return Filter(must=conditions)

    def search(
        self,
        dense_query_vector: List[float],
        sparse_query_vector: SparseVector,
        limit: int = 5,
        score_threshold: Optional[float] = None,
        document_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
        heading_contains: Optional[str] = None,
        fetch_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda híbrida: fusiona resultados de la señal densa (semántica,
        coseno) y la señal sparse (léxica, BM25) mediante RRF nativo de
        Qdrant.

        Args:
            dense_query_vector: Vector semántico de la consulta (prefijo "query:")
            sparse_query_vector: Vector léxico de la consulta (query_embed)
            limit: Número máximo de resultados finales (post-fusión)
            score_threshold: Umbral mínimo de score RRF
            document_filter: Opcional, filtrar por document_name específico
            page_filter: Opcional, filtrar por número de página
            heading_contains: Opcional, filtrar por texto libre en headings
            fetch_k: Cuántos candidatos trae cada señal (dense/sparse) antes
                de fusionar; debe ser >= limit para que RRF tenga margen

        Returns:
            Lista de resultados con score y payload
        """
        try:
            query_filter = self._build_filter(document_filter, page_filter, heading_contains)

            # client.search() fue removido en qdrant-client >=1.14; la API
            # vigente es query_points(). Para hybrid search se usa prefetch
            # (una búsqueda por cada named vector) + FusionQuery(RRF), que
            # Qdrant resuelve server-side.
            response = self.client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=dense_query_vector,
                        using=DENSE_VECTOR_NAME,
                        limit=fetch_k,
                        filter=query_filter,
                    ),
                    Prefetch(
                        query=sparse_query_vector,
                        using=SPARSE_VECTOR_NAME,
                        limit=fetch_k,
                        filter=query_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                score_threshold=score_threshold,
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
            logger.error(f"Error en búsqueda híbrida: {e}")
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
