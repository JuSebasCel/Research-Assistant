"""Repository de operaciones CRUD sobre la colección Qdrant (crear/verificar,
upsert, búsqueda híbrida, borrado, estadísticas)."""

import logging
import uuid
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchText,
    MatchValue,
    Modifier,
    PointStruct,
    Prefetch,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

logger = logging.getLogger(__name__)

# Nombres de los named vectors dentro de cada punto. Qdrant requiere
# nombrar los vectores cuando una colección combina denso + sparse.
DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"


class VectorRepository:
    def __init__(
        self,
        client: QdrantClient,
        collection_name: str,
        vector_size: int,
    ):
        self.client = client
        self.collection_name = collection_name
        self.vector_size = vector_size
    
    def ensure_collection_exists(self) -> None:
        """
        Idempotente. Crea dos named vectors por punto: "dense" (e5,
        COSINE) y "sparse" (BM25 vía fastembed, con Modifier.IDF para que
        Qdrant calcule el IDF real sobre el corpus indexado al buscar —
        fastembed solo aporta la frecuencia de término al indexar).
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
        chunks_data: list[dict[str, Any]],
        dense_embeddings: list[list[float]],
        sparse_embeddings: list[SparseVector],
    ) -> int:
        """UUID determinístico (document_name + chunk_id) hace la operación
        idempotente: re-indexar el mismo documento actualiza en vez de duplicar."""
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
            point_id = str(
                uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_name}:{chunk['chunk_id']}")
            )

            meta = chunk["metadata"]["docling_meta"]
            pages = sorted({
                prov["page_no"]
                for doc_item in meta.get("doc_items", [])
                for prov in doc_item.get("prov", [])
                if "page_no" in prov
            })

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
        
        try:
            self.client.upsert(collection_name=self.collection_name, points=points)
            logger.info(f"✓ {len(points)} chunks indexados para '{document_name}'")
            return len(points)

        except Exception as e:
            logger.error(f"Error al insertar puntos en Qdrant: {e}")
            raise
    
    def _build_filter(
        self,
        document_filter: str | None = None,
        page_filter: int | None = None,
        heading_contains: str | None = None,
    ) -> Filter | None:
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
        dense_query_vector: list[float],
        sparse_query_vector: SparseVector,
        limit: int = 5,
        score_threshold: float | None = None,
        document_filter: str | None = None,
        page_filter: int | None = None,
        heading_contains: str | None = None,
        fetch_k: int = 20,
    ) -> list[dict[str, Any]]:
        """Fusiona la señal densa (coseno) y sparse (BM25) vía RRF nativo de
        Qdrant. fetch_k es cuántos candidatos trae cada señal antes de
        fusionar, debe ser >= limit para que RRF tenga margen."""
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
        try:
            count_filter = Filter(
                must=[FieldCondition(key="document_name", match=MatchValue(value=document_name))]
            )

            # Qdrant no tiene count directo, se usa scroll para contar.
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=count_filter,
                limit=10000,
                with_payload=False,
                with_vectors=False,
            )

            count = len(points)
            if count == 0:
                logger.warning(f"No se encontraron chunks para '{document_name}'")
                return 0

            self.client.delete(collection_name=self.collection_name, points_selector=count_filter)
            logger.info(f"✓ Eliminados ~{count} chunks de '{document_name}'")
            return count

        except Exception as e:
            logger.error(f"Error al eliminar documento: {e}")
            raise

    def get_collection_stats(self) -> dict[str, Any]:
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
    
    def list_documents(self) -> list[str]:
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,
                with_payload=True,
                with_vectors=False,
            )

            document_names = {
                point.payload.get("document_name")
                for point in points
                if point.payload.get("document_name")
            }

            return sorted(document_names)

        except Exception as e:
            logger.error(f"Error al listar documentos: {e}")
            raise
