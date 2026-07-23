"""
Servicio de indexación de documentos.

Orquesta el proceso completo de generación de embeddings e indexación
en Qdrant. Maneja batching, prefijos e5, y coordinación entre componentes.

Pipeline:
    chunks.json → EmbeddingProvider (con prefijo "passage:") → VectorRepository → Qdrant

Responsabilidades:
- Cargar chunks desde cache
- Generar embeddings en batches (eficiencia)
- Coordinar con VectorRepository para persistencia
- Manejo de errores y logging
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from rag_app.providers.embeddings import EmbeddingProvider
from rag_app.providers.sparse_embeddings import SparseEmbeddingProvider
from rag_app.repositories.vector_repository import VectorRepository

logger = logging.getLogger(__name__)


class IndexingService:
    """
    Servicio que maneja la indexación de documentos en el vector store.
    
    Coordina la generación de embeddings (con prefijos e5 correctos) y
    la persistencia en Qdrant vía VectorRepository.
    """
    
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        sparse_embedding_provider: SparseEmbeddingProvider,
        vector_repository: VectorRepository,
        batch_size: int = 32,
    ):
        """
        Inicializa el servicio de indexación.

        Args:
            embedding_provider: Provider para generar embeddings densos (semánticos)
            sparse_embedding_provider: Provider para generar embeddings sparse (BM25)
            vector_repository: Repository para operaciones con Qdrant
            batch_size: Tamaño de batch para embeddings (32-64 recomendado)
        """
        self.embedding_provider = embedding_provider
        self.sparse_embedding_provider = sparse_embedding_provider
        self.vector_repository = vector_repository
        self.batch_size = batch_size
    
    def index_document(
        self,
        chunks_json_path: Path,
        document_name: str,
    ) -> Dict[str, Any]:
        """
        Indexa un documento completo (todos sus chunks) en Qdrant.
        
        Lee chunks.json, genera embeddings en batches con prefijo "passage:",
        y los sube a Qdrant con UUIDs determinísticos (idempotente).
        
        Args:
            chunks_json_path: Path a chunks.json del documento
            document_name: Nombre del documento (para namespace de UUIDs)
        
        Returns:
            Dict con estadísticas de la indexación
        
        Raises:
            FileNotFoundError: Si chunks.json no existe
            ValueError: Si chunks.json está mal formado
        """
        if not chunks_json_path.exists():
            raise FileNotFoundError(f"Chunks no encontrados: {chunks_json_path}")
        
        logger.info(f"Indexando documento: {document_name}")
        
        # Cargar chunks
        try:
            with open(chunks_json_path, "r", encoding="utf-8") as f:
                chunks = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Error al parsear chunks.json: {e}")
        
        if not chunks:
            logger.warning(f"No hay chunks para indexar en {document_name}")
            return {
                "document_name": document_name,
                "chunks_processed": 0,
                "chunks_indexed": 0,
                "batches": 0,
            }
        
        logger.info(f"Cargados {len(chunks)} chunks")
        
        # Generar embeddings (dense + sparse) en batches
        all_dense_embeddings = []
        all_sparse_embeddings = []
        num_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.debug(
                f"Procesando batch {batch_num}/{num_batches} "
                f"({len(batch)} chunks)"
            )

            # Denso (semántico): usa contextualized_text con prefijo
            # "passage:" (e5 requirement) — captura el contexto de headings.
            dense_texts = [chunk["contextualized_text"] for chunk in batch]
            dense_embeddings = self.embedding_provider.encode(
                dense_texts,
                mode="passage",  # Esto añade "passage: " automáticamente
            )
            all_dense_embeddings.extend(dense_embeddings)

            # Sparse (léxico/BM25): usa content crudo, no contextualized_text,
            # para que la señal léxica sea complementaria a la densa en vez
            # de redundante con ella (mejora la fusión RRF).
            sparse_texts = [chunk["content"] for chunk in batch]
            sparse_embeddings = self.sparse_embedding_provider.encode(
                sparse_texts,
                mode="passage",
            )
            all_sparse_embeddings.extend(sparse_embeddings)

        logger.info(f"✓ Generados {len(all_dense_embeddings)} embeddings (dense + sparse)")

        # Indexar en Qdrant
        chunks_indexed = self.vector_repository.upsert_chunks(
            document_name=document_name,
            chunks_data=chunks,
            dense_embeddings=all_dense_embeddings,
            sparse_embeddings=all_sparse_embeddings,
        )
        
        return {
            "document_name": document_name,
            "chunks_processed": len(chunks),
            "chunks_indexed": chunks_indexed,
            "batches": num_batches,
        }
    
    def index_all_cached_documents(
        self,
        cache_dir: Path,
    ) -> List[Dict[str, Any]]:
        """
        Indexa todos los documentos que tienen chunks en cache.
        
        Busca automáticamente todos los subdirectorios en cache_dir que
        contengan chunks/chunks.json y los indexa.
        
        Args:
            cache_dir: Directorio base de cache (data/cache)
        
        Returns:
            Lista de resultados de indexación por documento
        """
        if not cache_dir.exists():
            logger.warning(f"Cache dir no existe: {cache_dir}")
            return []
        
        results = []
        
        # Buscar todos los chunks.json en cache
        for doc_dir in cache_dir.iterdir():
            if not doc_dir.is_dir():
                continue
            
            chunks_json = doc_dir / "chunks" / "chunks.json"
            
            if not chunks_json.exists():
                logger.debug(f"Sin chunks.json: {doc_dir.name}")
                continue
            
            document_name = doc_dir.name
            
            try:
                result = self.index_document(
                    chunks_json_path=chunks_json,
                    document_name=document_name,
                )
                results.append(result)
            
            except Exception as e:
                logger.error(f"Error indexando {document_name}: {e}")
                results.append({
                    "document_name": document_name,
                    "error": str(e),
                })
        
        return results
    
    def reindex_document(
        self,
        chunks_json_path: Path,
        document_name: str,
    ) -> Dict[str, Any]:
        """
        Re-indexa un documento (elimina e indexa nuevamente).
        
        Útil cuando has modificado el chunking y quieres refrescar
        los embeddings sin dejar versiones antiguas en Qdrant.
        
        Args:
            chunks_json_path: Path a chunks.json del documento
            document_name: Nombre del documento
        
        Returns:
            Dict con estadísticas de la re-indexación
        """
        logger.info(f"Re-indexando documento: {document_name}")
        
        # Eliminar versión anterior
        try:
            deleted_count = self.vector_repository.delete_document(document_name)
            logger.info(f"✓ Eliminados {deleted_count} chunks antiguos")
        except Exception as e:
            logger.warning(f"Error al eliminar chunks antiguos: {e}")
        
        # Indexar versión nueva
        return self.index_document(
            chunks_json_path=chunks_json_path,
            document_name=document_name,
        )
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
        document_filter: Optional[str] = None,
        page_filter: Optional[int] = None,
        heading_contains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Búsqueda híbrida (semántica + léxica, fusionadas con RRF) sobre
        documentos indexados.

        Args:
            query: Consulta del usuario (texto natural)
            top_k: Número de chunks a retornar
            score_threshold: Umbral mínimo de score RRF
            document_filter: Opcional, buscar solo en documento específico
            page_filter: Opcional, buscar solo en una página específica
            heading_contains: Opcional, filtrar por texto libre en headings

        Returns:
            Lista de chunks relevantes con scores
        """
        logger.info(f"Búsqueda: '{query}' (top_k={top_k})")

        # Denso: prefijo "query:" (e5 requirement)
        dense_query_vector = self.embedding_provider.encode_single(
            query,
            mode="query",
        )

        # Sparse: sin ponderación IDF propia (la aplica Qdrant server-side
        # vía Modifier.IDF sobre el corpus indexado)
        sparse_query_vector = self.sparse_embedding_provider.encode_single(
            query,
            mode="query",
        )

        # Buscar en Qdrant (fusión RRF de ambas señales)
        results = self.vector_repository.search(
            dense_query_vector=dense_query_vector,
            sparse_query_vector=sparse_query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            document_filter=document_filter,
            page_filter=page_filter,
            heading_contains=heading_contains,
        )

        logger.info(f"✓ Encontrados {len(results)} resultados")

        return results
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del estado de la indexación.
        
        Returns:
            Dict con información de la colección Qdrant
        """
        return self.vector_repository.get_collection_stats()
    
    def list_indexed_documents(self) -> List[str]:
        """
        Lista todos los documentos indexados en Qdrant.
        
        Returns:
            Lista de nombres de documentos
        """
        return self.vector_repository.list_documents()
