"""
Indexación y búsqueda híbrida de documentos en Qdrant.

Pipeline: chunks.json -> EmbeddingProvider (prefijo "passage:") -> VectorRepository -> Qdrant
"""

import json
import logging
from pathlib import Path
from typing import Any

from rag_app.providers.embeddings import EmbeddingProvider
from rag_app.providers.sparse_embeddings import SparseEmbeddingProvider
from rag_app.repositories.vector_repository import VectorRepository

logger = logging.getLogger(__name__)


class IndexingService:
    def __init__(
        self,
        embedding_provider: EmbeddingProvider,
        sparse_embedding_provider: SparseEmbeddingProvider,
        vector_repository: VectorRepository,
        batch_size: int = 32,
    ):
        self.embedding_provider = embedding_provider
        self.sparse_embedding_provider = sparse_embedding_provider
        self.vector_repository = vector_repository
        self.batch_size = batch_size

    def index_document(
        self,
        chunks_json_path: Path,
        document_name: str,
    ) -> dict[str, Any]:
        """Genera embeddings en batches y hace upsert idempotente en Qdrant."""
        if not chunks_json_path.exists():
            raise FileNotFoundError(f"Chunks no encontrados: {chunks_json_path}")

        logger.info(f"Indexando documento: {document_name}")

        try:
            with open(chunks_json_path, encoding="utf-8") as f:
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

        all_dense_embeddings = []
        all_sparse_embeddings = []
        num_batches = (len(chunks) + self.batch_size - 1) // self.batch_size

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            batch_num = i // self.batch_size + 1

            logger.debug(f"Procesando batch {batch_num}/{num_batches} ({len(batch)} chunks)")

            # contextualized_text (con headings) para la señal densa, content
            # crudo para la sparse: mantiene la señal léxica complementaria a
            # la densa en vez de redundante, mejora la fusión RRF.
            dense_texts = [chunk["contextualized_text"] for chunk in batch]
            dense_embeddings = self.embedding_provider.encode(dense_texts, mode="passage")
            all_dense_embeddings.extend(dense_embeddings)

            sparse_texts = [chunk["content"] for chunk in batch]
            sparse_embeddings = self.sparse_embedding_provider.encode(
                sparse_texts, mode="passage"
            )
            all_sparse_embeddings.extend(sparse_embeddings)

        logger.info(f"Generados {len(all_dense_embeddings)} embeddings (dense + sparse)")

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

    def index_all_cached_documents(self, cache_dir: Path) -> list[dict[str, Any]]:
        """Indexa todos los documentos con chunks.json bajo cache_dir."""
        if not cache_dir.exists():
            logger.warning(f"Cache dir no existe: {cache_dir}")
            return []

        results = []
        for doc_dir in cache_dir.iterdir():
            if not doc_dir.is_dir():
                continue

            chunks_json = doc_dir / "chunks" / "chunks.json"
            if not chunks_json.exists():
                continue

            document_name = doc_dir.name
            try:
                results.append(
                    self.index_document(chunks_json_path=chunks_json, document_name=document_name)
                )
            except Exception as e:
                logger.error(f"Error indexando {document_name}: {e}")
                results.append({"document_name": document_name, "error": str(e)})

        return results

    def reindex_document(self, chunks_json_path: Path, document_name: str) -> dict[str, Any]:
        try:
            deleted_count = self.vector_repository.delete_document(document_name)
            logger.info(f"Eliminados {deleted_count} chunks antiguos")
        except Exception as e:
            logger.warning(f"Error al eliminar chunks antiguos: {e}")

        return self.index_document(chunks_json_path=chunks_json_path, document_name=document_name)

    def search(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float | None = None,
        document_filter: str | None = None,
        page_filter: int | None = None,
        heading_contains: str | None = None,
    ) -> list[dict[str, Any]]:
        """Búsqueda híbrida (dense + sparse, fusionadas con RRF)."""
        logger.info(f"Búsqueda: '{query}' (top_k={top_k})")

        dense_query_vector = self.embedding_provider.encode_single(query, mode="query")
        sparse_query_vector = self.sparse_embedding_provider.encode_single(query, mode="query")

        results = self.vector_repository.search(
            dense_query_vector=dense_query_vector,
            sparse_query_vector=sparse_query_vector,
            limit=top_k,
            score_threshold=score_threshold,
            document_filter=document_filter,
            page_filter=page_filter,
            heading_contains=heading_contains,
        )

        logger.info(f"Encontrados {len(results)} resultados")
        return results

    def search_across_documents(
        self,
        query: str,
        max_total: int = 10,
        per_doc_top_k: int = 3,
        page_filter: int | None = None,
        heading_contains: str | None = None,
        document_names: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Búsqueda repetida por cada documento en vez de una sola búsqueda
        global: un documento que domina el ranking puede dejar a otros
        completamente afuera aunque también sean relevantes. document_names
        opcionalmente acota el fan-out a un subconjunto (ej. una carpeta) en
        vez de todos los documentos indexados.
        """
        logger.info(f"Búsqueda multi-documento: '{query}' (max_total={max_total})")

        # Se calcula una sola vez y se reusa por documento: el costo caro es
        # encode(), no la búsqueda en Qdrant en sí.
        dense_query_vector = self.embedding_provider.encode_single(query, mode="query")
        sparse_query_vector = self.sparse_embedding_provider.encode_single(query, mode="query")

        if document_names is not None:
            documents = document_names
        else:
            documents = self.vector_repository.list_documents()

        all_results: list[dict[str, Any]] = []
        for document_name in documents:
            results = self.vector_repository.search(
                dense_query_vector=dense_query_vector,
                sparse_query_vector=sparse_query_vector,
                limit=per_doc_top_k,
                document_filter=document_name,
                page_filter=page_filter,
                heading_contains=heading_contains,
            )
            all_results.extend(results)

        all_results.sort(key=lambda r: r["score"], reverse=True)
        final_results = all_results[:max_total]

        logger.info(
            f"{len(final_results)} resultados de {len(documents)} documentos "
            f"({len(all_results)} candidatos antes de recortar)"
        )
        return final_results

    def get_stats(self) -> dict[str, Any]:
        return self.vector_repository.get_collection_stats()

    def list_indexed_documents(self) -> list[str]:
        return self.vector_repository.list_documents()
