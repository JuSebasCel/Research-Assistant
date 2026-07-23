"""
Fixtures compartidas para los tests del pipeline de embeddings/indexing.

Los chunks sintéticos usan el mismo schema que produce DoclingChunker
(ver services/docling_chunker.py: ChunkData/ChunkMetadata vía asdict),
para que los tests ejerciten el código de parseo real (chunk["metadata"]
["docling_meta"], etc.) en vez de un formato inventado.
"""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from qdrant_client import QdrantClient
from qdrant_client.models import SparseVector

from rag_app.providers.embeddings import EmbeddingProvider
from rag_app.providers.sparse_embeddings import SparseEmbeddingProvider
from rag_app.repositories.vector_repository import VectorRepository
from rag_app.services.indexing_service import IndexingService

REAL_VECTOR_SIZE = 1024

# Dimensión pequeña a propósito: los tests de VectorRepository/IndexingService
# (con providers mockeados) no necesitan vectores e5 reales de 1024 dims,
# solo vectores de longitud consistente para ejercitar la lógica de Qdrant.
FAKE_VECTOR_SIZE = 4


def _make_chunk(
    chunk_id: str, index: int, content: str, heading: str, page: int, token_count: int = 10
):
    return {
        "chunk_id": chunk_id,
        "index": index,
        "content": content,
        "contextualized_text": f"{heading}\n{content}",
        "enriched_text": None,
        "metadata": {
            "docling_meta": {
                "headings": [heading],
                "doc_items": [
                    {"self_ref": f"#/texts/{index}", "prov": [{"page_no": page}]}
                ],
            }
        },
        "token_count": token_count,
        "image_paths": [],
    }


@pytest.fixture
def sample_chunks() -> list[dict]:
    """3 chunks sintéticos con page/heading/contenido distintos entre sí,
    para poder probar filtros y ranking de forma determinística."""
    return [
        _make_chunk(
            "chunk_0001", 0,
            "The Transformer architecture uses self-attention mechanisms.",
            "Introduction", page=1,
        ),
        _make_chunk(
            "chunk_0002", 1,
            "We trained the model using the Adam optimizer with warmup steps.",
            "Methods", page=2,
        ),
        _make_chunk(
            "chunk_0003", 2,
            "Our model achieved a BLEU score of 28.4 on the WMT dataset.",
            "Results", page=3,
        ),
    ]


@pytest.fixture
def sample_chunks_json(tmp_path, sample_chunks) -> Path:
    """Escribe sample_chunks a un chunks.json temporal (formato real de
    DoclingChunker), para probar IndexingService.index_document tal como
    lee el archivo en producción."""
    path = tmp_path / "chunks.json"
    path.write_text(json.dumps(sample_chunks), encoding="utf-8")
    return path


@pytest.fixture
def fake_dense_vectors() -> dict[str, list[float]]:
    """Vectores densos ortogonales por chunk_id: predecibles para saber
    exactamente qué chunk debería ganar en una búsqueda dada."""
    return {
        "chunk_0001": [1.0, 0.0, 0.0, 0.0],
        "chunk_0002": [0.0, 1.0, 0.0, 0.0],
        "chunk_0003": [0.0, 0.0, 1.0, 0.0],
    }


@pytest.fixture
def fake_sparse_vectors() -> dict[str, SparseVector]:
    """Vectores sparse con un término único por chunk_id (simula BM25)."""
    return {
        "chunk_0001": SparseVector(indices=[10], values=[2.0]),
        "chunk_0002": SparseVector(indices=[20], values=[2.0]),
        "chunk_0003": SparseVector(indices=[30], values=[2.0]),
    }


@pytest.fixture
def qdrant_memory_client() -> QdrantClient:
    """Cliente Qdrant en memoria: mismo mecanismo que usamos para validar
    manualmente el hybrid search sin necesitar Docker."""
    return QdrantClient(":memory:")


@pytest.fixture
def vector_repo(qdrant_memory_client) -> VectorRepository:
    repo = VectorRepository(
        client=qdrant_memory_client,
        collection_name="test_collection",
        vector_size=FAKE_VECTOR_SIZE,
    )
    repo.ensure_collection_exists()
    return repo


def make_fake_providers():
    """EmbeddingProvider y SparseEmbeddingProvider falsos: devuelven un
    vector por texto sin importar el contenido, suficiente para probar
    orquestación sin cargar modelos reales."""
    fake_dense = [0.1, 0.2, 0.3, 0.4]
    fake_sparse = SparseVector(indices=[1], values=[1.0])

    embedding_provider = MagicMock()
    embedding_provider.encode.side_effect = lambda texts, mode: [fake_dense for _ in texts]
    embedding_provider.encode_single.side_effect = lambda text, mode: fake_dense

    sparse_provider = MagicMock()
    sparse_provider.encode.side_effect = lambda texts, mode: [fake_sparse for _ in texts]
    sparse_provider.encode_single.side_effect = lambda text, mode: fake_sparse

    return embedding_provider, sparse_provider


@pytest.fixture
def indexing_service(vector_repo) -> IndexingService:
    embedding_provider, sparse_provider = make_fake_providers()
    return IndexingService(embedding_provider, sparse_provider, vector_repo, batch_size=2)


@pytest.fixture(scope="session")
def real_embedding_provider():
    """Modelo e5-large real. Session-scoped: cargarlo toma ~8s, se paga una
    sola vez por corrida de tests en vez de por test."""
    return EmbeddingProvider(model_name="intfloat/multilingual-e5-large")


@pytest.fixture(scope="session")
def real_sparse_embedding_provider():
    return SparseEmbeddingProvider()


@pytest.fixture
def real_indexing_service(
    qdrant_memory_client, real_embedding_provider, real_sparse_embedding_provider
):
    repo = VectorRepository(
        client=qdrant_memory_client,
        collection_name="integration_test",
        vector_size=REAL_VECTOR_SIZE,
    )
    repo.ensure_collection_exists()
    return IndexingService(
        real_embedding_provider, real_sparse_embedding_provider, repo, batch_size=32
    )
