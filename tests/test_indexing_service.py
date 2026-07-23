"""
Tests de IndexingService con providers de embeddings mockeados (rápidos,
sin cargar modelos reales) pero VectorRepository real contra Qdrant en
memoria — así se valida la orquestación real sin pagar el costo de
cargar e5-large/BM25 en cada test.
"""

from unittest.mock import MagicMock

import pytest
from qdrant_client.models import SparseVector

from rag_app.services.indexing_service import IndexingService


def make_fake_providers():
    """EmbeddingProvider y SparseEmbeddingProvider falsos: devuelven un
    vector por texto sin importar el contenido, suficiente para probar
    el batching/orquestación de IndexingService."""
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
def indexing_service(vector_repo):
    embedding_provider, sparse_provider = make_fake_providers()
    return IndexingService(embedding_provider, sparse_provider, vector_repo, batch_size=2)


def test_index_document_indexes_all_chunks_and_uses_both_embedding_types(
    indexing_service, sample_chunks_json
):
    result = indexing_service.index_document(sample_chunks_json, "doc_a")

    assert result["chunks_processed"] == 3
    assert result["chunks_indexed"] == 3
    # batch_size=2 sobre 3 chunks -> 2 batches
    assert result["batches"] == 2

    indexing_service.embedding_provider.encode.assert_called()
    indexing_service.sparse_embedding_provider.encode.assert_called()
    # Dense debe generarse con mode="passage" (prefijo e5 al indexar)
    for call in indexing_service.embedding_provider.encode.call_args_list:
        assert call.kwargs["mode"] == "passage"


def test_index_document_missing_file_raises(indexing_service, tmp_path):
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError):
        indexing_service.index_document(missing, "doc_a")


def test_index_document_empty_chunks_returns_zero_stats(indexing_service, tmp_path):
    empty_path = tmp_path / "chunks.json"
    empty_path.write_text("[]", encoding="utf-8")

    result = indexing_service.index_document(empty_path, "doc_a")

    assert result == {
        "document_name": "doc_a",
        "chunks_processed": 0,
        "chunks_indexed": 0,
        "batches": 0,
    }


def test_index_all_cached_documents_discovers_and_continues_on_error(
    indexing_service, tmp_path, sample_chunks
):
    import json

    cache_dir = tmp_path / "cache"
    good_dir = cache_dir / "good_doc" / "chunks"
    good_dir.mkdir(parents=True)
    (good_dir / "chunks.json").write_text(json.dumps(sample_chunks), encoding="utf-8")

    # Directorio sin chunks.json: debe ignorarse, no romper el batch completo
    (cache_dir / "no_chunks_doc").mkdir(parents=True)

    results = indexing_service.index_all_cached_documents(cache_dir)

    assert len(results) == 1
    assert results[0]["document_name"] == "good_doc"
    assert results[0]["chunks_indexed"] == 3


def test_search_uses_query_prefix_for_dense_and_passes_filters_through(
    indexing_service, sample_chunks_json
):
    indexing_service.index_document(sample_chunks_json, "doc_a")

    indexing_service.search("some query", top_k=5, page_filter=2, heading_contains="Results")

    dense_call = indexing_service.embedding_provider.encode_single.call_args
    assert dense_call.kwargs["mode"] == "query"

    sparse_call = indexing_service.sparse_embedding_provider.encode_single.call_args
    assert sparse_call.kwargs["mode"] == "query"
