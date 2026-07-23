"""
Tests de VectorRepository contra Qdrant en memoria (sin Docker).

Cubren lo que validamos manualmente durante el desarrollo del hybrid
search: idempotencia de upsert, no-colisión de UUIDs entre documentos,
fusión dense+sparse, y los tres filtros de metadata.
"""

import pytest
from qdrant_client.models import SparseVector

from rag_app.repositories.vector_repository import DENSE_VECTOR_NAME, SPARSE_VECTOR_NAME

BROAD_DENSE_QUERY = [0.34, 0.33, 0.33, 0.0]
BROAD_SPARSE_QUERY = SparseVector(indices=[10, 20, 30], values=[1.0, 1.0, 1.0])


def upsert(vector_repo, document_name, chunks, fake_dense_vectors, fake_sparse_vectors):
    dense = [fake_dense_vectors[c["chunk_id"]] for c in chunks]
    sparse = [fake_sparse_vectors[c["chunk_id"]] for c in chunks]
    return vector_repo.upsert_chunks(document_name, chunks, dense, sparse)


def test_ensure_collection_exists_creates_named_vectors_schema(vector_repo, qdrant_memory_client):
    info = qdrant_memory_client.get_collection(vector_repo.collection_name)
    assert DENSE_VECTOR_NAME in info.config.params.vectors
    assert SPARSE_VECTOR_NAME in info.config.params.sparse_vectors


def test_ensure_collection_exists_is_idempotent(vector_repo):
    # No debe lanzar ni recrear la colección si ya existe.
    vector_repo.ensure_collection_exists()
    stats = vector_repo.get_collection_stats()
    assert stats["points_count"] == 0


def test_upsert_chunks_raises_on_length_mismatch(vector_repo, sample_chunks, fake_dense_vectors):
    dense = [fake_dense_vectors[c["chunk_id"]] for c in sample_chunks]
    with pytest.raises(ValueError):
        vector_repo.upsert_chunks("doc_a", sample_chunks, dense, sparse_embeddings=[])


def test_upsert_chunks_is_idempotent(
    vector_repo, sample_chunks, fake_dense_vectors, fake_sparse_vectors
):
    upsert(vector_repo, "doc_a", sample_chunks, fake_dense_vectors, fake_sparse_vectors)
    upsert(vector_repo, "doc_a", sample_chunks, fake_dense_vectors, fake_sparse_vectors)

    stats = vector_repo.get_collection_stats()
    assert stats["points_count"] == 3


def test_upsert_chunks_same_chunk_id_different_documents_dont_collide(
    vector_repo, sample_chunks, fake_dense_vectors, fake_sparse_vectors
):
    upsert(vector_repo, "doc_a", sample_chunks, fake_dense_vectors, fake_sparse_vectors)
    upsert(vector_repo, "doc_b", sample_chunks, fake_dense_vectors, fake_sparse_vectors)

    stats = vector_repo.get_collection_stats()
    assert stats["points_count"] == 6
    assert vector_repo.list_documents() == ["doc_a", "doc_b"]


def test_search_returns_expected_top_result_from_fused_signals(
    vector_repo, sample_chunks, fake_dense_vectors, fake_sparse_vectors
):
    upsert(vector_repo, "doc_a", sample_chunks, fake_dense_vectors, fake_sparse_vectors)

    results = vector_repo.search(
        dense_query_vector=fake_dense_vectors["chunk_0001"],
        sparse_query_vector=fake_sparse_vectors["chunk_0001"],
        limit=3,
    )

    assert results[0]["chunk_id"] == "chunk_0001"


def test_search_document_filter_isolates_correct_document(
    vector_repo, sample_chunks, fake_dense_vectors, fake_sparse_vectors
):
    upsert(vector_repo, "doc_a", sample_chunks, fake_dense_vectors, fake_sparse_vectors)
    upsert(vector_repo, "doc_b", sample_chunks, fake_dense_vectors, fake_sparse_vectors)

    results = vector_repo.search(
        dense_query_vector=BROAD_DENSE_QUERY,
        sparse_query_vector=BROAD_SPARSE_QUERY,
        limit=10,
        document_filter="doc_a",
    )

    assert len(results) == 3
    assert all(r["document_name"] == "doc_a" for r in results)


def test_search_page_filter_excludes_other_pages(
    vector_repo, sample_chunks, fake_dense_vectors, fake_sparse_vectors
):
    upsert(vector_repo, "doc_a", sample_chunks, fake_dense_vectors, fake_sparse_vectors)

    results = vector_repo.search(
        dense_query_vector=BROAD_DENSE_QUERY,
        sparse_query_vector=BROAD_SPARSE_QUERY,
        limit=10,
        page_filter=2,
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_0002"
    assert 2 in results[0]["pages"]


def test_search_heading_contains_filters_correctly(
    vector_repo, sample_chunks, fake_dense_vectors, fake_sparse_vectors
):
    upsert(vector_repo, "doc_a", sample_chunks, fake_dense_vectors, fake_sparse_vectors)

    results = vector_repo.search(
        dense_query_vector=BROAD_DENSE_QUERY,
        sparse_query_vector=BROAD_SPARSE_QUERY,
        limit=10,
        heading_contains="Results",
    )

    assert len(results) == 1
    assert results[0]["chunk_id"] == "chunk_0003"


def test_delete_document_removes_only_that_documents_points(
    vector_repo, sample_chunks, fake_dense_vectors, fake_sparse_vectors
):
    upsert(vector_repo, "doc_a", sample_chunks, fake_dense_vectors, fake_sparse_vectors)
    upsert(vector_repo, "doc_b", sample_chunks, fake_dense_vectors, fake_sparse_vectors)

    deleted = vector_repo.delete_document("doc_a")

    assert deleted == 3
    assert vector_repo.get_collection_stats()["points_count"] == 3
    assert vector_repo.list_documents() == ["doc_b"]
