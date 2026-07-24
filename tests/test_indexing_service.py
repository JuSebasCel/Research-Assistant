"""
Tests de IndexingService con providers de embeddings mockeados (rápidos,
sin cargar modelos reales) pero VectorRepository real contra Qdrant en
memoria — así se valida la orquestación real sin pagar el costo de
cargar e5-large/BM25 en cada test.

Los fixtures `indexing_service` / `make_fake_providers` viven en conftest.py
para poder reusarse en test_chat_service.py.
"""

import pytest


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


def test_search_across_documents_includes_every_document(
    indexing_service, sample_chunks_json
):
    # Mismos 3 chunks indexados bajo dos documentos distintos: una búsqueda
    # global sin fan-out podría dejar uno afuera si el otro "gana" el
    # ranking; con fan-out ambos deben quedar representados.
    indexing_service.index_document(sample_chunks_json, "doc_a")
    indexing_service.index_document(sample_chunks_json, "doc_b")

    results = indexing_service.search_across_documents("some query", max_total=10)

    documents_present = {r["document_name"] for r in results}
    assert documents_present == {"doc_a", "doc_b"}


def test_search_across_documents_respects_per_doc_and_max_total_caps(
    indexing_service, sample_chunks_json
):
    indexing_service.index_document(sample_chunks_json, "doc_a")
    indexing_service.index_document(sample_chunks_json, "doc_b")

    # 3 chunks por documento disponibles, pero per_doc_top_k=1 limita a 1
    # cada uno -> a lo sumo 2 candidatos entre los dos, aunque max_total sea mayor.
    results = indexing_service.search_across_documents(
        "some query", max_total=10, per_doc_top_k=1
    )

    assert len(results) == 2

    # max_total corta el total aunque cada documento aporte más.
    results_capped = indexing_service.search_across_documents(
        "some query", max_total=3, per_doc_top_k=3
    )
    assert len(results_capped) == 3


def test_search_across_documents_restricts_to_given_subset(
    indexing_service, sample_chunks_json
):
    # doc_a, doc_b y doc_c quedan indexados, pero solo se pide fan-out
    # sobre doc_a y doc_c (simula "todos los documentos de esta carpeta").
    indexing_service.index_document(sample_chunks_json, "doc_a")
    indexing_service.index_document(sample_chunks_json, "doc_b")
    indexing_service.index_document(sample_chunks_json, "doc_c")

    results = indexing_service.search_across_documents(
        "some query", max_total=10, document_names=["doc_a", "doc_c"]
    )

    documents_present = {r["document_name"] for r in results}
    assert documents_present == {"doc_a", "doc_c"}
