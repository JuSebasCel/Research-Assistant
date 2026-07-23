"""
Test de integración end-to-end: modelos reales (e5-large + BM25/fastembed)
+ Qdrant en memoria, sin mocks.

Los bugs reales que encontramos durante el desarrollo (client.search()
removido, CollectionInfo.vectors_count removido) fueron de compatibilidad
con la API de qdrant-client, no de nuestra lógica — un test con mocks no
los hubiera detectado. Este test replica a escala pequeña la validación
manual que hicimos contra la colección real.

Marcado con @pytest.mark.integration porque carga modelos reales (más
lento que el resto de la suite, ~10-15s de carga de modelos la primera vez).
"""

import pytest

pytestmark = pytest.mark.integration

# real_indexing_service (+ real_embedding_provider / real_sparse_embedding_provider)
# vive en conftest.py para poder reusarse en test_chat_integration.py.


def test_hybrid_search_end_to_end_with_real_models(real_indexing_service, sample_chunks_json):
    result = real_indexing_service.index_document(sample_chunks_json, "doc_a")
    assert result["chunks_indexed"] == 3

    # Reindexar el mismo documento no debe duplicar puntos.
    real_indexing_service.index_document(sample_chunks_json, "doc_a")
    assert real_indexing_service.get_stats()["points_count"] == 3

    # "BLEU" solo aparece en chunk_0003 ("Results"): valida que la señal
    # léxica (BM25) realmente participa en el resultado final, tal como
    # confirmamos manualmente contra la colección real con "ESCRT".
    results = real_indexing_service.search("BLEU", top_k=1)
    assert results[0]["chunk_id"] == "chunk_0003"

    # Filtro de página end-to-end con vectores reales.
    page_results = real_indexing_service.search("model", top_k=5, page_filter=1)
    assert len(page_results) >= 1
    assert all(1 in r["pages"] for r in page_results)
