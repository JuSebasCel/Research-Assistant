"""
Test de integración end-to-end del chat: retrieval real + Gemini real.

Se salta automáticamente si no hay GEMINI_API_KEY configurada (gratis, sin
tarjeta, vía https://aistudio.google.com/apikey) - no requiere red ni key
para el resto de la suite.
"""

import pytest

from rag_app.core.config import get_settings
from rag_app.providers.llm import GeminiProvider
from rag_app.services.chat_service import ChatService

settings = get_settings()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not settings.gemini_api_key,
        reason="GEMINI_API_KEY no configurada - consigue una gratis en https://aistudio.google.com/apikey",
    ),
]


@pytest.fixture
def real_llm_provider():
    return GeminiProvider(api_key=settings.gemini_api_key, model_name=settings.gemini_model_name)


def test_chat_answers_with_citations_when_content_is_relevant(
    real_llm_provider, real_indexing_service, sample_chunks_json, tmp_path
):
    real_indexing_service.index_document(sample_chunks_json, "doc_a")
    service = ChatService(
        real_indexing_service, real_llm_provider, tmp_path, min_score_threshold=0.1
    )

    events = list(service.answer_stream("What is the BLEU score mentioned in the text?"))

    chunk_events = [e for e in events if e["type"] == "chunk"]
    done_event = events[-1]

    assert chunk_events, "esperaba al menos un chunk de respuesta del LLM"
    assert done_event["type"] == "done"
    assert done_event["citations"], "esperaba al menos una cita"


def test_chat_is_honest_about_unrelated_question(
    real_llm_provider, real_indexing_service, sample_chunks_json, tmp_path
):
    """
    Con una colección de solo 3 chunks, el umbral de score por sí solo no
    es un filtro confiable (hasta el resultado más irrelevante puede sacar
    un score RRF decente por pura falta de competencia) — la red de
    seguridad real es que el LLM, instruido a no inventar, reconozca que
    la pregunta no tiene respuesta en los fragmentos dados. Validamos esa
    honestidad, sin importar cuál de las dos capas la produjo.
    """
    real_indexing_service.index_document(sample_chunks_json, "doc_a")
    service = ChatService(
        real_indexing_service, real_llm_provider, tmp_path, min_score_threshold=0.5
    )

    events = list(service.answer_stream("What is the capital of France?"))

    if events == [{"type": "no_results"}]:
        return  # cortado por el umbral de score: también válido

    full_answer = "".join(e["text"] for e in events if e["type"] == "chunk").lower()
    honest_markers = ("no encontr", "no está", "no se menciona")
    assert any(marker in full_answer for marker in honest_markers), (
        f"Esperaba que el LLM admitiera no tener la respuesta, pero dijo: {full_answer!r}"
    )
