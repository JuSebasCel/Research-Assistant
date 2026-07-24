"""
Tests de ChatService: umbral de confianza, construcción del prompt, y
forma de los eventos de streaming — todo con GeminiProvider mockeado
(sin red, sin API key).
"""

from unittest.mock import MagicMock

import pytest
from google.genai import errors as genai_errors

from rag_app.services.chat_service import ChatService


@pytest.fixture
def fake_llm_provider():
    provider = MagicMock()
    provider.generate_stream.side_effect = lambda system, prompt, images=None: iter(
        ["Hola, ", "mundo."]
    )
    return provider


@pytest.fixture
def chat_service(indexing_service, fake_llm_provider, tmp_path):
    return ChatService(indexing_service, fake_llm_provider, tmp_path, min_score_threshold=0.0)


def test_no_indexed_documents_yields_no_results_without_calling_llm(
    chat_service, fake_llm_provider
):
    events = list(chat_service.answer_stream("cualquier pregunta"))

    assert events == [{"type": "no_results"}]
    fake_llm_provider.generate_stream.assert_not_called()


def test_score_below_threshold_yields_no_results_without_calling_llm(
    indexing_service, fake_llm_provider, sample_chunks_json, tmp_path
):
    indexing_service.index_document(sample_chunks_json, "doc_a")
    # Umbral imposible de superar -> siempre debe cortar antes del LLM,
    # sin importar el score real que devuelva el retrieval.
    service = ChatService(indexing_service, fake_llm_provider, tmp_path, min_score_threshold=999.0)

    events = list(service.answer_stream("The Transformer architecture"))

    assert events == [{"type": "no_results"}]
    fake_llm_provider.generate_stream.assert_not_called()


def test_answer_stream_yields_chunks_then_done_with_citations(
    indexing_service, fake_llm_provider, sample_chunks_json, tmp_path
):
    indexing_service.index_document(sample_chunks_json, "doc_a")
    service = ChatService(indexing_service, fake_llm_provider, tmp_path, min_score_threshold=0.0)

    events = list(service.answer_stream("The Transformer architecture"))

    assert events[0] == {"type": "chunk", "text": "Hola, "}
    assert events[1] == {"type": "chunk", "text": "mundo."}

    done = events[-1]
    assert done["type"] == "done"
    assert len(done["citations"]) == 3  # los 3 sample_chunks quedaron indexados
    for citation in done["citations"]:
        assert set(citation.keys()) == {"document_name", "chunk_id", "pages", "image_urls"}
        assert citation["document_name"] == "doc_a"
        assert citation["image_urls"] == []  # sample_chunks no traen imágenes

    fake_llm_provider.generate_stream.assert_called_once()


def test_quota_exceeded_yields_clear_error_event(
    indexing_service, sample_chunks_json, tmp_path
):
    provider = MagicMock()

    def raise_quota_error(system, prompt, images=None):
        raise genai_errors.ClientError(
            429, {"error": {"message": "quota stuff", "status": "RESOURCE_EXHAUSTED"}}
        )
        yield  # pragma: no cover - hace de esto un generador, nunca se llega acá

    provider.generate_stream.side_effect = raise_quota_error
    indexing_service.index_document(sample_chunks_json, "doc_a")
    service = ChatService(indexing_service, provider, tmp_path, min_score_threshold=0.0)

    events = list(service.answer_stream("The Transformer architecture"))

    assert events == [
        {
            "type": "error",
            "error": "Se acabó la cuota gratuita de Gemini por hoy. Intenta más tarde.",
        }
    ]


def test_other_gemini_error_yields_generic_error_event(
    indexing_service, sample_chunks_json, tmp_path
):
    provider = MagicMock()

    def raise_server_error(system, prompt, images=None):
        raise genai_errors.ServerError(500, {"error": {"message": "boom", "status": "INTERNAL"}})
        yield  # pragma: no cover

    provider.generate_stream.side_effect = raise_server_error
    indexing_service.index_document(sample_chunks_json, "doc_a")
    service = ChatService(indexing_service, provider, tmp_path, min_score_threshold=0.0)

    events = list(service.answer_stream("The Transformer architecture"))

    assert len(events) == 1
    assert events[0]["type"] == "error"


def test_build_prompt_includes_document_name_pages_and_question(tmp_path):
    service = ChatService(
        indexing_service=MagicMock(), llm_provider=MagicMock(), cache_dir=tmp_path
    )
    results = [
        {
            "document_name": "attallyouneed",
            "chunk_id": "chunk_0009",
            "content": "Scaled dot-product attention computes...",
            "pages": [4],
        }
    ]

    prompt = service._build_prompt(
        "What is scaled dot-product attention?", results, ["attallyouneed", "otropaper"]
    )

    assert "attallyouneed" in prompt
    assert "página(s): 4" in prompt
    assert "Scaled dot-product attention computes..." in prompt
    assert "What is scaled dot-product attention?" in prompt


def test_build_prompt_lists_all_documents_even_if_not_in_results(tmp_path):
    service = ChatService(
        indexing_service=MagicMock(), llm_provider=MagicMock(), cache_dir=tmp_path
    )
    results = [
        {
            "document_name": "doc_a",
            "chunk_id": "chunk_0001",
            "content": "algo relevante",
            "pages": [1],
        }
    ]

    # doc_b y doc_c no aparecen en los resultados del retrieval, pero
    # igual deben quedar listados como parte del catálogo disponible.
    prompt = service._build_prompt("pregunta", results, ["doc_a", "doc_b", "doc_c"])

    assert "Documentos disponibles en el sistema: doc_a, doc_b, doc_c" in prompt


def test_answer_stream_prompt_includes_full_document_catalog(
    indexing_service, fake_llm_provider, sample_chunks_json, tmp_path
):
    indexing_service.index_document(sample_chunks_json, "doc_a")
    indexing_service.index_document(sample_chunks_json, "doc_b")
    service = ChatService(indexing_service, fake_llm_provider, tmp_path, min_score_threshold=0.0)

    list(service.answer_stream("The Transformer architecture", document_filter="doc_a"))

    prompt_sent = fake_llm_provider.generate_stream.call_args.args[1]
    assert "doc_a" in prompt_sent
    assert "doc_b" in prompt_sent


def test_load_images_dedupes_caps_and_skips_missing_files(tmp_path):
    doc_dir = tmp_path / "doc_a"
    doc_dir.mkdir()
    for i in range(5):
        (doc_dir / f"fig_{i}.png").write_bytes(b"fake-png-bytes")

    service = ChatService(
        indexing_service=MagicMock(), llm_provider=MagicMock(), cache_dir=tmp_path
    )

    results = [
        {
            "document_name": "doc_a",
            # "missing.png" no existe en disco -> debe saltarse sin contar
            # para el tope; fig_0 luego se repite en el siguiente chunk.
            "image_paths": ["missing.png", "fig_0.png"],
        },
        {
            "document_name": "doc_a",
            "image_paths": ["fig_0.png", "fig_1.png", "fig_2.png", "fig_3.png", "fig_4.png"],
        },
    ]

    images = service._load_images(results)

    # 4 = MAX_IMAGES_PER_ANSWER: fig_0..fig_3 entran, fig_4 se queda afuera.
    assert len(images) == 4
    for data, mime_type in images:
        assert data == b"fake-png-bytes"
        assert mime_type == "image/png"
