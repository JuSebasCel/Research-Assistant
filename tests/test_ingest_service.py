"""
Tests de IngestService con extractor/chunker de Docling mockeados (sin PDFs
reales, sin cargar modelos) pero IndexingService real (fixture compartida
con Qdrant en memoria) — valida la orquestación de punta a punta y la
secuencia de eventos de progreso, no el parsing de PDF en sí.
"""

import json
from unittest.mock import MagicMock

import pytest

from rag_app.services.ingest_service import IngestService


@pytest.fixture
def fake_extractor():
    extractor = MagicMock()
    extractor.extract.return_value = "fake-docling-document"
    extractor.get_figures_index.return_value = {}
    return extractor


@pytest.fixture
def fake_chunker(tmp_path, sample_chunks):
    """Simula DoclingChunker.chunk_document escribiendo chunks.json en el
    mismo path que index_document espera leer (cache_dir/doc/chunks/)."""
    chunker = MagicMock()

    def _chunk_document(doc, document_name, figures_index=None, force_rechunk=False):
        chunks_dir = tmp_path / "cache" / document_name / "chunks"
        chunks_dir.mkdir(parents=True, exist_ok=True)
        (chunks_dir / "chunks.json").write_text(json.dumps(sample_chunks), encoding="utf-8")
        return sample_chunks

    chunker.chunk_document.side_effect = _chunk_document
    return chunker


@pytest.fixture
def ingest_service(fake_extractor, fake_chunker, indexing_service, tmp_path):
    return IngestService(
        extractor=fake_extractor,
        chunker=fake_chunker,
        indexing_service=indexing_service,
        uploads_dir=tmp_path / "uploads",
        cache_dir=tmp_path / "cache",
    )


def test_ingest_stream_yields_stages_then_done(ingest_service):
    events = list(ingest_service.ingest_stream(b"%PDF-fake-bytes", "My Paper.pdf"))

    stages = [e["stage"] for e in events if e["type"] == "stage"]
    assert stages == ["uploading", "extracting", "chunking", "indexing"]

    done = events[-1]
    assert done == {"type": "done", "document_name": "My Paper", "chunks_indexed": 3}


def test_ingest_stream_saves_pdf_with_sanitized_filename(ingest_service, tmp_path):
    # "../../evil.pdf" no debe escapar uploads_dir: es un nombre de archivo
    # que llega crudo desde un upload HTTP, no es de confianza.
    list(ingest_service.ingest_stream(b"%PDF-fake-bytes", "../../evil.pdf"))

    saved = tmp_path / "uploads" / "evil.pdf"
    assert saved.exists()
    assert saved.read_bytes() == b"%PDF-fake-bytes"


def test_ingest_stream_yields_error_event_on_extraction_failure(
    fake_chunker, indexing_service, tmp_path
):
    extractor = MagicMock()
    extractor.extract.side_effect = RuntimeError("docling boom")

    service = IngestService(
        extractor=extractor,
        chunker=fake_chunker,
        indexing_service=indexing_service,
        uploads_dir=tmp_path / "uploads",
        cache_dir=tmp_path / "cache",
    )

    events = list(service.ingest_stream(b"%PDF-fake-bytes", "paper.pdf"))

    assert events[-1]["type"] == "error"
    assert "docling boom" in events[-1]["error"]
