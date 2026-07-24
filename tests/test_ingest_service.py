"""
Tests de IngestService con extractor/chunker de Docling mockeados (sin PDFs
reales, sin cargar modelos) pero IndexingService real (fixture compartida
con Qdrant en memoria) — valida la orquestación de punta a punta y la
secuencia de eventos de progreso, no el parsing de PDF en sí.
"""

import json
from unittest.mock import MagicMock

import pytest
from docling_core.types.doc import DocItemLabel

from rag_app.services.document_metadata_service import DocumentMetadataService
from rag_app.services.ingest_service import IngestService


def _fake_doc_item(label, text):
    item = MagicMock()
    item.label = label
    item.text = text
    return item


@pytest.fixture
def fake_extractor():
    extractor = MagicMock()
    doc = MagicMock()
    doc.texts = []
    extractor.extract.return_value = doc
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
def document_metadata_service(tmp_path) -> DocumentMetadataService:
    return DocumentMetadataService(tmp_path / "document_metadata.json")


@pytest.fixture
def ingest_service(
    fake_extractor, fake_chunker, indexing_service, document_metadata_service, tmp_path
):
    return IngestService(
        extractor=fake_extractor,
        chunker=fake_chunker,
        indexing_service=indexing_service,
        document_metadata_service=document_metadata_service,
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
    fake_chunker, indexing_service, document_metadata_service, tmp_path
):
    extractor = MagicMock()
    extractor.extract.side_effect = RuntimeError("docling boom")

    service = IngestService(
        extractor=extractor,
        chunker=fake_chunker,
        indexing_service=indexing_service,
        document_metadata_service=document_metadata_service,
        uploads_dir=tmp_path / "uploads",
        cache_dir=tmp_path / "cache",
    )

    events = list(service.ingest_stream(b"%PDF-fake-bytes", "paper.pdf"))

    assert events[-1]["type"] == "error"
    assert "docling boom" in events[-1]["error"]


def test_ingest_stream_sets_display_name_from_extracted_title(
    fake_chunker, indexing_service, document_metadata_service, tmp_path
):
    extractor = MagicMock()
    doc = MagicMock()
    doc.texts = [
        _fake_doc_item(DocItemLabel.PAGE_HEADER, "Running header"),
        _fake_doc_item(DocItemLabel.TITLE, "Attention Is All You Need"),
        _fake_doc_item(DocItemLabel.SECTION_HEADER, "Introduction"),
    ]
    extractor.extract.return_value = doc
    extractor.get_figures_index.return_value = {}

    service = IngestService(
        extractor=extractor,
        chunker=fake_chunker,
        indexing_service=indexing_service,
        document_metadata_service=document_metadata_service,
        uploads_dir=tmp_path / "uploads",
        cache_dir=tmp_path / "cache",
    )

    list(service.ingest_stream(b"%PDF-fake-bytes", "paper.pdf"))

    assert document_metadata_service.get("paper")["display_name"] == "Attention Is All You Need"


def test_ingest_stream_falls_back_to_section_header_when_no_title(
    fake_chunker, indexing_service, document_metadata_service, tmp_path
):
    extractor = MagicMock()
    doc = MagicMock()
    doc.texts = [_fake_doc_item(DocItemLabel.SECTION_HEADER, "Abstract")]
    extractor.extract.return_value = doc
    extractor.get_figures_index.return_value = {}

    service = IngestService(
        extractor=extractor,
        chunker=fake_chunker,
        indexing_service=indexing_service,
        document_metadata_service=document_metadata_service,
        uploads_dir=tmp_path / "uploads",
        cache_dir=tmp_path / "cache",
    )

    list(service.ingest_stream(b"%PDF-fake-bytes", "paper.pdf"))

    assert document_metadata_service.get("paper")["display_name"] == "Abstract"


def test_ingest_stream_does_not_overwrite_manual_display_name(
    fake_extractor, fake_chunker, indexing_service, document_metadata_service, tmp_path
):
    document_metadata_service.set_display_name("paper", "Nombre puesto a mano")
    fake_extractor.extract.return_value.texts = [
        _fake_doc_item(DocItemLabel.TITLE, "Título extraído del PDF")
    ]

    service = IngestService(
        extractor=fake_extractor,
        chunker=fake_chunker,
        indexing_service=indexing_service,
        document_metadata_service=document_metadata_service,
        uploads_dir=tmp_path / "uploads",
        cache_dir=tmp_path / "cache",
    )

    list(service.ingest_stream(b"%PDF-fake-bytes", "paper.pdf"))

    assert document_metadata_service.get("paper")["display_name"] == "Nombre puesto a mano"
