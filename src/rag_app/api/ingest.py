"""Router de ingesta: POST /documents/upload, progreso por etapas (SSE)."""

import logging
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import StreamingResponse

from rag_app.api.chat import get_chat_service
from rag_app.api.documents import get_document_metadata_service
from rag_app.core.config import Settings, get_settings
from rag_app.core.sse import format_sse
from rag_app.services.chat_service import ChatService
from rag_app.services.docling_chunker import DoclingChunker
from rag_app.services.docling_extractor import DoclingExtractor
from rag_app.services.ingest_service import IngestService

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache
def get_ingest_service() -> IngestService:
    """Reusa el IndexingService de get_chat_service(), solo levanta el
    extractor/chunker de Docling (misma configuración que la CLI)."""
    settings: Settings = get_settings()
    cache_dir = Path(settings.cache_dir)

    extractor = DoclingExtractor(
        cache_dir=cache_dir,
        enable_ocr=False,
        enable_table_structure=True,
        generate_picture_images=True,
        generate_page_images=False,
    )
    chunker = DoclingChunker(
        cache_dir=cache_dir,
        embedding_model_name=settings.embedding_model_name,
    )

    chat_service: ChatService = get_chat_service()

    return IngestService(
        extractor=extractor,
        chunker=chunker,
        indexing_service=chat_service.indexing_service,
        document_metadata_service=get_document_metadata_service(),
        uploads_dir=Path(settings.uploads_dir),
        cache_dir=cache_dir,
    )


def _events(ingest_service: IngestService, pdf_bytes: bytes, filename: str) -> Iterator[dict]:
    yield from ingest_service.ingest_stream(pdf_bytes, filename)


@router.post("/documents/upload")
async def upload_document(
    file: UploadFile,
    ingest_service: IngestService = Depends(get_ingest_service),
) -> StreamingResponse:
    pdf_bytes = await file.read()
    filename = file.filename or "documento.pdf"
    return StreamingResponse(
        format_sse(_events(ingest_service, pdf_bytes, filename)),
        media_type="text/event-stream",
    )
