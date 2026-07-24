"""Router de documentos: listado (con metadata) y edición de nombre/carpeta."""

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends

from rag_app.api.chat import get_indexing_service
from rag_app.core.config import Settings, get_settings
from rag_app.models.documents import DocumentInfo, DocumentMetadataUpdate
from rag_app.services.document_metadata_service import DocumentMetadataService
from rag_app.services.indexing_service import IndexingService

router = APIRouter()


@lru_cache
def get_document_metadata_service() -> DocumentMetadataService:
    settings: Settings = get_settings()
    return DocumentMetadataService(Path(settings.document_metadata_path))


@router.get("/documents")
async def list_documents(
    indexing_service: IndexingService = Depends(get_indexing_service),
    metadata_service: DocumentMetadataService = Depends(get_document_metadata_service),
) -> list[DocumentInfo]:
    document_names = indexing_service.list_indexed_documents()
    metadata = metadata_service.get_all()
    return [
        DocumentInfo(
            document_name=name,
            display_name=metadata.get(name, {}).get("display_name", name),
            folder=metadata.get(name, {}).get("folder"),
        )
        for name in document_names
    ]


@router.patch("/documents/{document_name}")
async def update_document(
    document_name: str,
    update: DocumentMetadataUpdate,
    metadata_service: DocumentMetadataService = Depends(get_document_metadata_service),
) -> DocumentInfo:
    changes = update.model_dump(exclude_unset=True)

    if "display_name" in changes and changes["display_name"] is not None:
        metadata_service.set_display_name(document_name, changes["display_name"])
    if "folder" in changes:
        metadata_service.set_folder(document_name, changes["folder"])

    meta = metadata_service.get(document_name)
    return DocumentInfo(
        document_name=document_name,
        display_name=meta.get("display_name", document_name),
        folder=meta.get("folder"),
    )
