"""Router de documentos: GET /documents (lista para el sidebar del frontend)."""

from fastapi import APIRouter, Depends

from rag_app.api.chat import get_indexing_service
from rag_app.services.indexing_service import IndexingService

router = APIRouter()


@router.get("/documents")
async def list_documents(
    indexing_service: IndexingService = Depends(get_indexing_service),
) -> list[str]:
    return indexing_service.list_indexed_documents()
