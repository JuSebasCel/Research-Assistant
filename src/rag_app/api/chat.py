"""Router de chat: POST /chat, respuesta en streaming (SSE)."""

import json
import logging
from collections.abc import Iterator
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse

from rag_app.core.config import Settings, get_settings
from rag_app.models.chat import ChatRequest
from rag_app.providers.embeddings import EmbeddingProvider
from rag_app.providers.llm import GeminiProvider
from rag_app.providers.qdrant_client import create_qdrant_client
from rag_app.providers.sparse_embeddings import SparseEmbeddingProvider
from rag_app.repositories.vector_repository import VectorRepository
from rag_app.services.chat_service import ChatService
from rag_app.services.indexing_service import IndexingService

logger = logging.getLogger(__name__)

router = APIRouter()


@lru_cache
def get_chat_service() -> ChatService:
    """
    Construye el grafo de servicios una sola vez por proceso (carga el
    modelo de embeddings, que es costosa) y lo reutiliza en cada request,
    igual que el patrón de create_services() en run_indexing.py.
    """
    settings: Settings = get_settings()

    embedding_provider = EmbeddingProvider(model_name=settings.embedding_model_name)
    sparse_embedding_provider = SparseEmbeddingProvider()

    qdrant_client = create_qdrant_client(settings)
    vector_repository = VectorRepository(
        client=qdrant_client,
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.embedding_dimension,
    )
    vector_repository.ensure_collection_exists()

    indexing_service = IndexingService(
        embedding_provider=embedding_provider,
        sparse_embedding_provider=sparse_embedding_provider,
        vector_repository=vector_repository,
    )

    llm_provider = GeminiProvider(
        api_key=settings.gemini_api_key,
        model_name=settings.gemini_model_name,
    )

    return ChatService(
        indexing_service=indexing_service,
        llm_provider=llm_provider,
        cache_dir=Path(settings.cache_dir),
    )


def get_indexing_service(
    chat_service: ChatService = Depends(get_chat_service),
) -> IndexingService:
    """Reusa el IndexingService ya construido dentro de ChatService (evita
    levantar el grafo de servicios dos veces)."""
    return chat_service.indexing_service


def _sse_format(chat_service: ChatService, request: ChatRequest) -> Iterator[str]:
    for event in chat_service.answer_stream(
        query=request.query,
        top_k=request.top_k,
        document_filter=request.document_filter,
        page_filter=request.page_filter,
        heading_contains=request.heading_contains,
    ):
        yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@router.post("/chat")
async def chat(
    request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> StreamingResponse:
    return StreamingResponse(
        _sse_format(chat_service, request),
        media_type="text/event-stream",
    )
