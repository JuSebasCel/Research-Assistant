"""Schemas Pydantic para el endpoint de chat."""

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Pregunta del usuario")
    top_k: int = Field(default=5, ge=1, le=20, description="Número de chunks a recuperar")
    document_filter: str | None = Field(default=None, description="Restringir a un documento")
    page_filter: int | None = Field(default=None, description="Restringir a una página")
    heading_contains: str | None = Field(
        default=None, description="Restringir por texto libre en headings"
    )
