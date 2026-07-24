"""Schemas Pydantic para el endpoint de documentos."""

from pydantic import BaseModel


class DocumentInfo(BaseModel):
    document_name: str
    display_name: str
    folder: str | None = None


class DocumentMetadataUpdate(BaseModel):
    display_name: str | None = None
    folder: str | None = None
