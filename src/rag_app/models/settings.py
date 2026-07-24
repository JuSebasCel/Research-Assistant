"""Schemas Pydantic para el endpoint de configuración."""

from pydantic import BaseModel


class GeminiKeyStatus(BaseModel):
    has_custom_key: bool
    key_hint: str | None = None


class GeminiKeyUpdate(BaseModel):
    api_key: str
