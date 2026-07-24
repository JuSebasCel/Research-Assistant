"""Router de configuración: API key de Gemini que cada usuario puede traer."""

from fastapi import APIRouter, Depends, HTTPException

from rag_app.api.chat import get_app_settings_service, get_chat_service
from rag_app.core.config import Settings, get_settings
from rag_app.models.settings import GeminiKeyStatus, GeminiKeyUpdate
from rag_app.services.app_settings_service import AppSettingsService
from rag_app.services.chat_service import ChatService

router = APIRouter()


def _status(app_settings_service: AppSettingsService) -> GeminiKeyStatus:
    custom_key = app_settings_service.get_gemini_api_key()
    if not custom_key:
        return GeminiKeyStatus(has_custom_key=False)
    # Solo se expone el sufijo — nunca la key completa de vuelta al cliente.
    return GeminiKeyStatus(has_custom_key=True, key_hint=custom_key[-4:])


@router.get("/settings/gemini-key")
async def get_gemini_key_status(
    app_settings_service: AppSettingsService = Depends(get_app_settings_service),
) -> GeminiKeyStatus:
    return _status(app_settings_service)


@router.patch("/settings/gemini-key")
async def set_gemini_key(
    update: GeminiKeyUpdate,
    app_settings_service: AppSettingsService = Depends(get_app_settings_service),
    chat_service: ChatService = Depends(get_chat_service),
) -> GeminiKeyStatus:
    api_key = update.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="La API key no puede estar vacía")

    app_settings_service.set_gemini_api_key(api_key)
    chat_service.llm_provider.set_api_key(api_key)
    return _status(app_settings_service)


@router.delete("/settings/gemini-key")
async def clear_gemini_key(
    app_settings_service: AppSettingsService = Depends(get_app_settings_service),
    chat_service: ChatService = Depends(get_chat_service),
) -> GeminiKeyStatus:
    app_settings_service.clear_gemini_api_key()
    settings: Settings = get_settings()
    chat_service.llm_provider.set_api_key(settings.gemini_api_key)
    return _status(app_settings_service)
