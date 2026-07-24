"""Tests de AppSettingsService: round-trip JSON de la API key de Gemini."""

import pytest

from rag_app.services.app_settings_service import AppSettingsService


@pytest.fixture
def service(tmp_path) -> AppSettingsService:
    return AppSettingsService(tmp_path / "app_settings.json")


def test_get_gemini_api_key_returns_none_when_unset(service):
    assert service.get_gemini_api_key() is None


def test_set_and_get_gemini_api_key_round_trips(service):
    service.set_gemini_api_key("sk-real-key-123")

    assert service.get_gemini_api_key() == "sk-real-key-123"


def test_set_gemini_api_key_overwrites_previous(service):
    service.set_gemini_api_key("old-key")
    service.set_gemini_api_key("new-key")

    assert service.get_gemini_api_key() == "new-key"


def test_clear_gemini_api_key_removes_it(service):
    service.set_gemini_api_key("sk-real-key-123")
    service.clear_gemini_api_key()

    assert service.get_gemini_api_key() is None


def test_clear_gemini_api_key_is_safe_when_never_set(service):
    service.clear_gemini_api_key()

    assert service.get_gemini_api_key() is None


def test_new_instance_reads_persisted_key_from_disk(tmp_path):
    path = tmp_path / "app_settings.json"
    AppSettingsService(path).set_gemini_api_key("persisted-key")

    reloaded = AppSettingsService(path)

    assert reloaded.get_gemini_api_key() == "persisted-key"
