"""
Tests de GeminiProvider con google.genai.Client mockeado (sin red, sin API
key real) — solo valida que set_api_key reconstruye el cliente con la key
nueva, no el comportamiento de la API de Gemini en sí.
"""

from unittest.mock import MagicMock, patch

from rag_app.providers.llm import GeminiProvider


def test_set_api_key_rebuilds_client_with_new_key():
    with patch("rag_app.providers.llm.genai.Client") as mock_client_cls:
        mock_client_cls.side_effect = lambda **kwargs: MagicMock()

        provider = GeminiProvider(api_key="key-original")
        original_client = provider.client
        mock_client_cls.assert_called_once_with(api_key="key-original")

        provider.set_api_key("key-nueva")

        mock_client_cls.assert_called_with(api_key="key-nueva")
        assert provider.client is not original_client
