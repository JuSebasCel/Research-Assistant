"""
Configuración editable en runtime desde la interfaz (hoy: una API key de
Gemini propia del usuario en vez de la del .env del servidor). Mismo patrón
que DocumentMetadataService, pero este archivo sí contiene un secreto real
y está en .gitignore.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class AppSettingsService:
    def __init__(self, settings_path: Path):
        self.settings_path = settings_path
        self.settings_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, Any]:
        if not self.settings_path.exists():
            return {}
        try:
            return json.loads(self.settings_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(f"Configuración corrupta, se ignora: {self.settings_path}")
            return {}

    def _save(self, data: dict[str, Any]) -> None:
        self.settings_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get_gemini_api_key(self) -> str | None:
        return self._load().get("gemini_api_key")

    def set_gemini_api_key(self, api_key: str) -> None:
        data = self._load()
        data["gemini_api_key"] = api_key
        self._save(data)

    def clear_gemini_api_key(self) -> None:
        data = self._load()
        data.pop("gemini_api_key", None)
        self._save(data)
