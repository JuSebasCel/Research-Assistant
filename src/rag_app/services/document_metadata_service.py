"""
Nombre visible y carpeta por documento, por fuera de Qdrant — document_name
sigue siendo la clave real de filtrado en el vector store, esto es solo
presentación/organización encima. JSON simple en vez de una base de datos
relacional: el volumen de datos no justifica más.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DocumentMetadataService:
    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)

    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.metadata_path.exists():
            return {}
        try:
            return json.loads(self.metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning(f"Metadata de documentos corrupta, se ignora: {self.metadata_path}")
            return {}

    def _save(self, data: dict[str, dict[str, Any]]) -> None:
        self.metadata_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def get(self, document_name: str) -> dict[str, Any]:
        return self._load().get(document_name, {})

    def get_all(self) -> dict[str, dict[str, Any]]:
        return self._load()

    def set_display_name(
        self, document_name: str, display_name: str, overwrite: bool = True
    ) -> None:
        data = self._load()
        entry = data.setdefault(document_name, {})
        if overwrite or "display_name" not in entry:
            entry["display_name"] = display_name
            self._save(data)

    def set_folder(self, document_name: str, folder: str | None) -> None:
        data = self._load()
        entry = data.setdefault(document_name, {})
        entry["folder"] = folder
        self._save(data)
