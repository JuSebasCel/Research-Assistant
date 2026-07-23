"""
Servicio de ingesta: sube un PDF nuevo y lo procesa completo (extracción +
chunking + indexación), emitiendo eventos de progreso por etapa.

Orquesta exactamente lo que ya hacen run_chunking.py + run_indexing.py por
CLI (DoclingExtractor -> DoclingChunker -> IndexingService), sin lógica de
pipeline nueva — el único agregado es reportar cada etapa como evento en
vez de solo loguearla, para que la interfaz pueda mostrar progreso vía SSE.
"""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rag_app.services.docling_chunker import DoclingChunker
from rag_app.services.docling_extractor import DoclingExtractor
from rag_app.services.indexing_service import IndexingService

logger = logging.getLogger(__name__)


class IngestService:
    """Orquesta extracción + chunking + indexación de un PDF subido, con
    progreso por etapas en vez de una llamada bloqueante ciega."""

    def __init__(
        self,
        extractor: DoclingExtractor,
        chunker: DoclingChunker,
        indexing_service: IndexingService,
        uploads_dir: Path,
        cache_dir: Path,
    ):
        self.extractor = extractor
        self.chunker = chunker
        self.indexing_service = indexing_service
        self.uploads_dir = uploads_dir
        self.cache_dir = cache_dir

    def ingest_stream(self, pdf_bytes: bytes, filename: str) -> Iterator[dict[str, Any]]:
        """
        Guarda el PDF y lo procesa de punta a punta, cediendo (yield) un
        evento por etapa para que el caller pueda transmitirlo por SSE.

        El progreso es por etapa, no por porcentaje continuo: extracción y
        chunking son llamadas síncronas bloqueantes de Docling, no hay
        manera de reportar avance fino dentro de ellas.
        """
        # Path(filename).name descarta cualquier componente de directorio
        # (ej. "../../etc/passwd") — el nombre viene de un upload HTTP, no
        # es de confianza.
        safe_filename = Path(filename).name
        document_name = Path(safe_filename).stem

        yield {
            "type": "stage",
            "stage": "uploading",
            "message": f"Guardando {safe_filename}...",
        }
        self.uploads_dir.mkdir(parents=True, exist_ok=True)
        pdf_path = self.uploads_dir / safe_filename
        pdf_path.write_bytes(pdf_bytes)

        try:
            yield {
                "type": "stage",
                "stage": "extracting",
                "message": "Extrayendo estructura del PDF...",
            }
            doc = self.extractor.extract(pdf_path=pdf_path, force_reprocess=False)

            yield {
                "type": "stage",
                "stage": "chunking",
                "message": "Generando chunks...",
            }
            figures_index = self.extractor.get_figures_index(document_name)
            chunks = self.chunker.chunk_document(
                doc=doc,
                document_name=document_name,
                figures_index=figures_index,
            )

            yield {
                "type": "stage",
                "stage": "indexing",
                "message": f"Indexando {len(chunks)} chunks...",
            }
            chunks_json_path = self.cache_dir / document_name / "chunks" / "chunks.json"
            result = self.indexing_service.index_document(chunks_json_path, document_name)

        except Exception as e:
            logger.exception(f"Error al ingerir {safe_filename}")
            yield {"type": "error", "error": f"Error al procesar el documento: {e}"}
            return

        yield {
            "type": "done",
            "document_name": document_name,
            "chunks_indexed": result["chunks_indexed"],
        }
