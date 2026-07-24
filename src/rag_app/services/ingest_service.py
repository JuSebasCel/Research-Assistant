"""Sube un PDF, lo extrae/chunkea/indexa, y emite un evento de progreso por
etapa (mismo pipeline que run_chunking.py + run_indexing.py, vía CLI)."""

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from docling_core.types.doc import DocItemLabel

from rag_app.services.docling_chunker import DoclingChunker
from rag_app.services.docling_extractor import DoclingExtractor
from rag_app.services.document_metadata_service import DocumentMetadataService
from rag_app.services.indexing_service import IndexingService

logger = logging.getLogger(__name__)


def _extract_title(doc: Any) -> str | None:
    """
    Heurística barata (sin llamar al LLM) para el título real del paper:
    primero busca un item TITLE explícito de Docling; si no hay, el primer
    section_header sirve de aproximación razonable; si tampoco, no hay
    título confiable y el caller cae al nombre de archivo.
    """
    section_header_fallback: str | None = None
    for item in doc.texts:
        if item.label == DocItemLabel.TITLE:
            return item.text.strip()
        if section_header_fallback is None and item.label == DocItemLabel.SECTION_HEADER:
            section_header_fallback = item.text.strip()
    return section_header_fallback


class IngestService:
    def __init__(
        self,
        extractor: DoclingExtractor,
        chunker: DoclingChunker,
        indexing_service: IndexingService,
        document_metadata_service: DocumentMetadataService,
        uploads_dir: Path,
        cache_dir: Path,
    ):
        self.extractor = extractor
        self.chunker = chunker
        self.indexing_service = indexing_service
        self.document_metadata_service = document_metadata_service
        self.uploads_dir = uploads_dir
        self.cache_dir = cache_dir

    def ingest_stream(self, pdf_bytes: bytes, filename: str) -> Iterator[dict[str, Any]]:
        """
        Progreso por etapa, no por porcentaje: extracción y chunking son
        llamadas bloqueantes de Docling, sin forma de medir avance fino.
        """
        # El nombre viene de un upload HTTP, no es de confianza (path traversal).
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

            # overwrite=False: no pisa un rename manual ya hecho.
            title = _extract_title(doc)
            if title:
                self.document_metadata_service.set_display_name(
                    document_name, title, overwrite=False
                )

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
