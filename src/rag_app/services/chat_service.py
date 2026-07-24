"""Orquesta hybrid search + generación con Gemini: retrieval, umbral de
confianza, prompt con citas obligatorias, streaming."""

import logging
import mimetypes
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from google.genai import errors as genai_errors

from rag_app.providers.llm import GeminiProvider
from rag_app.services.indexing_service import IndexingService

logger = logging.getLogger(__name__)

MAX_IMAGES_PER_ANSWER = 4

SYSTEM_INSTRUCTION = """Eres un asistente que responde preguntas sobre papers científicos \
basándote exclusivamente en los fragmentos de texto que se te proporcionan.

Reglas estrictas:
- Responde solo con información presente en los fragmentos dados. No completes \
con conocimiento general ni infieras más allá de lo que dicen los fragmentos.
- Si los fragmentos no contienen la respuesta a la pregunta, dilo explícitamente \
("No encontré esa información en los documentos disponibles") en vez de inventar \
o aproximar una respuesta.
- Cada afirmación debe venir acompañada de su cita: nombre del documento y página, \
tal como aparecen en los fragmentos (formato: [documento, p. X]).
- Se te da además la lista completa de "Documentos disponibles en el sistema": \
úsala solo para responder preguntas sobre qué documentos existen o si un paper \
en particular está cargado. Nunca la uses como fuente de contenido — cualquier \
afirmación sobre lo que dice un documento debe seguir viniendo de los fragmentos.
- Si la pregunta pide comparar, contrastar, o encontrar diferencias o similitudes \
entre varios documentos, estructura la respuesta explícitamente por documento \
(usando su nombre como encabezado) antes de dar una conclusión final."""


class ChatService:
    """
    Orquesta retrieval híbrido + generación con Gemini, con umbral de
    confianza y citas obligatorias.
    """

    def __init__(
        self,
        indexing_service: IndexingService,
        llm_provider: GeminiProvider,
        cache_dir: Path,
        min_score_threshold: float = 0.05,
    ):
        """
        min_score_threshold se deja bajo a propósito: el score RRF es
        ranking relativo dentro de la colección, no similitud absoluta, así
        que preguntas fuera de tema (0.33-0.5) y relevantes (0.5-0.57) caen
        en rangos casi idénticos. No distingue "irrelevante pero mejor
        rankeado" de "relevante" — eso lo hace el LLM vía SYSTEM_INSTRUCTION.
        Este umbral solo filtra el caso de colección vacía.
        """
        self.indexing_service = indexing_service
        self.llm_provider = llm_provider
        self.cache_dir = cache_dir
        self.min_score_threshold = min_score_threshold

    def answer_stream(
        self,
        query: str,
        top_k: int = 5,
        document_filter: str | None = None,
        document_filters: list[str] | None = None,
        page_filter: int | None = None,
        heading_contains: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Genera la respuesta en streaming, con retrieval + citas.

        document_filter acota a un solo documento; document_filters acota
        el fan-out a un subconjunto (ej. una carpeta). Sin ninguno de los
        dos, el fan-out corre sobre todos los documentos indexados — una
        búsqueda global puede dejar afuera documentos relevantes que no
        ganan el ranking.

        Yields:
            - {"type": "no_results"}
            - {"type": "chunk", "text": "..."}
            - {"type": "done", "citations": [...]}
            - {"type": "error", "error": "..."} — el stream termina sin "done"
        """
        if document_filter is None:
            results = self.indexing_service.search_across_documents(
                query=query,
                max_total=top_k,
                page_filter=page_filter,
                heading_contains=heading_contains,
                document_names=document_filters,
            )
        else:
            results = self.indexing_service.search(
                query=query,
                top_k=top_k,
                document_filter=document_filter,
                page_filter=page_filter,
                heading_contains=heading_contains,
            )

        if not results or results[0]["score"] < self.min_score_threshold:
            logger.info(f"Sin resultados suficientemente relevantes para: '{query}'")
            yield {"type": "no_results"}
            return

        # Lista completa siempre, no solo lo que trajo el retrieval — si no,
        # "¿qué documentos tienes?" nunca tiene una respuesta real.
        document_names = self.indexing_service.list_indexed_documents()
        user_prompt = self._build_prompt(query, results, document_names)
        images = self._load_images(results)

        try:
            for text_chunk in self.llm_provider.generate_stream(
                SYSTEM_INSTRUCTION, user_prompt, images=images
            ):
                yield {"type": "chunk", "text": text_chunk}
        except genai_errors.ClientError as e:
            if e.code == 429 or e.status == "RESOURCE_EXHAUSTED":
                logger.warning(f"Cuota de Gemini agotada: {e.message}")
                yield {
                    "type": "error",
                    "error": "Se acabó la cuota gratuita de Gemini por hoy. Intenta más tarde.",
                }
            else:
                logger.error(f"Error de Gemini: {e.message}")
                yield {"type": "error", "error": f"Error al generar la respuesta: {e.message}"}
            return
        except genai_errors.APIError as e:
            logger.error(f"Error de Gemini: {e}")
            yield {"type": "error", "error": "Error al generar la respuesta. Intenta de nuevo."}
            return

        citations = [
            {
                "document_name": r["document_name"],
                "chunk_id": r["chunk_id"],
                "pages": r["pages"],
                "image_urls": [
                    f"/static/cache/{r['document_name']}/{p}" for p in r.get("image_paths", [])
                ],
            }
            for r in results
        ]
        yield {"type": "done", "citations": citations}

    def _load_images(self, results: list[dict[str, Any]]) -> list[tuple[bytes, str]]:
        """
        Carga (bytes, mime_type) de las figuras asociadas a los chunks
        recuperados, deduplicadas y acotadas a MAX_IMAGES_PER_ANSWER.
        """
        images: list[tuple[bytes, str]] = []
        seen_paths: set[Path] = set()

        for r in results:
            for image_path in r.get("image_paths", []):
                if len(images) >= MAX_IMAGES_PER_ANSWER:
                    return images

                full_path = self.cache_dir / r["document_name"] / image_path
                if full_path in seen_paths:
                    continue
                seen_paths.add(full_path)

                if not full_path.exists():
                    logger.warning(f"Imagen referenciada no encontrada en disco: {full_path}")
                    continue

                mime_type, _ = mimetypes.guess_type(full_path.name)
                images.append((full_path.read_bytes(), mime_type or "image/png"))

        return images

    def _build_prompt(
        self, query: str, results: list[dict[str, Any]], document_names: list[str]
    ) -> str:
        """Arma el prompt: catálogo de documentos + fragmentos recuperados + la pregunta."""
        fragments = []
        for r in results:
            pages = ", ".join(str(p) for p in r["pages"]) or "?"
            fragments.append(
                f"[Documento: {r['document_name']}, página(s): {pages}]\n{r['content']}"
            )

        context = "\n\n---\n\n".join(fragments)
        documents_line = "Documentos disponibles en el sistema: " + ", ".join(document_names)
        return (
            f"{documents_line}\n\n"
            f"Fragmentos disponibles:\n\n{context}\n\n---\n\nPregunta: {query}"
        )
