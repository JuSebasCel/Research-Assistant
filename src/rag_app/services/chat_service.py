"""
Servicio de chat: orquesta hybrid search + generación con Gemini.

Pipeline:
    query -> IndexingService.search() (hybrid: BM25+dense+RRF)
          -> umbral de confianza (corta antes de llamar al LLM si no hay
             nada relevante, en vez de dejar que el modelo invente)
          -> prompt con citas obligatorias -> GeminiProvider.generate_stream()
"""

import logging
import mimetypes
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from rag_app.providers.llm import GeminiProvider
from rag_app.services.indexing_service import IndexingService

logger = logging.getLogger(__name__)

# Tope de imágenes por respuesta: evita disparar tokens/costo mandando
# figuras de chunks solo marginalmente relevantes (top_k puede traer varios
# chunks del mismo documento, cada uno con su propia imagen).
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
tal como aparecen en los fragmentos (formato: [documento, p. X])."""


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
        Args:
            indexing_service: Servicio de retrieval ya construido (hybrid search)
            llm_provider: Provider de Gemini
            cache_dir: Directorio base de cache (donde viven las figuras:
                cache_dir/{document_name}/{image_path})
            min_score_threshold: Score RRF mínimo del mejor resultado para
                intentar responder. Medido contra el corpus real: preguntas
                claramente fuera de tema (clima, recetas, deportes) sacan
                scores RRF de 0.33-0.5 — el mismo rango que preguntas
                genuinamente relevantes (0.5-0.57). El score RRF es
                puramente ranking relativo dentro de la colección, no
                similitud absoluta: siempre hay un "candidato más cercano"
                aunque sea irrelevante, y ese consigue un score decente
                solo por quedar primero. Por eso este umbral se deja bajo
                (solo filtra el caso de colección vacía / sin resultados) —
                no puede distinguir de forma confiable "irrelevante pero
                mejor rankeado" de "relevante"; esa distinción la hace el
                LLM (ver SYSTEM_INSTRUCTION), que si demostró funcionar
                bien en pruebas reales.
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
        page_filter: int | None = None,
        heading_contains: str | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Genera la respuesta en streaming, con retrieval + citas.

        Yields:
            Eventos con forma:
            - {"type": "no_results"} — no hay contenido suficientemente relevante
            - {"type": "chunk", "text": "..."} — fragmento de la respuesta
            - {"type": "done", "citations": [...]} — fin, con las fuentes usadas
        """
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

        user_prompt = self._build_prompt(query, results)
        images = self._load_images(results)

        for text_chunk in self.llm_provider.generate_stream(
            SYSTEM_INSTRUCTION, user_prompt, images=images
        ):
            yield {"type": "chunk", "text": text_chunk}

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

    def _build_prompt(self, query: str, results: list[dict[str, Any]]) -> str:
        """Arma el prompt de usuario: fragmentos recuperados + la pregunta."""
        fragments = []
        for r in results:
            pages = ", ".join(str(p) for p in r["pages"]) or "?"
            fragments.append(
                f"[Documento: {r['document_name']}, página(s): {pages}]\n{r['content']}"
            )

        context = "\n\n---\n\n".join(fragments)
        return f"Fragmentos disponibles:\n\n{context}\n\n---\n\nPregunta: {query}"
