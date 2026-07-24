"""
Chunking con Docling HybridChunker: DoclingDocument -> ChunkData -> embeddings.

contextualized_text queda cacheado (permite auditar qué texto exacto se
embebe y comparar modelos sin re-chunkear). enriched_text está reservado
para una futura etapa de enriquecimiento con LLM, hoy siempre None.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from docling.chunking import HybridChunker
from docling_core.types.doc.document import DoclingDocument
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """chunk.meta.model_dump() tal cual — fuente única de verdad, sin
    duplicar campos que Docling ya expone."""

    docling_meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkData:
    chunk_id: str
    index: int
    content: str
    contextualized_text: str
    enriched_text: str | None = None
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    token_count: int = 0
    image_paths: list[str] = field(default_factory=list)


class DoclingChunker:
    def __init__(
        self,
        cache_dir: Path,
        embedding_model_name: str = "intfloat/multilingual-e5-large",
        **chunker_kwargs,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Loading tokenizer: {embedding_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)

        self.chunker = HybridChunker(tokenizer=self.tokenizer, **chunker_kwargs)
        self.embedding_model_name = embedding_model_name

        logger.info(
            f"DoclingChunker initialized: "
            f"tokenizer={embedding_model_name}, "
            f"chunker={type(self.chunker).__name__}"
        )

    def chunk_document(
        self,
        doc: DoclingDocument,
        document_name: str,
        force_rechunk: bool = False,
        figures_index: dict[str, str] | None = None,
    ) -> list[ChunkData]:
        chunks_cache_dir = self._get_chunks_cache_dir(document_name)

        if not force_rechunk and self._is_cached(chunks_cache_dir):
            logger.info(f"Loading cached chunks: {document_name}")
            return self._load_from_cache(chunks_cache_dir)

        logger.info(f"Chunking document: {document_name}")
        start_time = datetime.now()

        if figures_index is None:
            figures_index = {}
        if figures_index:
            logger.debug(f"Linking chunks with {len(figures_index)} figures")

        chunks_data = []
        for idx, chunk in enumerate(self.chunker.chunk(doc)):
            chunk_id = f"chunk_{idx + 1:04d}"
            content = chunk.text

            # Enriquecido con contexto (headings, etc.), listo para embedding.
            # Cacheado para poder auditar qué se embebió sin re-chunkear.
            contextualized_text = self.chunker.contextualize(chunk)
            metadata = self._extract_metadata(chunk)

            image_paths = []
            doc_items = metadata.docling_meta.get("doc_items", [])
            for doc_item in doc_items:
                self_ref = doc_item.get("self_ref")
                parent = doc_item.get("parent")
                parent_ref = parent.get("cref") if parent and isinstance(parent, dict) else None

                if self_ref in figures_index:
                    image_paths.append(figures_index[self_ref])
                if parent_ref in figures_index:
                    image_paths.append(figures_index[parent_ref])

            image_paths = list(dict.fromkeys(image_paths))
            token_count = len(self.tokenizer.encode(contextualized_text))

            chunk_data = ChunkData(
                chunk_id=chunk_id,
                index=idx,
                content=content,
                contextualized_text=contextualized_text,
                metadata=metadata,
                token_count=token_count,
                image_paths=image_paths,
            )
            chunks_data.append(chunk_data)

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Chunked into {len(chunks_data)} chunks in {processing_time:.2f}s")

        self._cache_chunks(chunks_data, document_name, chunks_cache_dir, processing_time)
        return chunks_data

    def _extract_metadata(self, chunk) -> ChunkMetadata:
        """docling_meta preserva chunk.meta.model_dump() completo — incluye
        page_numbers y bboxes dentro de doc_items[].prov[]."""
        metadata = ChunkMetadata()

        if not hasattr(chunk, "meta") or chunk.meta is None:
            return metadata

        try:
            metadata.docling_meta = chunk.meta.model_dump()
        except Exception as e:
            logger.warning(f"Could not serialize meta using model_dump(): {e}")
            try:
                if hasattr(chunk.meta, "dict"):
                    metadata.docling_meta = chunk.meta.dict()
            except Exception as fallback_e:
                logger.error(f"Failed to serialize metadata: {fallback_e}")

        return metadata
    
    def _cache_chunks(
        self,
        chunks_data: list[ChunkData],
        document_name: str,
        chunks_cache_dir: Path,
        processing_time: float,
    ) -> None:
        chunks_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Caching chunks to: {chunks_cache_dir}")

        chunks_json_path = chunks_cache_dir / "chunks.json"
        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump([asdict(c) for c in chunks_data], f, indent=2, ensure_ascii=False)
        logger.info(f"✓ chunks.json ({len(chunks_data)} chunks)")

        stats = self._generate_statistics(chunks_data, processing_time)
        stats_path = chunks_cache_dir / "statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info("✓ statistics.json")

        self._generate_audit_markdown(chunks_data, chunks_cache_dir, stats)
        logger.info("✓ chunks_audit.md")

    def _generate_statistics(
        self,
        chunks_data: list[ChunkData],
        processing_time: float,
    ) -> dict[str, Any]:
        if not chunks_data:
            return {}

        import statistics

        token_counts = [c.token_count for c in chunks_data]
        content_lengths = [len(c.content) for c in chunks_data]
        contextualized_lengths = [len(c.contextualized_text) for c in chunks_data]

        all_pages = set()
        chunks_with_headings = 0
        chunks_with_captions = 0
        chunks_with_origin = 0

        for chunk in chunks_data:
            meta = chunk.metadata.docling_meta

            if meta.get("headings"):
                chunks_with_headings += 1
            if meta.get("captions"):
                chunks_with_captions += 1
            if meta.get("origin"):
                chunks_with_origin += 1

            for doc_item in meta.get("doc_items", []):
                for prov in doc_item.get("prov", []):
                    if "page_no" in prov:
                        all_pages.add(prov["page_no"])

        stats = {
            "summary": {
                "total_chunks": len(chunks_data),
                "processing_time_seconds": round(processing_time, 3),
            },
            "token_statistics": {
                "total": sum(token_counts),
                "mean": round(statistics.mean(token_counts), 2),
                "median": statistics.median(token_counts),
                "min": min(token_counts),
                "max": max(token_counts),
                "stdev": round(statistics.stdev(token_counts), 2) if len(token_counts) > 1 else 0,
                "percentile_25": round(statistics.quantiles(token_counts, n=4)[0], 2),
                "percentile_75": round(statistics.quantiles(token_counts, n=4)[2], 2),
                "percentile_95": (
                    round(statistics.quantiles(token_counts, n=20)[18], 2)
                    if len(token_counts) >= 20
                    else max(token_counts)
                ),
            },
            "content_length_statistics": {
                "mean": round(statistics.mean(content_lengths), 2),
                "median": statistics.median(content_lengths),
                "min": min(content_lengths),
                "max": max(content_lengths),
            },
            "contextualized_text_statistics": {
                "mean": round(statistics.mean(contextualized_lengths), 2),
                "median": statistics.median(contextualized_lengths),
                "min": min(contextualized_lengths),
                "max": max(contextualized_lengths),
                "overhead_ratio": round(
                    statistics.mean(contextualized_lengths) / statistics.mean(content_lengths), 3
                ),
            },
            "coverage": {
                "pages_covered": sorted(list(all_pages)),
                "num_pages": len(all_pages),
                "avg_chunks_per_page": (
                    round(len(chunks_data) / len(all_pages), 2) if all_pages else 0
                ),
            },
            "metadata_richness": {
                "chunks_with_headings": chunks_with_headings,
                "chunks_with_captions": chunks_with_captions,
                "chunks_with_origin": chunks_with_origin,
                "pct_with_headings": round(100 * chunks_with_headings / len(chunks_data), 1),
            },
            "size_distribution": {
                "tiny_chunks_lt_100_tokens": sum(1 for t in token_counts if t < 100),
                "small_chunks_100_300": sum(1 for t in token_counts if 100 <= t < 300),
                "medium_chunks_300_600": sum(1 for t in token_counts if 300 <= t < 600),
                "large_chunks_gte_600": sum(1 for t in token_counts if t >= 600),
            }
        }
        
        return stats
    
    def _generate_audit_markdown(
        self,
        chunks_data: list[ChunkData],
        chunks_cache_dir: Path,
        stats: dict[str, Any],
    ) -> None:
        """Markdown para inspección manual — no lo usa el pipeline."""
        audit_path = chunks_cache_dir / "chunks_audit.md"

        with open(audit_path, "w", encoding="utf-8") as f:
            f.write("# Chunking Audit Report\n\n")
            f.write(f"**Total Chunks:** {stats['summary']['total_chunks']}\n\n")
            f.write(f"**Processing Time:** {stats['summary']['processing_time_seconds']}s\n\n")

            f.write("## Summary Statistics\n\n")
            f.write("### Token Distribution\n\n")
            f.write(f"- **Mean:** {stats['token_statistics']['mean']} tokens\n")
            f.write(f"- **Median:** {stats['token_statistics']['median']} tokens\n")
            token_stats = stats["token_statistics"]
            f.write(f"- **Range:** {token_stats['min']} - {token_stats['max']} tokens\n")
            f.write(f"- **95th Percentile:** {token_stats['percentile_95']} tokens\n\n")
            
            f.write("### Coverage\n\n")
            f.write(f"- **Pages:** {stats['coverage']['num_pages']}\n")
            f.write(f"- **Avg chunks/page:** {stats['coverage']['avg_chunks_per_page']}\n")
            pct_headings = stats["metadata_richness"]["pct_with_headings"]
            f.write(f"- **Chunks with headings:** {pct_headings}%\n\n")
            
            f.write("---\n\n")
            f.write("## Chunks Detail\n\n")

            for chunk in chunks_data:
                f.write(f"### {chunk.chunk_id}\n\n")

                meta = chunk.metadata.docling_meta
                headings = meta.get("headings", [])

                f.write(f"**Tokens:** {chunk.token_count}\n\n")

                if headings:
                    f.write(f"**Headings:** {' → '.join(headings)}\n\n")

                pages = set()
                for doc_item in meta.get("doc_items", []):
                    for prov in doc_item.get("prov", []):
                        if "page_no" in prov:
                            pages.add(prov["page_no"])

                if pages:
                    f.write(f"**Pages:** {', '.join(map(str, sorted(pages)))}\n\n")

                if chunk.image_paths:
                    f.write(f"**Imágenes asociadas:** {', '.join(chunk.image_paths)}\n\n")

                f.write("**Contextualized Text (for embedding):**\n\n")
                f.write("```\n")
                f.write(chunk.contextualized_text[:500])
                if len(chunk.contextualized_text) > 500:
                    f.write(f"\n... (truncated, {len(chunk.contextualized_text)} chars total)")
                f.write("\n```\n\n")

                f.write("**Original Content:**\n\n")
                f.write("```\n")
                f.write(chunk.content[:300])
                if len(chunk.content) > 300:
                    f.write(f"\n... (truncated, {len(chunk.content)} chars total)")
                f.write("\n```\n\n")

                f.write("<details>\n")
                f.write("<summary>Full Metadata</summary>\n\n")
                f.write("```json\n")
                f.write(json.dumps(meta, indent=2, default=str))
                f.write("\n```\n\n")
                f.write("</details>\n\n")

                f.write("---\n\n")

    def _get_chunks_cache_dir(self, document_name: str) -> Path:
        return self.cache_dir / document_name / "chunks"

    def _is_cached(self, chunks_cache_dir: Path) -> bool:
        return (chunks_cache_dir / "chunks.json").exists()

    def _load_from_cache(self, chunks_cache_dir: Path) -> list[ChunkData]:
        chunks_json_path = chunks_cache_dir / "chunks.json"

        with open(chunks_json_path, encoding="utf-8") as f:
            chunks_dicts = json.load(f)

        chunks_data = []
        for chunk_dict in chunks_dicts:
            metadata_dict = chunk_dict["metadata"]
            metadata = ChunkMetadata(docling_meta=metadata_dict.get("docling_meta", {}))

            chunk = ChunkData(
                chunk_id=chunk_dict["chunk_id"],
                index=chunk_dict["index"],
                content=chunk_dict["content"],
                contextualized_text=chunk_dict["contextualized_text"],
                enriched_text=chunk_dict.get("enriched_text"),
                metadata=metadata,
                token_count=chunk_dict["token_count"],
                image_paths=chunk_dict.get("image_paths", []),
            )
            chunks_data.append(chunk)

        return chunks_data
