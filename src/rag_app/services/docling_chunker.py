"""
Docling Hybrid Chunker Service

Uses official Docling HybridChunker API for intelligent document chunking.
NO manual chunking, NO heuristics, NO regex - only official APIs.

Verified API (from official Docling documentation):
- HybridChunker(tokenizer=...) - Pydantic-based configuration
- chunker.chunk(dl_doc) -> Iterator[DocChunk]
- chunker.contextualize(chunk) -> str (context-enriched text ready for embedding)
- DocChunk.text (original chunk text)
- DocChunk.meta (DocMeta with rich metadata)
- DocMeta.model_dump() (official Pydantic serialization - source of truth)

Architecture (RAG-ready pipeline):
    DoclingDocument → HybridChunker → ChunkData → [Future: LLM Enricher] → Embeddings → Vector DB

Key Design Decisions:
1. Tokenizer: Uses AutoTokenizer.from_pretrained() with HuggingFace Transformers
   - Official recommended approach per Docling documentation
   - Tokenizer matches embedding model (multilingual-e5-large) for accurate token counting

2. Metadata: Uses DocMeta.model_dump() as source of truth
   - Minimizes coupling with internal DocMeta structure
   - Preserves complete official metadata for future use
   - Derived fields (headings, captions, etc.) for convenience only

3. contextualized_text: Cached result from chunker.contextualize()
   - Enables auditing exact text that will be embedded
   - Allows changing embedding models without re-chunking
   - Supports comparing different embedding strategies with same input

4. enriched_text: Reserved field for future LLM enrichment (e.g., Qwen)
   - Currently None
   - Will be populated by future enrichment pipeline stage
   - Preserves architecture for RAG evolution
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict, field
from datetime import datetime

from transformers import AutoTokenizer
from docling.chunking import HybridChunker
from docling_core.types.doc.document import DoclingDocument

logger = logging.getLogger(__name__)


@dataclass
class ChunkMetadata:
    """
    Official metadata from DocChunk.meta (DocMeta).
    
    Single source of truth: chunk.meta.model_dump() from Pydantic API.
    No duplicate fields - all information is in docling_meta.
    
    Minimizes coupling with internal DocMeta structure.
    Future-proof: if Docling changes metadata schema, we preserve everything.
    
    Contains (from official API):
    - schema_name, version: Docling metadata version info
    - doc_items: List of document items with labels, provenance, etc.
    - headings: List of hierarchical headings for this chunk
    - captions: Optional captions (tables, figures)
    - origin: Optional document origin information
    """
    docling_meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChunkData:
    """
    Complete chunk information for RAG pipeline.
    
    Architecture-ready for multi-stage pipeline:
        Extractor → HybridChunker → [Future: LLM Enricher] → Embedding Model → Vector DB
    
    Fields:
    - chunk_id: Unique identifier (e.g., 'chunk_0001')
    - index: Zero-based chunk index
    - content: Original chunk text (from chunk.text)
    - contextualized_text: Context-enriched text from HybridChunker.contextualize()
      Ready for embedding model input. Cached for auditing and model comparison.
    - enriched_text: Reserved for future LLM enrichment stage (e.g., Qwen)
      Will be populated by future enrichment pipeline. None until then.
    - metadata: Official Docling metadata from chunk.meta.model_dump()
    - token_count: Token count using embedding model tokenizer
      (No official API available, calculated manually)
    - image_paths: List of relative paths to images referenced in this chunk
      (e.g., ['figures/figure_001.png'])
    """
    chunk_id: str
    index: int
    content: str
    contextualized_text: str
    enriched_text: Optional[str] = None  # Default None for future enrichment
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    token_count: int = 0
    image_paths: List[str] = field(default_factory=list)


class DoclingChunker:
    """
    Production chunker using official Docling HybridChunker API.
    
    Configuration:
    - Uses intfloat/multilingual-e5-large tokenizer (matches embedding model)
    - All chunking done by HybridChunker (no manual logic)
    - Contextualization via official contextualize() method
    
    Verified API usage (from official documentation):
    - HybridChunker(tokenizer=tokenizer, ...) - Pydantic config
    - chunker.chunk(doc) - Returns Iterator[DocChunk]
    - chunker.contextualize(chunk) - Returns contextualized string
    - chunk.text - Original text
    - chunk.meta - DocMeta with metadata
    - chunk.meta.model_dump() - Official Pydantic serialization
    
    Note on tokenizer:
    - Uses AutoTokenizer.from_pretrained() (HuggingFace Transformers)
    - This is the official recommended approach per Docling documentation
    - Tokenizer should match the embedding model for accurate token counting
    """
    
    def __init__(
        self,
        cache_dir: Path,
        embedding_model_name: str = "intfloat/multilingual-e5-large",
        **chunker_kwargs,
    ):
        """
        Initialize chunker with official HybridChunker.
        
        Args:
            cache_dir: Directory for caching chunks
            embedding_model_name: HuggingFace model for tokenizer
            **chunker_kwargs: Additional args for HybridChunker (e.g., max_tokens)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Load tokenizer matching embedding model
        logger.info(f"Loading tokenizer: {embedding_model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        
        # Initialize official HybridChunker
        logger.info("Initializing HybridChunker with official API")
        self.chunker = HybridChunker(
            tokenizer=self.tokenizer,
            **chunker_kwargs
        )
        
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
        figures_index: Optional[Dict[str, str]] = None,
    ) -> List[ChunkData]:
        """
        Chunk document using official HybridChunker API.
        
        Args:
            doc: DoclingDocument (from extractor)
            document_name: Name for caching
            force_rechunk: Force re-chunking even if cached
            figures_index: Optional mapping of self_ref to image_path (for linking chunks to images)
        
        Returns:
            List of ChunkData with content, contextualized_text, and metadata
        """
        chunks_cache_dir = self._get_chunks_cache_dir(document_name)
        
        # Check cache
        if not force_rechunk and self._is_cached(chunks_cache_dir):
            logger.info(f"Loading cached chunks: {document_name}")
            return self._load_from_cache(chunks_cache_dir)
        
        logger.info(f"Chunking document: {document_name}")
        start_time = datetime.now()
        
        # Use provided figures_index or empty dict
        if figures_index is None:
            figures_index = {}
        
        if figures_index:
            logger.debug(f"Linking chunks with {len(figures_index)} figures")
        
        # Use official chunk() API
        chunks_data = []
        for idx, chunk in enumerate(self.chunker.chunk(doc)):
            chunk_id = f"chunk_{idx + 1:04d}"
            
            # Extract content (verified API: chunk.text)
            content = chunk.text
            
            # Get contextualized text (verified API: chunker.contextualize())
            # This text is enriched with metadata context and ready for embedding
            # Cached to support:
            # - Auditing exactly what text will be embedded
            # - Changing embedding models without re-chunking
            # - Comparing different embedding strategies with same input
            contextualized_text = self.chunker.contextualize(chunk)
            
            # Extract metadata using official model_dump() (source of truth)
            metadata = self._extract_metadata(chunk)
            
            # Resolve image paths for this chunk
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
            
            # Remove duplicates while preserving order
            image_paths = list(dict.fromkeys(image_paths))
            
            # Count tokens using same tokenizer
            # Note: No official token_count in DocChunk API, must calculate manually
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
        logger.info(
            f"Chunked into {len(chunks_data)} chunks in {processing_time:.2f}s"
        )
        
        # Cache results
        self._cache_chunks(chunks_data, document_name, chunks_cache_dir, processing_time)
        
        return chunks_data
    
    def _extract_metadata(self, chunk) -> ChunkMetadata:
        """
        Extract metadata from chunk.meta (DocMeta).
        
        Uses official Pydantic model_dump() as single source of truth.
        No duplicate fields - everything is preserved in docling_meta.
        
        Verified fields in model_dump() (from official API):
        - schema_name: "docling_core.transforms.chunker.DocMeta"
        - version: Version string (e.g., "1.0.0")
        - doc_items: List of document items with labels and provenance
        - headings: List of hierarchical headings
        - captions: Optional captions dict
        - origin: Optional document origin info
        
        Note: page_numbers and bboxes are inside doc_items[].prov[]
        """
        metadata = ChunkMetadata()
        
        if not hasattr(chunk, "meta") or chunk.meta is None:
            return metadata
        
        # Serialize official metadata using Pydantic API (source of truth)
        try:
            metadata.docling_meta = chunk.meta.model_dump()
        except Exception as e:
            logger.warning(f"Could not serialize meta using model_dump(): {e}")
            # Fallback: try dict() method (Pydantic v1 compatibility)
            try:
                if hasattr(chunk.meta, "dict"):
                    metadata.docling_meta = chunk.meta.dict()
            except Exception as fallback_e:
                logger.error(f"Failed to serialize metadata: {fallback_e}")
        
        return metadata
    
    def _cache_chunks(
        self,
        chunks_data: List[ChunkData],
        document_name: str,
        chunks_cache_dir: Path,
        processing_time: float,
    ) -> None:
        """Cache chunks with comprehensive audit files."""
        chunks_cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Caching chunks to: {chunks_cache_dir}")
        
        # 1. Save all chunks as single JSON
        chunks_json_path = chunks_cache_dir / "chunks.json"
        with open(chunks_json_path, "w", encoding="utf-8") as f:
            json.dump(
                [asdict(c) for c in chunks_data],
                f,
                indent=2,
                ensure_ascii=False,
            )
        logger.info(f"✓ chunks.json ({len(chunks_data)} chunks)")
        
        # 2. Generate statistics
        stats = self._generate_statistics(chunks_data, processing_time)
        stats_path = chunks_cache_dir / "statistics.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        logger.info(f"✓ statistics.json")
        
        # 3. Generate human-readable audit file
        self._generate_audit_markdown(chunks_data, chunks_cache_dir, stats)
        logger.info(f"✓ chunks_audit.md")
    
    def _generate_statistics(
        self,
        chunks_data: List[ChunkData],
        processing_time: float,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive chunking statistics for auditing.
        
        Includes distribution metrics to evaluate chunking quality.
        """
        if not chunks_data:
            return {}
        
        import statistics
        
        token_counts = [c.token_count for c in chunks_data]
        content_lengths = [len(c.content) for c in chunks_data]
        contextualized_lengths = [len(c.contextualized_text) for c in chunks_data]
        
        # Extract page numbers and headings from docling_meta
        all_pages = set()
        chunks_with_headings = 0
        chunks_with_captions = 0
        chunks_with_origin = 0
        
        for chunk in chunks_data:
            meta = chunk.metadata.docling_meta
            
            # Count headings
            if meta.get("headings"):
                chunks_with_headings += 1
            
            # Count captions
            if meta.get("captions"):
                chunks_with_captions += 1
            
            # Count origin
            if meta.get("origin"):
                chunks_with_origin += 1
            
            # Extract page numbers from doc_items provenance
            for doc_item in meta.get("doc_items", []):
                for prov in doc_item.get("prov", []):
                    if "page_no" in prov:
                        all_pages.add(prov["page_no"])
        
        # Calculate comprehensive statistics
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
                "percentile_95": round(statistics.quantiles(token_counts, n=20)[18], 2) if len(token_counts) >= 20 else max(token_counts),
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
                "overhead_ratio": round(statistics.mean(contextualized_lengths) / statistics.mean(content_lengths), 3),
            },
            "coverage": {
                "pages_covered": sorted(list(all_pages)),
                "num_pages": len(all_pages),
                "avg_chunks_per_page": round(len(chunks_data) / len(all_pages), 2) if all_pages else 0,
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
        chunks_data: List[ChunkData],
        chunks_cache_dir: Path,
        stats: Dict[str, Any],
    ) -> None:
        """
        Generate human-readable audit file in Markdown.
        
        For manual inspection of chunking quality.
        Not used by the pipeline - purely for human review.
        """
        audit_path = chunks_cache_dir / "chunks_audit.md"
        
        with open(audit_path, "w", encoding="utf-8") as f:
            # Header
            f.write("# Chunking Audit Report\n\n")
            f.write(f"**Total Chunks:** {stats['summary']['total_chunks']}\n\n")
            f.write(f"**Processing Time:** {stats['summary']['processing_time_seconds']}s\n\n")
            
            # Quick stats
            f.write("## Summary Statistics\n\n")
            f.write("### Token Distribution\n\n")
            f.write(f"- **Mean:** {stats['token_statistics']['mean']} tokens\n")
            f.write(f"- **Median:** {stats['token_statistics']['median']} tokens\n")
            f.write(f"- **Range:** {stats['token_statistics']['min']} - {stats['token_statistics']['max']} tokens\n")
            f.write(f"- **95th Percentile:** {stats['token_statistics']['percentile_95']} tokens\n\n")
            
            f.write("### Coverage\n\n")
            f.write(f"- **Pages:** {stats['coverage']['num_pages']}\n")
            f.write(f"- **Avg chunks/page:** {stats['coverage']['avg_chunks_per_page']}\n")
            f.write(f"- **Chunks with headings:** {stats['metadata_richness']['pct_with_headings']}%\n\n")
            
            f.write("---\n\n")
            
            # Individual chunks
            f.write("## Chunks Detail\n\n")
            
            for chunk in chunks_data:
                f.write(f"### {chunk.chunk_id}\n\n")
                
                # Metadata summary
                meta = chunk.metadata.docling_meta
                headings = meta.get("headings", [])
                
                f.write(f"**Tokens:** {chunk.token_count}\n\n")
                
                if headings:
                    f.write(f"**Headings:** {' → '.join(headings)}\n\n")
                
                # Extract page numbers from doc_items
                pages = set()
                for doc_item in meta.get("doc_items", []):
                    for prov in doc_item.get("prov", []):
                        if "page_no" in prov:
                            pages.add(prov["page_no"])
                
                if pages:
                    f.write(f"**Pages:** {', '.join(map(str, sorted(pages)))}\n\n")
                
                if chunk.image_paths:
                    f.write(f"**Imágenes asociadas:** {', '.join(chunk.image_paths)}\n\n")
                
                # Contextualized text (what will be embedded)
                f.write("**Contextualized Text (for embedding):**\n\n")
                f.write("```\n")
                f.write(chunk.contextualized_text[:500])
                if len(chunk.contextualized_text) > 500:
                    f.write(f"\n... (truncated, {len(chunk.contextualized_text)} chars total)")
                f.write("\n```\n\n")
                
                # Original content
                f.write("**Original Content:**\n\n")
                f.write("```\n")
                f.write(chunk.content[:300])
                if len(chunk.content) > 300:
                    f.write(f"\n... (truncated, {len(chunk.content)} chars total)")
                f.write("\n```\n\n")
                
                # Metadata summary
                f.write("<details>\n")
                f.write("<summary>Full Metadata</summary>\n\n")
                f.write("```json\n")
                f.write(json.dumps(meta, indent=2, default=str))
                f.write("\n```\n\n")
                f.write("</details>\n\n")
                
                f.write("---\n\n")
    
    def _get_chunks_cache_dir(self, document_name: str) -> Path:
        """Get cache directory for document chunks."""
        return self.cache_dir / document_name / "chunks"
    
    def _is_cached(self, chunks_cache_dir: Path) -> bool:
        """Check if chunks are already cached."""
        return (chunks_cache_dir / "chunks.json").exists()
    
    def _load_from_cache(self, chunks_cache_dir: Path) -> List[ChunkData]:
        """Load chunks from cache."""
        chunks_json_path = chunks_cache_dir / "chunks.json"
        
        with open(chunks_json_path, "r", encoding="utf-8") as f:
            chunks_dicts = json.load(f)
        
        # Reconstruct ChunkData objects
        chunks_data = []
        for chunk_dict in chunks_dicts:
            # Reconstruct metadata
            metadata_dict = chunk_dict["metadata"]
            metadata = ChunkMetadata(
                docling_meta=metadata_dict.get("docling_meta", {})
            )
            
            # Reconstruct chunk
            chunk = ChunkData(
                chunk_id=chunk_dict["chunk_id"],
                index=chunk_dict["index"],
                content=chunk_dict["content"],
                contextualized_text=chunk_dict["contextualized_text"],
                enriched_text=chunk_dict.get("enriched_text"),  # May be None
                metadata=metadata,
                token_count=chunk_dict["token_count"],
                image_paths=chunk_dict.get("image_paths", []),
            )
            chunks_data.append(chunk)
        
        return chunks_data
