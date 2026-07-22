"""
Chunk Audit Tool

Audits the quality of HybridChunker results before embeddings phase.

Usage:
    python inspect_chunks.py path/to/document.pdf

Output:
    Creates audit_chunks_<pdf>_<timestamp>/ with:
    - REPORT.txt (comprehensive statistics and quality checks)
    - chunks.md (human-readable chunk visualization)
    - statistics.json (detailed statistics)
    - chunks/ (individual chunk JSON files)
"""

import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_app.services.docling_extractor import DoclingExtractor
from rag_app.services.docling_chunker import DoclingChunker
from rag_app.core.config import get_settings

settings = get_settings()


def generate_chunks_markdown(chunks_data: List[Dict[str, Any]]) -> str:
    """
    Generate human-readable chunks.md for visual inspection.
    
    This is the MOST IMPORTANT file for auditing chunking quality.
    """
    lines = []
    
    for chunk in chunks_data:
        chunk_id = chunk["chunk_id"]
        index = chunk["index"]
        content = chunk["content"]
        embedding_text = chunk["embedding_text"]
        metadata = chunk["metadata"]
        token_count = chunk["token_count"]
        
        lines.extend([
            "=" * 80,
            f"CHUNK {index + 1:03d} - {chunk_id}",
            "=" * 80,
            "",
            "Metadata:",
            "-" * 80,
        ])
        
        # Show metadata
        if metadata.get("page_numbers"):
            lines.append(f"  Pages: {', '.join(map(str, metadata['page_numbers']))}")
        
        if metadata.get("headings"):
            lines.append("  Headings:")
            for heading in metadata["headings"]:
                lines.append(f"    - {heading}")
        
        if metadata.get("captions"):
            lines.append("  Captions:")
            for key, value in metadata["captions"].items():
                lines.append(f"    {key}: {value}")
        
        if metadata.get("doc_items"):
            lines.append(f"  Doc Items: {len(metadata['doc_items'])} items")
        
        if metadata.get("bboxes"):
            lines.append(f"  Bounding Boxes: {len(metadata['bboxes'])} boxes")
        
        lines.extend([
            f"  Token Count: {token_count}",
            "",
            "Content:",
            "-" * 80,
            content,
            "",
            "Embedding Text (from contextualize()):",
            "-" * 80,
            embedding_text,
            "",
            "",
        ])
    
    return "\n".join(lines)


def generate_audit_report(
    pdf_path: Path,
    chunks_data: List[Dict[str, Any]],
    statistics: Dict[str, Any],
    audit_dir: Path,
) -> str:
    """Generate comprehensive audit report."""
    lines = [
        "=" * 80,
        "DOCLING CHUNK AUDIT REPORT",
        "=" * 80,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"PDF: {pdf_path.name}",
        f"Audit Directory: {audit_dir.absolute()}",
        "",
        "-" * 80,
        "CHUNKING STATISTICS",
        "-" * 80,
        "",
        f"Total Chunks: {statistics.get('total_chunks', 0)}",
        f"Processing Time: {statistics.get('processing_time_seconds', 0):.2f}s",
        "",
    ]
    
    # Token statistics
    token_stats = statistics.get("token_statistics", {})
    if token_stats:
        lines.extend([
            "Token Statistics (multilingual-e5-large tokenizer):",
            f"  Mean: {token_stats.get('mean', 0):.1f} tokens",
            f"  Min:  {token_stats.get('min', 0)} tokens",
            f"  Max:  {token_stats.get('max', 0)} tokens",
            f"  Total: {token_stats.get('total', 0)} tokens",
            "",
        ])
    
    # Content length statistics
    content_stats = statistics.get("content_length_statistics", {})
    if content_stats:
        lines.extend([
            "Content Length Statistics:",
            f"  Mean: {content_stats.get('mean', 0):.1f} characters",
            f"  Min:  {content_stats.get('min', 0)} characters",
            f"  Max:  {content_stats.get('max', 0)} characters",
            "",
        ])
    
    # Coverage
    coverage = statistics.get("coverage", {})
    if coverage:
        pages_covered = coverage.get("pages_covered", [])
        lines.extend([
            "Coverage:",
            f"  Pages Covered: {len(pages_covered)} pages",
            f"  Page Numbers: {', '.join(map(str, pages_covered[:20]))}{'...' if len(pages_covered) > 20 else ''}",
            "",
        ])
    
    # Metadata counts
    meta_counts = statistics.get("metadata_counts", {})
    if meta_counts:
        lines.extend([
            "Metadata Distribution:",
            f"  Chunks with Headings: {meta_counts.get('chunks_with_headings', 0)}",
            f"  Chunks with Captions: {meta_counts.get('chunks_with_captions', 0)}",
            f"  Chunks with BBoxes:   {meta_counts.get('chunks_with_bboxes', 0)}",
            "",
        ])
    
    # Quality checks
    lines.extend([
        "-" * 80,
        "QUALITY CHECKS",
        "-" * 80,
        "",
    ])
    
    total_chunks = len(chunks_data)
    
    # Check 1: Empty chunks
    empty_chunks = [c for c in chunks_data if not c["content"].strip()]
    if empty_chunks:
        lines.append(f"⚠️  Warning: {len(empty_chunks)} empty chunks detected")
    else:
        lines.append(f"✓ No empty chunks")
    
    # Check 2: Very small chunks
    small_chunks = [c for c in chunks_data if c["token_count"] < 50]
    if small_chunks:
        lines.append(f"⚠️  Warning: {len(small_chunks)} chunks with < 50 tokens")
    else:
        lines.append(f"✓ No very small chunks")
    
    # Check 3: Very large chunks
    large_chunks = [c for c in chunks_data if c["token_count"] > 1000]
    if large_chunks:
        lines.append(f"⚠️  Warning: {len(large_chunks)} chunks with > 1000 tokens")
    else:
        lines.append(f"✓ No oversized chunks")
    
    # Check 4: Chunks without headings
    no_heading = [c for c in chunks_data if not c["metadata"].get("headings")]
    if no_heading:
        pct = (len(no_heading) / total_chunks * 100) if total_chunks > 0 else 0
        lines.append(f"⚠️  Info: {len(no_heading)} chunks ({pct:.1f}%) without headings")
    else:
        lines.append(f"✓ All chunks have headings")
    
    # Check 5: Token distribution
    if token_stats:
        mean_tokens = token_stats.get("mean", 0)
        if mean_tokens < 100:
            lines.append(f"⚠️  Warning: Average chunk size is small ({mean_tokens:.0f} tokens)")
        elif mean_tokens > 800:
            lines.append(f"⚠️  Warning: Average chunk size is large ({mean_tokens:.0f} tokens)")
        else:
            lines.append(f"✓ Good average chunk size ({mean_tokens:.0f} tokens)")
    
    lines.append("")
    
    # Recommendations
    lines.extend([
        "-" * 80,
        "MANUAL INSPECTION CHECKLIST",
        "-" * 80,
        "",
        "1. Open chunks.md ⭐⭐⭐⭐⭐ (MOST IMPORTANT)",
        "   - Verify chunks are coherent and complete",
        "   - Check that chunk boundaries make sense",
        "   - Verify headings are correctly captured",
        "   - Check embedding_text vs content differences",
        "   - Look for duplicated or missing content",
        "",
        "2. Review statistics.json",
        "   - Check token distribution",
        "   - Verify page coverage",
        "   - Check metadata completeness",
        "",
        "3. Spot-check individual chunks/",
        "   - Open random chunk JSON files",
        "   - Verify metadata is rich and useful",
        "   - Check that contextualize() adds value",
        "",
        "4. Compare against original markdown",
        "   - Open markdown.md from extraction phase",
        "   - Verify no content was lost in chunking",
        "   - Check that structure is preserved",
        "",
        "5. Check for common issues",
        "   - Split paragraphs or sentences",
        "   - Orphaned table/figure references",
        "   - Missing context in chunks",
        "   - Duplicated headings",
        "",
    ])
    
    # Inspection order
    lines.extend([
        "=" * 80,
        "RECOMMENDED INSPECTION ORDER",
        "=" * 80,
        "",
        "1. chunks.md             ⭐⭐⭐⭐⭐  (START HERE - visual inspection)",
        "2. REPORT.txt            ⭐⭐⭐⭐   (this file - overview)",
        "3. statistics.json       ⭐⭐⭐     (detailed metrics)",
        "4. chunks/               ⭐⭐⭐     (spot-check individual chunks)",
        "",
        f"Chunks markdown: {(audit_dir / 'chunks.md').absolute()}",
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])
    
    return "\n".join(lines)


def main():
    """Main audit routine."""
    if len(sys.argv) < 2:
        print("Usage: python inspect_chunks.py <path_to_pdf>")
        print("Example: python inspect_chunks.py data/uploads/paper.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print(f"Auditing chunks for PDF: {pdf_path}")
    print(f"Cache directory: {settings.cache_dir}")
    print(f"Audit directory: {settings.audit_dir}")
    print()
    
    # Initialize extractor
    cache_dir = Path(settings.cache_dir)
    extractor = DoclingExtractor(cache_dir=cache_dir)
    
    # Load or extract document
    doc = extractor.extract(pdf_path)
    print("✓ Document loaded/extracted")
    print()
    
    # Initialize chunker
    chunker = DoclingChunker(cache_dir=cache_dir)
    
    # Chunk document
    chunks_data = chunker.chunk_document(
        doc=doc,
        document_name=pdf_path.stem,
    )
    print(f"✓ Document chunked into {len(chunks_data)} chunks")
    print()
    
    # Load statistics
    stats_path = cache_dir / pdf_path.stem / "chunks" / "statistics.json"
    with open(stats_path, "r") as f:
        statistics = json.load(f)
    
    # Create audit directory
    audit_base = Path(settings.audit_dir)
    audit_base.mkdir(parents=True, exist_ok=True)
    
    audit_dir = audit_base / f"audit_chunks_{pdf_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    audit_dir.mkdir(exist_ok=True)
    
    print(f"Creating audit directory: {audit_dir}")
    print()
    
    # Convert chunks to dicts for report
    chunks_dicts = []
    for chunk in chunks_data:
        chunk_dict = {
            "chunk_id": chunk.chunk_id,
            "index": chunk.index,
            "content": chunk.content,
            "embedding_text": chunk.embedding_text,
            "metadata": {
                "doc_items": chunk.metadata.doc_items,
                "headings": chunk.metadata.headings,
                "captions": chunk.metadata.captions,
                "page_numbers": chunk.metadata.page_numbers,
                "bboxes": chunk.metadata.bboxes,
            },
            "token_count": chunk.token_count,
        }
        chunks_dicts.append(chunk_dict)
    
    # Generate chunks.md (MOST IMPORTANT)
    print("Generating chunks.md...")
    chunks_md = generate_chunks_markdown(chunks_dicts)
    chunks_md_path = audit_dir / "chunks.md"
    with open(chunks_md_path, "w", encoding="utf-8") as f:
        f.write(chunks_md)
    print(f"✓ chunks.md ({len(chunks_md)} characters)")
    
    # Generate audit report
    print("Generating REPORT.txt...")
    report = generate_audit_report(pdf_path, chunks_dicts, statistics, audit_dir)
    report_path = audit_dir / "REPORT.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("✓ REPORT.txt")
    
    # Copy statistics
    stats_dest = audit_dir / "statistics.json"
    shutil.copy(stats_path, stats_dest)
    print("✓ statistics.json")
    
    # Copy individual chunk files
    chunks_src_dir = cache_dir / pdf_path.stem / "chunks"
    chunks_dest_dir = audit_dir / "chunks"
    chunks_dest_dir.mkdir(exist_ok=True)
    
    for chunk_file in chunks_src_dir.glob("chunk_*.json"):
        shutil.copy(chunk_file, chunks_dest_dir / chunk_file.name)
    
    print(f"✓ chunks/ ({len(chunks_data)} files)")
    print()
    
    # Print report to console
    print(report)
    print()
    
    print("=" * 80)
    print("AUDIT COMPLETE!")
    print("=" * 80)
    print()
    print(f"📁 Audit directory: {audit_dir.absolute()}")
    print()
    print("NEXT STEPS:")
    print(f"1. Open {audit_dir / 'chunks.md'} ⭐⭐⭐⭐⭐ (MOST IMPORTANT)")
    print(f"2. Read {audit_dir / 'REPORT.txt'}")
    print(f"3. Review statistics.json for detailed metrics")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
