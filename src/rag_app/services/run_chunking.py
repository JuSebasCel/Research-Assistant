"""
Orchestration Script: PDF Extraction + Chunking Pipeline

Simple orchestrator connecting DoclingExtractor and DoclingChunker.
No business logic - services handle their own cache and processing decisions.

Usage:
    python scripts/run_chunking.py path/to/document.pdf
    python scripts/run_chunking.py --all
    python scripts/run_chunking.py --all --force --verbose
"""

import sys
import argparse
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rag_app.services.docling_extractor import DoclingExtractor
from rag_app.services.docling_chunker import DoclingChunker

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool) -> None:
    """Configure logging."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format='%(message)s'
    )
    if not verbose:
        logging.getLogger("transformers").setLevel(logging.WARNING)
        logging.getLogger("docling").setLevel(logging.WARNING)


def create_services(cache_dir: Path, enable_ocr: bool = False) -> tuple[DoclingExtractor, DoclingChunker]:
    """Initialize extractor and chunker services."""
    extractor = DoclingExtractor(
        cache_dir=cache_dir,
        enable_ocr=enable_ocr,
        enable_table_structure=True,
        generate_picture_images=True,
        generate_page_images=False,
    )
    
    chunker = DoclingChunker(
        cache_dir=cache_dir,
        embedding_model_name="intfloat/multilingual-e5-large",
    )
    
    return extractor, chunker


def process_pdf(
    pdf_path: Path,
    cache_dir: Path,
    force_rechunk: bool,
    enable_ocr: bool = False,
) -> int:
    """Process a single PDF through extraction and chunking."""
    if not pdf_path.exists():
        logger.error(f"PDF not found: {pdf_path}")
        return 1
    
    document_name = pdf_path.stem
    
    logger.info("=" * 80)
    logger.info(f"Document: {pdf_path.name}")
    logger.info("=" * 80)
    
    try:
        # Initialize services
        extractor, chunker = create_services(cache_dir, enable_ocr)
        
        # Extract (service handles cache)
        doc = extractor.extract(pdf_path=pdf_path, force_reprocess=False)
        logger.info("✓ Extraction complete")
        
        # Get figures index for linking chunks to images
        figures_index = extractor.get_figures_index(document_name)
        
        # Chunk (service handles cache)
        chunks = chunker.chunk_document(
            doc=doc,
            document_name=document_name,
            force_rechunk=force_rechunk,
            figures_index=figures_index
        )
        logger.info(f"✓ Generated {len(chunks)} chunks")
        
        # Show output location
        chunks_dir = cache_dir / document_name / "chunks"
        logger.info(f"✓ Saved to {chunks_dir}/")
        logger.info("")
        
        return 0
        
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return 1


def process_all(
    cache_dir: Path,
    force_rechunk: bool,
    enable_ocr: bool = False,
) -> int:
    """Process all cached documents."""
    logger.info("=" * 80)
    logger.info("Processing All Cached Documents")
    logger.info("=" * 80)
    logger.info("")
    
    try:
        # Initialize services
        extractor, chunker = create_services(cache_dir, enable_ocr)
        
        # Get cached documents
        cached_docs = extractor.list_cached_documents()
        
        if not cached_docs:
            logger.info("No cached documents found.")
            logger.info(f"Cache directory: {cache_dir}")
            logger.info("")
            logger.info("Extract a PDF first:")
            logger.info("  python scripts/run_chunking.py path/to/document.pdf")
            return 0
        
        logger.info(f"Found {len(cached_docs)} document(s)\n")
        
        # Process each
        success_count = 0
        for doc_name in cached_docs:
            logger.info(f"Processing: {doc_name}")
            
            try:
                # Load document (from extractor cache)
                doc = extractor.load_document(doc_name)
                if doc is None:
                    logger.error(f"  ✗ Could not load from cache")
                    continue
                
                # Get figures index for linking chunks to images
                figures_index = extractor.get_figures_index(doc_name)
                
                # Chunk (chunker handles its cache)
                chunks = chunker.chunk_document(
                    doc=doc,
                    document_name=doc_name,
                    force_rechunk=force_rechunk,
                    figures_index=figures_index
                )
                
                logger.info(f"  ✓ {len(chunks)} chunks\n")
                success_count += 1
                
            except Exception as e:
                logger.error(f"  ✗ Error: {e}\n")
        
        # Summary
        logger.info("=" * 80)
        logger.info(f"Processed: {success_count}/{len(cached_docs)} documents")
        logger.info("=" * 80)
        logger.info("")
        
        return 0 if success_count == len(cached_docs) else 1
        
    except Exception as e:
        logger.error(f"✗ Error: {e}")
        return 1


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Orchestrate PDF extraction and chunking",
        epilog="""
Examples:
  python scripts/run_chunking.py data/uploads/paper.pdf
  python scripts/run_chunking.py --all
  python scripts/run_chunking.py --all --force
        """
    )
    
    parser.add_argument("pdf_path", nargs="?", type=Path, help="Path to PDF file")
    parser.add_argument("--all", action="store_true", help="Process all cached documents")
    parser.add_argument("--force", action="store_true", help="Force re-chunking")
    parser.add_argument("--ocr", action="store_true", help="Activar OCR (usar solo con PDFs escaneados)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"), help="Cache directory")
    
    args = parser.parse_args()
    
    setup_logging(args.verbose)
    
    # Validate
    if not args.all and not args.pdf_path:
        parser.error("Provide a PDF path or use --all")
    
    if args.all and args.pdf_path:
        logger.warning("--all specified, ignoring PDF path\n")
    
    # Process
    try:
        if args.all:
            return process_all(args.cache_dir, args.force, args.ocr)
        else:
            return process_pdf(args.pdf_path, args.cache_dir, args.force, args.ocr)
    
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user")
        return 130
    
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
