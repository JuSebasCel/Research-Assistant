"""
Docling Extraction Audit Tool

Audits the quality of PDF extraction before chunking/RAG phase.
Works with the production DoclingExtractor cache structure.

Usage:
    python inspect_docling.py path/to/document.pdf

Output:
    Creates audit_<pdf>_<timestamp>/ with:
    - REPORT.txt (comprehensive audit report)
    - markdown.md (copy of extracted markdown)
    - All cached artifacts for manual inspection
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
from rag_app.core.config import get_settings

settings = get_settings()


def format_bytes(size_bytes: int) -> str:
    """Format bytes to human-readable size."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.2f} TB"


def generate_tree_visualization(layout: Dict[str, Any], max_items: int = 50) -> str:
    """
    Generate ASCII tree from layout structure.
    
    Uses level information from iterate_items() for proper indentation.
    """
    lines = []
    tree = layout.get("tree", [])
    
    for idx, node in enumerate(tree[:max_items]):
        level = node.get("level", 0)
        node_type = node.get("type", "unknown")
        label = node.get("label", "")
        text_preview = node.get("text_preview", "")
        
        # Create indentation based on level
        indent = "  " * level
        prefix = "├─ " if level > 0 else ""
        
        # Format line
        line_parts = [f"{indent}{prefix}{label or node_type}"]
        
        if text_preview:
            line_parts.append(f": {text_preview}")
        
        lines.append("".join(line_parts))
    
    if len(tree) > max_items:
        lines.append(f"  ... and {len(tree) - max_items} more nodes")
    
    return "\n".join(lines)


def generate_audit_report(
    pdf_path: Path,
    artifacts: Dict[str, Any],
    audit_dir: Path,
) -> str:
    """Generate comprehensive audit report."""
    metadata = artifacts.get("metadata", {})
    layout = artifacts.get("layout", {})
    nodes_data = artifacts.get("nodes", [])

    
    report_lines = [
        "=" * 80,
        "DOCLING EXTRACTION AUDIT REPORT",
        "=" * 80,
        "",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"PDF: {pdf_path.name}",
        f"Audit Directory: {audit_dir.absolute()}",
        "",
        "─" * 80,
        "EXTRACTION METADATA",
        "─" * 80,
        "",
        f"Source File: {metadata.get('source_file', 'N/A')}",
        f"File Size: {format_bytes(metadata.get('file_size_bytes', 0))}",
        f"Extraction Time: {metadata.get('processing_time_seconds', 0):.2f}s",
        f"Timestamp: {metadata.get('extraction_timestamp', 'N/A')}",
        "",
        "Configuration:",
        f"  Backend: {metadata.get('backend', 'N/A')}",
        f"  OCR: {metadata.get('ocr_enabled', False)}",
        f"  Table Structure: {metadata.get('table_structure_enabled', False)}",
        f"  Picture Images: {metadata.get('picture_images_enabled', False)}",
        f"  Page Images: {metadata.get('page_images_enabled', False)}",
        "",
        "─" * 80,
        "DOCUMENT STATISTICS",
        "─" * 80,
        "",
        f"Total Pages: {metadata.get('num_pages', 0)}",
        f"Total Nodes: {metadata.get('num_nodes', 0)}",
        f"Total Texts: {metadata.get('num_texts', 0)}",
        f"Total Tables: {metadata.get('num_tables', 0)}",
        f"Total Figures: {metadata.get('num_pictures', 0)}",
        "",
        f"Processing Speed: {metadata.get('num_pages', 1) / max(metadata.get('processing_time_seconds', 1), 0.001):.2f} pages/sec",
        "",
    ]
    
    # Label distribution from layout
    if layout:
        label_dist = layout.get("label_distribution", {})
        if label_dist:
            report_lines.extend([
                "─" * 80,
                "LABEL DISTRIBUTION",
                "─" * 80,
                "",
            ])
            
            # Sort by count descending
            sorted_labels = sorted(label_dist.items(), key=lambda x: -x[1])
            for label, count in sorted_labels:
                report_lines.append(f"  {label:30s}: {count:5d}")
            
            report_lines.append("")
    
    # Figures summary
    figures = artifacts.get("figures", [])
    if figures:
        report_lines.extend([
            "─" * 80,
            f"FIGURES ({len(figures)} found)",
            "─" * 80,
            "",
        ])
        
        # Load and summarize first 5 figures
        for fig_path in figures[:5]:
            try:
                with open(fig_path, "r", encoding="utf-8") as f:
                    fig_data = json.load(f)
                
                fig_id = fig_data.get("figure_id", "unknown")
                caption = fig_data.get("caption", "No caption")
                has_image = fig_data.get("has_image", False)
                provenance = fig_data.get("provenance", [])
                page_no = provenance[0].get("page_no") if provenance else "?"
                
                report_lines.extend([
                    f"Figure: {fig_id}",
                    f"  Page: {page_no}",
                    f"  Caption: {caption[:80]}{'...' if len(caption) > 80 else ''}",
                    f"  Image: {'✓' if has_image else '✗'}",
                    "",
                ])
            except Exception:
                continue
        
        if len(figures) > 5:
            report_lines.append(f"  ... and {len(figures) - 5} more figures")
            report_lines.append("")

    
    # Tables summary
    tables = artifacts.get("tables", [])
    if tables:
        report_lines.extend([
            "─" * 80,
            f"TABLES ({len(tables)} found)",
            "─" * 80,
            "",
        ])
        
        for tbl_path in tables[:5]:
            try:
                with open(tbl_path, "r", encoding="utf-8") as f:
                    tbl_data = json.load(f)
                
                tbl_id = tbl_data.get("table_id", "unknown")
                caption = tbl_data.get("caption", "No caption")
                num_rows = tbl_data.get("num_rows", 0)
                num_cols = tbl_data.get("num_cols", 0)
                provenance = tbl_data.get("provenance", [])
                page_no = provenance[0].get("page_no") if provenance else "?"
                
                report_lines.extend([
                    f"Table: {tbl_id}",
                    f"  Page: {page_no}",
                    f"  Dimensions: {num_rows} rows × {num_cols} cols",
                    f"  Caption: {caption[:80]}{'...' if len(caption) > 80 else ''}",
                    "",
                ])
            except Exception:
                continue
        
        if len(tables) > 5:
            report_lines.append(f"  ... and {len(tables) - 5} more tables")
            report_lines.append("")
    
    # Document tree
    if layout:
        report_lines.extend([
            "─" * 80,
            "DOCUMENT STRUCTURE TREE",
            "─" * 80,
            "",
            generate_tree_visualization(layout, max_items=50),
            "",
        ])
    
    # Quality checks
    report_lines.extend([
        "─" * 80,
        "QUALITY CHECKS",
        "─" * 80,
        "",
    ])
    
    # Check 1: Nodes extracted
    num_nodes = metadata.get("num_nodes", 0)
    if num_nodes == 0:
        report_lines.append("⚠️  Warning: No nodes extracted")
    else:
        report_lines.append(f"✓ Nodes extracted: {num_nodes}")
    
    # Check 2: Tables
    num_tables = metadata.get("num_tables", 0)
    if metadata.get("table_structure_enabled") and num_tables == 0:
        report_lines.append("⚠️  Warning: Table structure enabled but no tables found")
    elif num_tables > 0:
        report_lines.append(f"✓ Tables extracted: {num_tables}")
    
    # Check 3: Figures
    num_figures = metadata.get("num_pictures", 0)
    if metadata.get("picture_images_enabled") and num_figures == 0:
        report_lines.append("⚠️  Warning: Picture images enabled but no figures found")
    elif num_figures > 0:
        figures_with_images = sum(1 for f in figures if "has_image" in str(f))
        report_lines.append(f"✓ Figures extracted: {num_figures}")
    
    # Check 4: Processing time
    processing_time = metadata.get("processing_time_seconds", 0)
    pages = metadata.get("num_pages", 1)
    time_per_page = processing_time / pages if pages > 0 else 0
    if time_per_page > 10:
        report_lines.append(f"⚠️  Slow processing: {time_per_page:.2f}s per page")
    else:
        report_lines.append(f"✓ Processing time: {processing_time:.2f}s ({time_per_page:.2f}s/page)")
    
    # Check 5: Markdown generated
    markdown_path = audit_dir / "markdown.md"
    if markdown_path.exists():
        md_size = markdown_path.stat().st_size
        report_lines.append(f"✓ Markdown generated: {format_bytes(md_size)}")
    else:
        report_lines.append("⚠️  Warning: Markdown not found")
    
    report_lines.append("")

    
    # Generated artifacts
    report_lines.extend([
        "─" * 80,
        "GENERATED ARTIFACTS",
        "─" * 80,
        "",
    ])
    
    artifact_files = [
        ("markdown.md", "⭐⭐⭐⭐⭐", "Primary extraction output"),
        ("document.json", "⭐⭐⭐", "Full DoclingDocument"),
        ("nodes.json", "⭐⭐⭐⭐", "All document nodes"),
        ("layout.json", "⭐⭐⭐", "Hierarchical structure"),
        ("metadata.json", "⭐⭐", "Extraction statistics"),
    ]
    
    for filename, importance, description in artifact_files:
        file_path = audit_dir / filename
        if file_path.exists():
            size = format_bytes(file_path.stat().st_size)
            report_lines.append(f"✓ {filename:20s} {importance} - {description} ({size})")
        else:
            report_lines.append(f"✗ {filename:20s} - NOT FOUND")
    
    # Directories
    report_lines.append("")
    for dirname in ["figures", "tables", "pages"]:
        dir_path = audit_dir / dirname
        if dir_path.exists():
            file_count = len(list(dir_path.iterdir()))
            report_lines.append(f"✓ {dirname + '/':20s} - {file_count} files")
    
    report_lines.append("")
    
    # Manual inspection checklist
    report_lines.extend([
        "─" * 80,
        "MANUAL INSPECTION CHECKLIST",
        "─" * 80,
        "",
        "1. Open markdown.md ⭐⭐⭐⭐⭐ (MOST IMPORTANT)",
        "   - Verify document title is correct",
        "   - Check section headings hierarchy",
        "   - Verify paragraphs are complete and readable",
        "   - Check lists formatting (bullet/numbered)",
        "   - Verify tables are correctly formatted",
        "   - Check equations/formulas rendering",
        "   - Look for missing or duplicated content",
        "   - Verify figure/table captions",
        "",
        "2. Review nodes.json",
        "   - Check total node count matches expectations",
        "   - Verify label distribution makes sense",
        "   - Check hierarchy (parent/children relationships)",
        "   - Verify bounding boxes are present",
        "   - Check content_layer values (BODY, HEADER, etc.)",
        "",
        "3. Inspect figures/ directory",
        "   - Open each figure PNG",
        "   - Verify images are clear and complete",
        "   - Check figure metadata JSON files",
        "   - Verify captions match PDF",
        "   - Check provenance (page numbers, bbox)",
        "",
        "4. Inspect tables/ directory",
        "   - Open table Markdown files",
        "   - Verify table structure is correct",
        "   - Check for merged cells handling",
        "   - Verify captions match PDF",
        "   - Compare with table HTML if available",
        "",
        "5. Compare against original PDF",
        "   - Open original PDF side-by-side",
        "   - Spot-check key sections",
        "   - Verify page count matches",
        "   - Check for missing pages",
        "   - Verify figure/table count",
        "",
        "6. Check for common issues",
        "   - Headers/footers duplicated in text",
        "   - Multi-column layout problems",
        "   - Equation rendering issues",
        "   - Missing references section",
        "   - Garbled or missing text in scanned pages",
        "",
    ])
    
    # Recommended inspection order
    report_lines.extend([
        "=" * 80,
        "RECOMMENDED INSPECTION ORDER",
        "=" * 80,
        "",
        "1. markdown.md           ⭐⭐⭐⭐⭐  (START HERE - most important)",
        "2. REPORT.txt            ⭐⭐⭐⭐   (this file - overview)",
        "3. figures/              ⭐⭐⭐⭐   (visual inspection)",
        "4. tables/               ⭐⭐⭐⭐   (structure validation)",
        "5. nodes.json            ⭐⭐⭐     (detailed structure)",
        "6. layout.json           ⭐⭐⭐     (hierarchy check)",
        "7. document.json         ⭐⭐      (full document - if needed)",
        "",
        f"Markdown location: {(audit_dir / 'markdown.md').absolute()}",
        "",
        "=" * 80,
        "END OF REPORT",
        "=" * 80,
    ])
    
    return "\n".join(report_lines)


def main():
    """Main audit routine."""
    if len(sys.argv) < 2:
        print("Usage: python inspect_docling.py <path_to_pdf>")
        print("Example: python inspect_docling.py data/uploads/paper.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    if not pdf_path.suffix.lower() == ".pdf":
        print(f"Error: Not a PDF file: {pdf_path}")
        sys.exit(1)
    
    print(f"Auditing PDF: {pdf_path}")
    print(f"Cache directory: {settings.cache_dir}")
    print(f"Audit directory: {settings.audit_dir}")
    print()

    
    # Initialize extractor
    cache_dir = Path(settings.cache_dir)
    extractor = DoclingExtractor(
        cache_dir=cache_dir,
        enable_ocr=True,
        enable_table_structure=True,
        generate_picture_images=True,
        generate_page_images=False,
    )
    
    # Check if already cached
    cached_artifacts = extractor.get_cached_artifacts(pdf_path)
    
    if cached_artifacts:
        print("✓ Document already cached. Using existing extraction.")
        print()
    else:
        print("⚙ Document not cached. Extracting...")
        print()
        
        # Extract document
        doc = extractor.extract(pdf_path)
        
        print("✓ Extraction complete!")
        print()
        
        # Reload artifacts
        cached_artifacts = extractor.get_cached_artifacts(pdf_path)
    
    if not cached_artifacts:
        print("Error: Failed to load cached artifacts")
        sys.exit(1)
    
    # Create audit directory in configured location
    audit_base = Path(settings.audit_dir)
    audit_base.mkdir(parents=True, exist_ok=True)
    
    audit_dir = audit_base / f"audit_{pdf_path.stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    audit_dir.mkdir(exist_ok=True)
    
    print(f"Creating audit directory: {audit_dir}")
    print()
    
    # Generate audit report
    report = generate_audit_report(pdf_path, cached_artifacts, audit_dir)
    report_path = audit_dir / "REPORT.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(report)
    print()
    
    # Copy artifacts to audit directory
    print("Copying artifacts to audit directory...")
    
    # Get cache directory
    cache_dir = Path(settings.cache_dir) / pdf_path.stem
    
    # Copy main files
    for filename in ["markdown.md", "document.json", "nodes.json", "layout.json", "metadata.json"]:
        src = cache_dir / filename
        if src.exists():
            dst = audit_dir / filename
            shutil.copy(src, dst)
            print(f"  ✓ {filename}")
    
    # Copy directories
    for dirname in ["figures", "tables", "pages"]:
        src_dir = cache_dir / dirname
        if src_dir.exists() and any(src_dir.iterdir()):
            dst_dir = audit_dir / dirname
            shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)
            file_count = len(list(dst_dir.iterdir()))
            print(f"  ✓ {dirname}/ ({file_count} files)")
    
    print()
    print("=" * 80)
    print("AUDIT COMPLETE!")
    print("=" * 80)
    print()
    print(f"📁 Audit directory: {audit_dir.absolute()}")
    print()
    print("NEXT STEPS:")
    print(f"1. Open {audit_dir / 'markdown.md'} ⭐⭐⭐⭐⭐")
    print(f"2. Read {audit_dir / 'REPORT.txt'}")
    print(f"3. Inspect figures/ and tables/")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
