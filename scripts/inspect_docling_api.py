"""
Docling API Deep Introspection

This script performs deep introspection of the Docling API to understand:
- Actual object structures (not assumptions)
- Available methods and attributes
- Document traversal mechanisms
- Proper API usage patterns

Purpose: Replace assumptions with actual API knowledge.
"""

import sys
import inspect
from pathlib import Path
from typing import Any, Dict, List, Set
from pprint import pprint

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from docling.document_converter import DocumentConverter
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.document_converter import PdfFormatOption


def print_separator(title: str = "", char: str = "="):
    """Print a visual separator."""
    if title:
        print(f"\n{char * 80}")
        print(f"{title.center(80)}")
        print(f"{char * 80}\n")
    else:
        print(f"{char * 80}\n")


def inspect_object(obj: Any, name: str) -> Dict[str, Any]:
    """Deep introspection of an object."""
    info = {
        "type": type(obj).__name__,
        "module": type(obj).__module__,
        "public_attributes": [],
        "public_methods": [],
        "properties": [],
        "special_methods": [],
    }
    
    # Get all members
    members = inspect.getmembers(obj)
    
    for member_name, member_value in members:
        # Skip private members (but note special methods)
        if member_name.startswith("_"):
            if member_name.startswith("__") and member_name.endswith("__"):
                info["special_methods"].append(member_name)
            continue
        
        # Check if it's a method
        if callable(member_value):
            # Get method signature if possible
            try:
                sig = inspect.signature(member_value)
                info["public_methods"].append(f"{member_name}{sig}")
            except (ValueError, TypeError):
                info["public_methods"].append(f"{member_name}(...)")
        # Check if it's a property
        elif isinstance(inspect.getattr_static(type(obj), member_name, None), property):
            info["properties"].append(f"{member_name}: {type(member_value).__name__}")
        # Regular attribute
        else:
            attr_type = type(member_value).__name__
            # For collections, show size
            if isinstance(member_value, (list, tuple, set)):
                attr_type = f"{attr_type}[{len(member_value)}]"
            elif isinstance(member_value, dict):
                attr_type = f"{attr_type}[{len(member_value)} keys]"
            info["public_attributes"].append(f"{member_name}: {attr_type}")
    
    return info


def print_object_info(info: Dict[str, Any], name: str):
    """Print object introspection info."""
    print_separator(name, "=")
    print(f"Type: {info['type']}")
    print(f"Module: {info['module']}")
    
    if info["properties"]:
        print(f"\nProperties ({len(info['properties'])}):")
        for prop in sorted(info["properties"]):
            print(f"  • {prop}")
    
    if info["public_attributes"]:
        print(f"\nPublic Attributes ({len(info['public_attributes'])}):")
        for attr in sorted(info["public_attributes"]):
            print(f"  • {attr}")
    
    if info["public_methods"]:
        print(f"\nPublic Methods ({len(info['public_methods'])}):")
        for method in sorted(info["public_methods"]):
            print(f"  • {method}")
    
    if info["special_methods"]:
        print(f"\nSpecial Methods ({len(info['special_methods'])}):")
        print(f"  {', '.join(sorted(info['special_methods']))}")


def explore_document_structure(doc) -> None:
    """Explore the actual document structure using available APIs."""
    print_separator("DOCUMENT STRUCTURE EXPLORATION", "=")
    
    # Try different ways to access document content
    print("Exploring document access patterns...\n")
    
    # 1. Check main document attributes
    print("─" * 80)
    print("Main Document Attributes:")
    print("─" * 80)
    
    important_attrs = [
        "name", "body", "main_text", "tables", "pictures", "pages",
        "metadata", "furniture", "children", "nodes", "items", "texts",
        "groups", "origin"
    ]
    
    for attr in important_attrs:
        if hasattr(doc, attr):
            value = getattr(doc, attr)
            value_type = type(value).__name__
            if isinstance(value, (list, tuple)):
                print(f"✓ doc.{attr}: {value_type}[{len(value)}]")
                if len(value) > 0:
                    print(f"    First item type: {type(value[0]).__name__}")
            elif isinstance(value, dict):
                print(f"✓ doc.{attr}: {value_type} with {len(value)} keys")
            else:
                print(f"✓ doc.{attr}: {value_type}")
        else:
            print(f"✗ doc.{attr}: not found")
    
    # 2. Try to find traversal methods
    print("\n" + "─" * 80)
    print("Document Traversal Methods:")
    print("─" * 80)
    
    traversal_methods = [
        "iterate_items", "walk", "traverse", "iter_items", 
        "get_children", "get_all_nodes", "iter"
    ]
    
    for method in traversal_methods:
        if hasattr(doc, method):
            print(f"✓ doc.{method}() exists")
        else:
            print(f"✗ doc.{method}() not found")
    
    # 3. Test iterate_items() method
    print("\n" + "─" * 80)
    print("Testing doc.iterate_items():")
    print("─" * 80)
    
    if hasattr(doc, "iterate_items"):
        try:
            # Get signature
            sig = inspect.signature(doc.iterate_items)
            print(f"Signature: iterate_items{sig}")
            print()
            
            # Iterate through document
            print("Document structure via iterate_items():")
            item_count = 0
            label_counts = {}
            
            for item, level in doc.iterate_items():
                item_count += 1
                item_type = type(item).__name__
                
                # Track labels
                if hasattr(item, "label"):
                    label = str(item.label)
                    label_counts[label] = label_counts.get(label, 0) + 1
                else:
                    label = "no_label"
                
                # Show first 20 items
                if item_count <= 20:
                    indent = "  " * level
                    
                    # Get text preview
                    text_preview = ""
                    if hasattr(item, "text"):
                        text = str(item.text)
                        text_preview = f": {text[:60]}..." if len(text) > 60 else f": {text}"
                    
                    print(f"{indent}[{level}] {item_type} ({label}){text_preview}")
            
            print(f"\n... Total items: {item_count}")
            print(f"\nLabel distribution:")
            for label, count in sorted(label_counts.items(), key=lambda x: -x[1]):
                print(f"  {label}: {count}")
                
        except Exception as e:
            print(f"Error using iterate_items(): {e}")
            import traceback
            traceback.print_exc()
    
    # 4. Explore body
    print("\n" + "─" * 80)
    print("Exploring doc.body:")
    print("─" * 80)
    
    if hasattr(doc, "body"):
        body = doc.body
        print(f"Type: {type(body).__name__}")
        body_info = inspect_object(body, "Body")
        
        # Show only key attributes
        if body_info["public_attributes"]:
            print(f"\nKey attributes:")
            for attr in body_info["public_attributes"][:10]:
                print(f"  • {attr}")


def explore_item_hierarchy(doc) -> None:
    """Try to understand parent-child relationships."""
    print_separator("ITEM HIERARCHY EXPLORATION", "=")
    
    # Check if items have parent/child references
    test_items = []
    
    if hasattr(doc, "tables") and doc.tables:
        test_items.append(("Table", doc.tables[0]))
    
    if hasattr(doc, "pictures") and doc.pictures:
        test_items.append(("Picture", doc.pictures[0]))
    
    for item_type, item in test_items:
        print(f"\n{item_type} item hierarchy attributes:")
        print("─" * 80)
        
        hierarchy_attrs = [
            "parent", "parent_id", "parent_ref", "children", 
            "siblings", "ancestors", "section", "heading"
        ]
        
        for attr in hierarchy_attrs:
            if hasattr(item, attr):
                value = getattr(item, attr)
                print(f"✓ {attr}: {type(value).__name__}")
                if value is not None and not callable(value):
                    print(f"    Value: {str(value)[:100]}")
            else:
                print(f"✗ {attr}: not found")


def explore_pages(doc) -> None:
    """Explore page structure."""
    print_separator("PAGES EXPLORATION", "=")
    
    if not hasattr(doc, "pages"):
        print("No 'pages' attribute found")
        return
    
    pages = doc.pages
    print(f"Pages type: {type(pages).__name__}")
    print(f"Number of pages: {len(pages) if hasattr(pages, '__len__') else 'N/A'}")
    
    if pages and len(pages) > 0:
        # Pages is a dict, get first key
        page_keys = list(pages.keys())
        first_page_key = page_keys[0]
        print(f"\nPage keys (first 10): {page_keys[:10]}")
        print(f"\nInspecting page {first_page_key}:")
        page = pages[first_page_key]
        page_info = inspect_object(page, "Page")
        print_object_info(page_info, "Page Object")


def explore_pictures(doc) -> None:
    """Deep exploration of picture items."""
    print_separator("PICTURES EXPLORATION", "=")
    
    if not hasattr(doc, "pictures") or not doc.pictures:
        print("No pictures found")
        return
    
    print(f"Total pictures: {len(doc.pictures)}")
    
    # Inspect first picture in detail
    picture = doc.pictures[0]
    print("\n" + "─" * 80)
    print("First Picture Detailed Inspection:")
    print("─" * 80)
    
    picture_info = inspect_object(picture, "PictureItem")
    print_object_info(picture_info, "PictureItem")
    
    # Try to understand provenance
    print("\n" + "─" * 80)
    print("Provenance (prov) Exploration:")
    print("─" * 80)
    
    if hasattr(picture, "prov") and picture.prov:
        print(f"prov type: {type(picture.prov).__name__}")
        print(f"prov length: {len(picture.prov) if hasattr(picture.prov, '__len__') else 'N/A'}")
        
        if hasattr(picture.prov, "__iter__"):
            for i, prov_item in enumerate(picture.prov):
                print(f"\nprov[{i}]:")
                prov_info = inspect_object(prov_item, f"ProvenanceItem[{i}]")
                print(f"  Type: {prov_info['type']}")
                if prov_info['public_attributes']:
                    print(f"  Attributes:")
                    for attr in prov_info['public_attributes']:
                        print(f"    • {attr}")
    
    # Try to get image
    print("\n" + "─" * 80)
    print("Image Extraction Test:")
    print("─" * 80)
    
    try:
        # Try get_image with backend
        if hasattr(picture, "get_image"):
            print("✓ get_image() method exists")
            img = picture.get_image(doc.backend)
            if img:
                print(f"  Image obtained: {type(img).__name__}")
                if hasattr(img, "size"):
                    print(f"  Size: {img.size}")
        else:
            print("✗ get_image() method not found")
            
            # Try alternative methods
            alt_methods = ["image", "get_pil_image", "to_image", "as_image"]
            for method in alt_methods:
                if hasattr(picture, method):
                    print(f"  Alternative found: {method}()")
    except Exception as e:
        print(f"Error extracting image: {e}")


def explore_tables(doc) -> None:
    """Deep exploration of table items."""
    print_separator("TABLES EXPLORATION", "=")
    
    if not hasattr(doc, "tables") or not doc.tables:
        print("No tables found")
        return
    
    print(f"Total tables: {len(doc.tables)}")
    
    # Inspect first table in detail
    table = doc.tables[0]
    print("\n" + "─" * 80)
    print("First Table Detailed Inspection:")
    print("─" * 80)
    
    table_info = inspect_object(table, "TableItem")
    print_object_info(table_info, "TableItem")
    
    # Test export methods
    print("\n" + "─" * 80)
    print("Table Export Methods:")
    print("─" * 80)
    
    export_methods = [
        "export_to_markdown", "export_to_html", "to_markdown", 
        "to_html", "export_to_dataframe", "to_dataframe"
    ]
    
    for method in export_methods:
        if hasattr(table, method):
            print(f"✓ {method}() exists")
            try:
                result = getattr(table, method)()
                print(f"    Returns: {type(result).__name__}")
                if isinstance(result, str):
                    print(f"    Length: {len(result)} chars")
            except Exception as e:
                print(f"    Error calling: {e}")
        else:
            print(f"✗ {method}() not found")
    
    # Check data structure
    print("\n" + "─" * 80)
    print("Table Data Structure:")
    print("─" * 80)
    
    data_attrs = ["data", "grid", "cells", "rows", "columns", "table_data"]
    
    for attr in data_attrs:
        if hasattr(table, attr):
            value = getattr(table, attr)
            print(f"✓ {attr}: {type(value).__name__}")
            if hasattr(value, "__len__"):
                print(f"    Length: {len(value)}")
        else:
            print(f"✗ {attr}: not found")


def explore_bounding_boxes(doc) -> None:
    """Explore bounding box representation."""
    print_separator("BOUNDING BOX EXPLORATION", "=")
    
    # Try to find items with bounding boxes
    items_with_bbox = []
    
    if hasattr(doc, "pictures") and doc.pictures:
        for pic in doc.pictures[:3]:
            if hasattr(pic, "prov") and pic.prov:
                for prov_item in pic.prov:
                    if hasattr(prov_item, "bbox"):
                        items_with_bbox.append(("Picture", prov_item.bbox))
                        break
    
    if hasattr(doc, "tables") and doc.tables:
        for table in doc.tables[:3]:
            if hasattr(table, "prov") and table.prov:
                for prov_item in table.prov:
                    if hasattr(prov_item, "bbox"):
                        items_with_bbox.append(("Table", prov_item.bbox))
                        break
    
    if not items_with_bbox:
        print("No bounding boxes found")
        return
    
    print(f"Found {len(items_with_bbox)} items with bounding boxes\n")
    
    # Inspect first bbox
    item_type, bbox = items_with_bbox[0]
    print(f"Inspecting BBox from {item_type}:")
    print("─" * 80)
    
    bbox_info = inspect_object(bbox, "BoundingBox")
    print_object_info(bbox_info, "BoundingBox")
    
    # Try to access coordinates
    print("\nCoordinate Access:")
    coord_attrs = ["l", "t", "r", "b", "left", "top", "right", "bottom", "x", "y", "width", "height"]
    
    for attr in coord_attrs:
        if hasattr(bbox, attr):
            value = getattr(bbox, attr)
            print(f"  ✓ bbox.{attr} = {value}")


def explore_export_methods(doc) -> None:
    """Explore all export methods."""
    print_separator("EXPORT METHODS EXPLORATION", "=")
    
    export_methods = [
        "export_to_markdown", "export_to_html", "export_to_dict",
        "export_to_json", "export_to_xml", "export_to_document_tokens",
        "to_markdown", "to_html", "to_dict", "to_json",
    ]
    
    for method in export_methods:
        if hasattr(doc, method):
            print(f"✓ doc.{method}() exists")
            try:
                # Get method signature
                method_obj = getattr(doc, method)
                sig = inspect.signature(method_obj)
                print(f"    Signature: {sig}")
            except Exception as e:
                print(f"    Could not get signature: {e}")
        else:
            print(f"✗ doc.{method}() not found")


def explore_serialization(doc) -> None:
    """Explore serialization/deserialization."""
    print_separator("SERIALIZATION EXPLORATION", "=")
    
    doc_class = type(doc)
    
    print("Instance methods:")
    instance_methods = ["serialize", "to_dict", "to_json", "save", "dump"]
    for method in instance_methods:
        if hasattr(doc, method):
            print(f"  ✓ doc.{method}()")
    
    print("\nClass methods:")
    class_methods = ["from_dict", "from_json", "load", "deserialize", "parse"]
    for method in class_methods:
        if hasattr(doc_class, method):
            print(f"  ✓ {doc_class.__name__}.{method}()")
    
    # Test export_to_dict
    print("\n" + "─" * 80)
    print("Testing export_to_dict():")
    print("─" * 80)
    
    if hasattr(doc, "export_to_dict"):
        try:
            doc_dict = doc.export_to_dict()
            print(f"✓ Successfully exported to dict")
            print(f"  Type: {type(doc_dict).__name__}")
            print(f"  Keys: {list(doc_dict.keys())[:10]}")
            
            # Test reconstruction
            print("\nTesting reconstruction from dict:")
            if hasattr(doc_class, "from_dict"):
                print(f"  ✓ {doc_class.__name__}.from_dict() exists")
            else:
                print(f"  ✗ {doc_class.__name__}.from_dict() not found")
                
                # Check for alternatives
                alt_methods = ["parse_obj", "model_validate", "from_json"]
                for alt in alt_methods:
                    if hasattr(doc_class, alt):
                        print(f"  Alternative: {doc_class.__name__}.{alt}()")
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main exploration routine."""
    if len(sys.argv) < 2:
        print("Usage: python inspect_docling_api.py <path_to_pdf>")
        print("Example: python inspect_docling_api.py data/uploads/paper.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"Error: File not found: {pdf_path}")
        sys.exit(1)
    
    print_separator("DOCLING API DEEP INTROSPECTION", "=")
    print(f"PDF: {pdf_path.name}")
    print(f"Starting introspection...\n")
    
    # Initialize converter with minimal config
    pipeline_options = PdfPipelineOptions()
    format_options = {
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options,
            backend=DoclingParseDocumentBackend,
        )
    }
    converter = DocumentConverter(format_options=format_options)
    
    print("Converting document (this may take a moment)...")
    result = converter.convert(pdf_path)
    doc = result.document
    print("✓ Conversion complete\n")
    
    # Run all explorations
    
    # 1. Inspect main DoclingDocument object
    doc_info = inspect_object(doc, "DoclingDocument")
    print_object_info(doc_info, "DoclingDocument")
    
    # 2. Explore document structure
    explore_document_structure(doc)
    
    # 3. Explore hierarchy
    explore_item_hierarchy(doc)
    
    # 4. Explore pages
    explore_pages(doc)
    
    # 5. Explore pictures
    explore_pictures(doc)
    
    # 6. Explore tables
    explore_tables(doc)
    
    # 7. Explore bounding boxes
    explore_bounding_boxes(doc)
    
    # 8. Explore export methods
    explore_export_methods(doc)
    
    # 9. Explore serialization
    explore_serialization(doc)
    
    print_separator("INTROSPECTION COMPLETE", "=")
    print("Review the output above to understand the actual Docling API.")
    print("Use this information to eliminate assumptions in the extractor.")


if __name__ == "__main__":
    main()
