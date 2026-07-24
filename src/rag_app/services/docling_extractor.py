"""
Capa de extracción de PDFs con Docling. DoclingDocument es la fuente de
verdad (no el Markdown exportado); se preserva toda la estructura
(jerarquía, bounding boxes, provenance) para el chunking downstream.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from docling.backend.docling_parse_backend import DoclingParseDocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.document import DoclingDocument
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    TableStructureOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import ContentLayer, DocItemLabel

logger = logging.getLogger(__name__)


@dataclass
class BBoxData:
    """Bounding box with all coordinates."""
    left: float
    top: float
    right: float
    bottom: float
    page: int | None = None


@dataclass
class ProvenanceData:
    """Complete provenance information."""
    page_no: int | None
    bbox: BBoxData | None
    
    @classmethod
    def from_prov_item(cls, prov_item) -> "ProvenanceData":
        """Extract from ProvenanceItem."""
        page_no = None
        bbox_data = None
        
        if hasattr(prov_item, "page_no"):
            page_no = prov_item.page_no

        
        if hasattr(prov_item, "bbox") and prov_item.bbox:
            bbox = prov_item.bbox
            bbox_data = BBoxData(
                left=bbox.l if hasattr(bbox, "l") else 0,
                top=bbox.t if hasattr(bbox, "t") else 0,
                right=bbox.r if hasattr(bbox, "r") else 0,
                bottom=bbox.b if hasattr(bbox, "b") else 0,
                page=page_no,
            )
        
        return cls(page_no=page_no, bbox=bbox_data)


@dataclass
class NodeData:
    """Nodo del árbol del documento, aplanado desde iterate_items()."""

    node_id: str
    node_type: str
    label: str | None
    parent_cref: str | None
    children_crefs: list[str] = field(default_factory=list)
    text: str | None = None
    text_length: int = 0
    page_numbers: list[int] = field(default_factory=list)
    bboxes: list[BBoxData] = field(default_factory=list)
    self_ref: str | None = None
    content_layer: str | None = None
    level: int = 0
    cref: str | None = None


@dataclass
class FigureData:
    """Rich figure metadata."""
    figure_id: str
    index: int
    caption: str | None
    image_path: str | None
    parent_cref: str | None
    self_ref: str | None = None
    children_crefs: list[str] = field(default_factory=list)
    provenance: list[ProvenanceData] = field(default_factory=list)
    has_image: bool = False


@dataclass
class TableData:
    """Rich table metadata."""
    table_id: str
    index: int
    caption: str | None
    markdown: str
    html: str | None
    num_rows: int
    num_cols: int
    parent_cref: str | None
    children_crefs: list[str] = field(default_factory=list)
    provenance: list[ProvenanceData] = field(default_factory=list)


@dataclass
class ExtractionMetadata:
    """Extraction process metadata."""
    source_file: str
    extraction_timestamp: str
    backend: str
    ocr_enabled: bool
    table_structure_enabled: bool
    picture_images_enabled: bool
    page_images_enabled: bool
    num_pages: int
    num_tables: int
    num_pictures: int
    num_texts: int
    num_nodes: int
    file_size_bytes: int
    processing_time_seconds: float


class DoclingExtractor:
    """Extrae y cachea la estructura completa de un PDF (árbol, bounding
    boxes, figuras, tablas) para no tener que releerlo nunca."""

    def __init__(
        self,
        cache_dir: Path,
        enable_ocr: bool = True,
        enable_table_structure: bool = True,
        generate_picture_images: bool = True,
        generate_page_images: bool = False,
    ):
        """Initialize with explicit configuration."""
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        pipeline_options = PdfPipelineOptions(
            do_ocr=enable_ocr,
            do_table_structure=enable_table_structure,
            table_structure_options=TableStructureOptions(
                do_cell_matching=True,
                mode=TableFormerMode.ACCURATE,
            ),
            generate_page_images=generate_page_images,
            generate_picture_images=generate_picture_images,
        )
        
        format_options = {
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
                backend=DoclingParseDocumentBackend,
            )
        }
        
        self.converter = DocumentConverter(format_options=format_options)
        self.generate_page_images = generate_page_images
        
        # Guardar flags reales para metadata.json
        self.enable_ocr = enable_ocr
        self.enable_table_structure = enable_table_structure
        self.generate_picture_images = generate_picture_images
        
        logger.info(
            f"DoclingExtractor initialized: backend=DoclingParse, "
            f"ocr={enable_ocr}, tables={enable_table_structure}, "
            f"pictures={generate_picture_images}, pages={generate_page_images}"
        )
    
    def _fix_running_headers(self, doc) -> int:
        """
        Detecta section_headers cuyo texto se repite idéntico en 2+ páginas
        (running header/footer mal clasificado por el layout model) y los
        reclasifica como page_header + furniture, para que el chunker deje
        de tratarlos como headings de sección real.
        """
        from collections import defaultdict

        text_to_items = defaultdict(list)
        for item in doc.texts:
            if item.label == DocItemLabel.SECTION_HEADER:
                text_to_items[item.text.strip()].append(item)

        fixed = 0
        for text, items in text_to_items.items():
            pages = {p.page_no for it in items for p in it.prov}
            if len(pages) >= 2:
                for it in items:
                    it.label = DocItemLabel.PAGE_HEADER
                    it.content_layer = ContentLayer.FURNITURE
                fixed += len(items)
                logger.info(
                    f"Reclasificado como page_header ({len(items)}x, "
                    f"{len(pages)} páginas): {text[:60]!r}"
                )
        return fixed
    
    def extract(
        self,
        pdf_path: Path,
        force_reprocess: bool = False,
    ) -> DoclingDocument:
        """Extract document and cache all artifacts."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        doc_cache_dir = self._get_cache_dir(pdf_path)
        
        if not force_reprocess and self._is_cached(doc_cache_dir):
            logger.info(f"Loading cached: {pdf_path.name}")
            return self._load_from_cache(doc_cache_dir)
        
        logger.info(f"Processing: {pdf_path.name}")
        start_time = datetime.now()
        
        result = self.converter.convert(pdf_path)
        doc = result.document
        
        n_fixed = self._fix_running_headers(doc)
        logger.info(f"Headers corregidos: {n_fixed}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Converted in {processing_time:.2f}s")
        
        self._cache_document(doc, pdf_path, doc_cache_dir, processing_time)
        
        return doc

    
    def _cache_document(
        self,
        doc: DoclingDocument,
        pdf_path: Path,
        cache_dir: Path,
        processing_time: float,
    ) -> None:
        """Cachea todos los artefactos derivados del documento."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Caching to: {cache_dir}")

        doc_json_path = cache_dir / "document.json"
        doc.save_as_json(filename=doc_json_path, indent=2)
        logger.info("✓ document.json")

        markdown_path = cache_dir / "markdown.md"
        doc.save_as_markdown(filename=markdown_path)
        logger.info("✓ markdown.md")

        figures = self._extract_figures(doc, cache_dir)
        logger.info(f"✓ figures/ ({len(figures)} figures)")

        tables = self._extract_tables(doc, cache_dir)
        logger.info(f"✓ tables/ ({len(tables)} tables)")

        if self.generate_page_images:
            self._extract_page_images(doc, cache_dir)
            logger.info(f"✓ pages/ ({len(doc.pages)} pages)")

    
    def _extract_all_nodes(self, doc: DoclingDocument) -> list[NodeData]:
        """Aplana el árbol completo del documento vía iterate_items()."""
        nodes = []
        node_counter = 0

        for item, level in doc.iterate_items():
            node_counter += 1
            node_id = f"node_{node_counter:05d}"

            node_type = type(item).__name__
            label = str(item.label) if hasattr(item, "label") else None

            parent_cref = None
            if hasattr(item, "parent") and item.parent:
                if hasattr(item.parent, "cref"):
                    parent_cref = item.parent.cref
            
            children_crefs = []
            if hasattr(item, "children") and item.children:
                for child in item.children:
                    if hasattr(child, "cref"):
                        children_crefs.append(child.cref)
            
            text = None
            text_length = 0
            if hasattr(item, "text"):
                text = str(item.text)
                text_length = len(text)

            page_numbers = []
            bboxes = []
            if hasattr(item, "prov") and item.prov:
                for prov_item in item.prov:
                    prov_data = ProvenanceData.from_prov_item(prov_item)
                    if prov_data.page_no is not None:
                        page_numbers.append(prov_data.page_no)
                    if prov_data.bbox:
                        bboxes.append(prov_data.bbox)

            self_ref = getattr(item, "self_ref", None)
            content_layer = None
            if hasattr(item, "content_layer"):
                content_layer = str(item.content_layer)

            cref = getattr(item, "cref", None) if hasattr(item, "cref") else None
            
            node = NodeData(
                node_id=node_id,
                node_type=node_type,
                label=label,
                parent_cref=parent_cref,
                children_crefs=children_crefs,
                text=text,
                text_length=text_length,
                page_numbers=page_numbers,
                bboxes=bboxes,
                self_ref=self_ref,
                content_layer=content_layer,
                level=level,
                cref=cref,
            )
            nodes.append(node)
        
        return nodes

    
    def _build_document_tree(self, nodes: list[NodeData]) -> dict[str, Any]:
        """Reconstruye el árbol jerárquico desde la lista plana de nodos."""
        nodes_by_cref = {}
        for node in nodes:
            if node.cref:
                nodes_by_cref[node.cref] = node
            if node.self_ref:
                nodes_by_cref[node.self_ref] = node

        tree_nodes = []
        for node in nodes:
            tree_node = {
                "node_id": node.node_id,
                "type": node.node_type,
                "label": node.label,
                "level": node.level,
                "text_preview": node.text[:100] if node.text else None,
                "page_numbers": node.page_numbers,
                "has_bbox": len(node.bboxes) > 0,
                "content_layer": node.content_layer,
                "children_count": len(node.children_crefs),
            }
            tree_nodes.append(tree_node)

        label_counts = {}
        for node in nodes:
            if node.label:
                label_counts[node.label] = label_counts.get(node.label, 0) + 1
        
        return {
            "total_nodes": len(nodes),
            "label_distribution": label_counts,
            "tree": tree_nodes,
        }
    
    def _resolve_text_reference(self, ref_item, doc: DoclingDocument) -> str | None:
        """Docling no expone una API oficial de resolución de RefItem, así
        que se busca a mano el TextItem cuyo self_ref coincide con el cref."""
        if not ref_item or not hasattr(ref_item, "cref"):
            return None

        cref = ref_item.cref
        for text_item in doc.texts:
            if hasattr(text_item, "self_ref") and text_item.self_ref == cref:
                if hasattr(text_item, "text"):
                    return str(text_item.text)

        return f"[ref:{cref}]"

    def _extract_figures(
        self,
        doc: DoclingDocument,
        cache_dir: Path,
    ) -> list[FigureData]:
        figures_dir = cache_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        figures = []

        for idx, picture in enumerate(doc.pictures):
            figure_id = f"figure_{idx + 1:03d}"

            image_path = None
            has_image = False
            if hasattr(picture, "image") and picture.image:
                has_image = True
                try:
                    if hasattr(picture.image, "pil_image") and picture.image.pil_image:
                        pil_img = picture.image.pil_image
                        img_filename = f"{figure_id}.png"
                        img_path = figures_dir / img_filename
                        pil_img.save(img_path)
                        image_path = f"figures/{img_filename}"
                except Exception as e:
                    logger.warning(f"Failed to save {figure_id}: {e}")

            caption = None
            if hasattr(picture, "caption") and picture.caption:
                if hasattr(picture.caption, "text"):
                    caption = str(picture.caption.text)
                else:
                    caption = self._resolve_text_reference(picture.caption, doc)

            self_ref = getattr(picture, "self_ref", None)

            parent_cref = None
            if hasattr(picture, "parent") and picture.parent:
                if hasattr(picture.parent, "cref"):
                    parent_cref = picture.parent.cref

            children_crefs = []
            if hasattr(picture, "children") and picture.children:
                for child in picture.children:
                    if hasattr(child, "cref"):
                        children_crefs.append(child.cref)

            provenance = []
            if hasattr(picture, "prov") and picture.prov:
                for prov_item in picture.prov:
                    provenance.append(ProvenanceData.from_prov_item(prov_item))

            figure = FigureData(
                figure_id=figure_id,
                index=idx,
                caption=caption,
                image_path=image_path,
                self_ref=self_ref,
                parent_cref=parent_cref,
                children_crefs=children_crefs,
                provenance=provenance,
                has_image=has_image,
            )
            figures.append(figure)

            figure_json = figures_dir / f"{figure_id}.json"
            with open(figure_json, "w", encoding="utf-8") as f:
                json.dump(asdict(figure), f, indent=2, ensure_ascii=False)

        return figures

    def _extract_tables(
        self,
        doc: DoclingDocument,
        cache_dir: Path,
    ) -> list[TableData]:
        tables_dir = cache_dir / "tables"
        tables_dir.mkdir(exist_ok=True)

        tables = []

        for idx, table in enumerate(doc.tables):
            table_id = f"table_{idx + 1:03d}"

            markdown = table.export_to_markdown()

            md_file = tables_dir / f"{table_id}.md"
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(markdown)

            html = None
            try:
                if hasattr(table, "export_to_html"):
                    html = table.export_to_html()
                    html_file = tables_dir / f"{table_id}.html"
                    with open(html_file, "w", encoding="utf-8") as f:
                        f.write(html)
            except Exception:
                pass

            num_rows = 0
            num_cols = 0
            if hasattr(table, "data") and table.data:
                if hasattr(table.data, "grid") and table.data.grid:
                    grid = table.data.grid
                    num_rows = len(grid)
                    if num_rows > 0:
                        num_cols = len(grid[0]) if grid[0] else 0

            caption = None
            if hasattr(table, "caption") and table.caption:
                if hasattr(table.caption, "text"):
                    caption = str(table.caption.text)
                else:
                    caption = self._resolve_text_reference(table.caption, doc)

            parent_cref = None
            if hasattr(table, "parent") and table.parent:
                if hasattr(table.parent, "cref"):
                    parent_cref = table.parent.cref

            children_crefs = []
            if hasattr(table, "children") and table.children:
                for child in table.children:
                    if hasattr(child, "cref"):
                        children_crefs.append(child.cref)

            provenance = []
            if hasattr(table, "prov") and table.prov:
                for prov_item in table.prov:
                    provenance.append(ProvenanceData.from_prov_item(prov_item))
            
            table_data = TableData(
                table_id=table_id,
                index=idx,
                caption=caption,
                markdown=markdown,
                html=html,
                num_rows=num_rows,
                num_cols=num_cols,
                parent_cref=parent_cref,
                children_crefs=children_crefs,
                provenance=provenance,
            )
            tables.append(table_data)

            table_json = tables_dir / f"{table_id}.json"
            with open(table_json, "w", encoding="utf-8") as f:
                json.dump(asdict(table_data), f, indent=2, ensure_ascii=False)

        return tables

    def _extract_page_images(
        self,
        doc: DoclingDocument,
        cache_dir: Path,
    ) -> None:
        pages_dir = cache_dir / "pages"
        pages_dir.mkdir(exist_ok=True)

        for page_num, page in doc.pages.items():
            if hasattr(page, "image") and page.image:
                try:
                    if hasattr(page.image, "pil_image") and page.image.pil_image:
                        pil_img = page.image.pil_image
                        img_filename = f"page_{page_num:03d}.png"
                        img_path = pages_dir / img_filename
                        pil_img.save(img_path)
                        logger.debug(f"Saved page {page_num} image")
                except Exception as e:
                    logger.warning(f"Failed to save page {page_num}: {e}")
    
    def _get_cache_dir(self, pdf_path: Path) -> Path:
        return self.cache_dir / pdf_path.stem

    def _is_cached(self, cache_dir: Path) -> bool:
        return (cache_dir / "document.json").exists()

    def _load_from_cache(self, cache_dir: Path) -> DoclingDocument:
        doc_json_path = cache_dir / "document.json"
        return DoclingDocument.load_from_json(doc_json_path)

    def load_document(self, document_name: str) -> DoclingDocument | None:
        doc_cache_dir = self.cache_dir / document_name
        if not self._is_cached(doc_cache_dir):
            return None
        return self._load_from_cache(doc_cache_dir)

    def get_figures_index(self, document_name: str) -> dict[str, str]:
        """Mapea self_ref (ej. "#/pictures/0") a image_path — el chunker
        usa esto para vincular cada chunk con su figura asociada."""
        doc_cache_dir = self.cache_dir / document_name
        figures_dir = doc_cache_dir / "figures"

        if not figures_dir.exists():
            return {}

        figures_index = {}
        for figure_json in sorted(figures_dir.glob("figure_*.json")):
            with open(figure_json, encoding="utf-8") as f:
                figure_data = json.load(f)
                self_ref = figure_data.get("self_ref")
                image_path = figure_data.get("image_path")
                if self_ref and image_path:
                    figures_index[self_ref] = image_path

        return figures_index

    def list_cached_documents(self) -> list[str]:
        if not self.cache_dir.exists():
            return []

        cached = []
        for item in self.cache_dir.iterdir():
            if item.is_dir() and self._is_cached(item):
                cached.append(item.name)

        return sorted(cached)

    def get_cached_artifacts(self, pdf_path: Path) -> dict[str, Any] | None:
        cache_dir = self._get_cache_dir(pdf_path)
        if not self._is_cached(cache_dir):
            return None

        artifacts = {}

        for json_file in ["metadata.json", "nodes.json", "layout.json"]:
            json_path = cache_dir / json_file
            if json_path.exists():
                with open(json_path, encoding="utf-8") as f:
                    artifacts[json_file.replace(".json", "")] = json.load(f)

        md_path = cache_dir / "markdown.md"
        if md_path.exists():
            with open(md_path, encoding="utf-8") as f:
                artifacts["markdown"] = f.read()

        figures_dir = cache_dir / "figures"
        if figures_dir.exists():
            artifacts["figures"] = [str(p) for p in figures_dir.glob("*.json")]

        tables_dir = cache_dir / "tables"
        if tables_dir.exists():
            artifacts["tables"] = [str(p) for p in tables_dir.glob("*.json")]

        pages_dir = cache_dir / "pages"
        if pages_dir.exists():
            artifacts["pages"] = [str(p) for p in pages_dir.glob("*.png")]

        return artifacts
