# Docling Extraction Layer - Documentación

## Resumen

El **DoclingExtractor** es una capa de extracción de nivel producción que convierte PDFs en información estructurada lista para chunking y RAG. Usa la API oficial de Docling verificada mediante introspección.

## Principios de Diseño

1. **DoclingDocument es la fuente de verdad** - No Markdown
2. **API verificada** - Cero suposiciones, todo basado en introspección real
3. **Extracción completa** - Preserva TODA la información estructural
4. **No re-lectura** - Extrae una vez, cachea todo
5. **Modular** - Organización lista para producción

## Arquitectura

```
PDF → DocumentConverter → DoclingDocument → Cache Artifacts
                                              ├── document.json (oficial)
                                              ├── markdown.md (exportación)
                                              ├── nodes.json (TODOS los nodos)
                                              ├── layout.json (árbol jerárquico)
                                              ├── metadata.json (estadísticas)
                                              ├── figures/ (PNG + metadata)
                                              ├── tables/ (MD/HTML + metadata)
                                              └── pages/ (imágenes completas)
```

## Estructura de Cache

```
data/cache/{pdf_name}/
├── document.json          # Full DoclingDocument (save_as_json)
├── markdown.md            # Markdown export (save_as_markdown)
├── metadata.json          # Extraction statistics
├── nodes.json             # ALL nodes via iterate_items()
├── layout.json            # Hierarchical tree structure
├── figures/
│   ├── figure_001.png
│   ├── figure_001.json    # Complete metadata
│   ├── figure_002.png
│   ├── figure_002.json
│   └── ...
├── tables/
│   ├── table_001.md       # Markdown export
│   ├── table_001.html     # HTML export (si disponible)
│   ├── table_001.json     # Complete metadata
│   └── ...
└── pages/                 # Solo si generate_page_images=True
    ├── page_001.png
    ├── page_002.png
    └── ...
```

## API Verificada

Toda la implementación está basada en introspección real de Docling:

```python
# DoclingDocument
doc.iterate_items() → Iterable[tuple[NodeItem, int]]  # ✅ Verificado
doc.save_as_json(), load_from_json()                  # ✅ Verificado
doc.body, doc.texts, doc.tables, doc.pictures         # ✅ Verificado
doc.pages  # dict[int, PageItem] NO lista              # ✅ Verificado

# NodeItem
item.parent → RefItem with .cref                       # ✅ Verificado
item.children → list[RefItem with .cref]               # ✅ Verificado
item.label → DocItemLabel                              # ✅ Verificado
item.text → str                                        # ✅ Verificado
item.prov → list[ProvenanceItem]                       # ✅ Verificado

# ProvenanceItem
prov.page_no → int                                     # ✅ Verificado
prov.bbox.l, .t, .r, .b → float                       # ✅ Verificado

# PictureItem
picture.image.pil_image → PIL.Image                    # ✅ Verificado
picture.caption → TextItem | RefItem | None            # ✅ Verificado

# TableItem
table.data.grid → list[list[TableCell]]                # ✅ Verificado
table.export_to_markdown() → str                       # ✅ Verificado
```

## Uso

### Extracción básica

```python
from pathlib import Path
from rag_app.services.docling_extractor import DoclingExtractor
from rag_app.core.config import get_settings

settings = get_settings()
extractor = DoclingExtractor(
    cache_dir=Path(settings.cache_dir),
    enable_ocr=True,
    enable_table_structure=True,
    generate_picture_images=True,
    generate_page_images=False,  # True para RAG multimodal
)

# Extrae y cachea (solo primera vez)
doc = extractor.extract(Path("paper.pdf"))

# Siguiente vez usa cache
doc = extractor.extract(Path("paper.pdf"))  # Instantáneo

# Obtener artefactos sin cargar documento
artifacts = extractor.get_cached_artifacts(Path("paper.pdf"))
```

### Auditoría de extracción

```bash
# Extrae (si necesario) y genera reporte de auditoría
uv run python scripts/inspect_docling.py data/uploads/paper.pdf

# Output: data/audit/audit_paper_YYYYMMDD_HHMMSS/
# - REPORT.txt (reporte completo)
# - markdown.md ⭐⭐⭐⭐⭐ (REVISAR PRIMERO)
# - Todos los artefactos copiados
```

## Información Extraída

### 1. Nodes (nodes.json)

TODOS los nodos del documento vía `iterate_items()`:

```json
{
  "node_id": "node_00123",
  "node_type": "TextItem",
  "label": "paragraph",
  "parent_cref": "#/body",
  "children_crefs": ["#/texts/456"],
  "text": "Contenido completo del párrafo...",
  "text_length": 234,
  "page_numbers": [5, 6],
  "bboxes": [
    {"left": 72.0, "top": 100.0, "right": 540.0, "bottom": 150.0, "page": 5}
  ],
  "content_layer": "BODY",
  "level": 2,
  "self_ref": "#/texts/455"
}
```

### 2. Figures (figures/*.json)

```json
{
  "figure_id": "figure_001",
  "index": 0,
  "caption": "Figure 3. System Architecture",  // Resuelto!
  "image_path": "figures/figure_001.png",
  "has_image": true,
  "parent_cref": "#/body",
  "children_crefs": ["#/texts/789"],
  "provenance": [
    {
      "page_no": 5,
      "bbox": {
        "left": 100.0,
        "top": 200.0,
        "right": 500.0,
        "bottom": 400.0,
        "page": 5
      }
    }
  ]
}
```

### 3. Tables (tables/*.json)

```json
{
  "table_id": "table_001",
  "index": 0,
  "caption": "Table 2. Performance Results",  // Resuelto!
  "markdown": "| Col1 | Col2 |\n|------|------|\n...",
  "html": "<table>...</table>",
  "num_rows": 10,
  "num_cols": 4,
  "parent_cref": "#/body",
  "provenance": [...]
}
```

### 4. Layout (layout.json)

Árbol jerárquico del documento:

```json
{
  "total_nodes": 1523,
  "label_distribution": {
    "paragraph": 450,
    "section_header": 45,
    "list_item": 120,
    "table": 15,
    "picture": 12
  },
  "tree": [
    {"node_id": "node_00001", "type": "TitleItem", "level": 0, ...},
    {"node_id": "node_00002", "type": "SectionHeaderItem", "level": 1, ...}
  ]
}
```

## Características Avanzadas

### Resolución de Referencias

Captions que antes eran `ref:#123` ahora se resuelven a texto real:

```python
# Antes: caption = "ref:#/texts/456"
# Ahora: caption = "Figure 3. System Architecture"
```

**Implementación**: Busca en `doc.texts` por `self_ref` coincidente.

### Bounding Boxes Completos

Todos los elementos con bbox guardan coordenadas completas:

```python
{
  "left": 72.0,
  "top": 100.0,
  "right": 540.0,
  "bottom": 150.0,
  "page": 5
}
```

**Uso futuro**: Multi-column handling, spatial chunking.

### ContentLayer

Preservado cuando disponible:

- `BODY` - contenido principal
- `HEADER` - encabezados de página
- `FOOTER` - pies de página
- `CAPTION` - captions de figuras/tablas

**Uso futuro**: Filtrar headers/footers sin heurísticas.

### Reading Order

Preservado por orden de `iterate_items()`. El índice del nodo = orden de lectura.

**Uso futuro**: PDFs multi-columna, lectura correcta.

## Configuración

### settings (config.py)

```python
cache_dir: str = "data/cache"   # Cache de extracción
audit_dir: str = "data/audit"   # Reportes de auditoría
```

### Pipeline Options

```python
PdfPipelineOptions(
    do_ocr=True,                      # OCR automático
    do_table_structure=True,          # TableFormer
    table_structure_options=TableStructureOptions(
        do_cell_matching=True,        # Match celdas
        mode=TableFormerMode.ACCURATE # Mejor calidad
    ),
    generate_page_images=False,       # True para multimodal
    generate_picture_images=True,     # Extraer figuras
)
```

## Limitaciones Conocidas

### 1. Resolución de Referencias

- **Issue**: Docling no ofrece API oficial de resolución
- **Solución actual**: Lookup manual en `doc.texts` por `self_ref`
- **Limitación**: Puede no resolver todas las referencias

### 2. HTML de Tablas

- **Issue**: `table.export_to_html()` no confirmado en API
- **Solución**: Wrapped en try/except
- **Fallback**: Solo Markdown si HTML no disponible

### 3. ContentLayer

- **Issue**: No todos los items lo tienen
- **Solución**: Solo guardado si presente
- **Impacto**: Algunos nodos sin clasificación de layer

## Flujo de Trabajo Recomendado

### 1. Extracción Inicial

```bash
uv run python scripts/inspect_docling.py paper.pdf
```

### 2. Auditoría Manual

1. Abrir `data/audit/audit_paper_*/markdown.md` ⭐⭐⭐⭐⭐
2. Verificar:
   - Títulos y jerarquía
   - Párrafos completos
   - Tablas formateadas
   - Figuras con captions
   - Referencias correctas

### 3. Revisión de Artefactos

- `nodes.json` - Estructura completa
- `figures/` - Imágenes extraídas
- `tables/` - Tablas en MD/HTML
- `layout.json` - Árbol jerárquico

### 4. Comparación con PDF Original

Abrir PDF y auditoría lado a lado, verificar:
- Número de páginas
- Número de figuras/tablas
- Contenido de secciones clave
- Ecuaciones y fórmulas

### 5. Proceder a Chunking

Una vez verificada la extracción, el chunker trabajará sobre el cache.

## Próximos Pasos

1. ✅ **Extracción completa** - Implementado
2. ✅ **Auditoría** - Implementado
3. ⏭️ **Chunking inteligente** - Usa nodes.json + layout.json
4. ⏭️ **Embeddings** - BGE-M3 sobre chunks
5. ⏭️ **Vector store** - Qdrant con metadata
6. ⏭️ **RAG multimodal** - Chunks + figuras + pages

## Referencias

- Código: `src/rag_app/services/docling_extractor.py`
- Inspector: `scripts/inspect_docling.py`
- Introspección: `scripts/inspect_docling_api.py`
- Notas: `DOCLING_EXTRACTOR_NOTES.md`
