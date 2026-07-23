# RAG Scientific Papers

Sistema RAG (Retrieval-Augmented Generation) para análisis y consulta de artículos científicos. Permite cargar papers en PDF y hacerle preguntas a un chatbot que responde basándose en el contenido de los documentos.

## Arquitectura

El proyecto sigue Clean Architecture con separación clara de responsabilidades:

```
src/rag_app/
├── api/          → Endpoints HTTP (FastAPI routers)
├── core/         → Configuración global y utilidades base
├── models/       → Schemas Pydantic para validación de datos
├── providers/    → Integraciones con servicios externos (embeddings, LLM, Qdrant)
├── repositories/ → Operaciones con la base de datos vectorial
└── services/     → Lógica de negocio y orquestación
```

## Stack Técnico

**Framework:** FastAPI para la API REST

**Extracción de PDFs:** Docling para parsing estructurado de documentos científicos. Mantiene jerarquía, tablas, figuras y referencias.

**Embeddings:** `intfloat/multilingual-e5-large` genera vectores de 1024 dimensiones. Elegido por su excelente rendimiento en tareas multilingües (español + inglés) y búsqueda semántica. Corre localmente sin necesidad de APIs externas.

**Base de datos vectorial:** Qdrant almacena los embeddings y metadata de cada chunk. Soporta búsqueda híbrida nativa (sparse + dense vectors) lo cual es crítico para combinar búsqueda léxica (BM25) y semántica (ANN).

**LLM:** OpenAI GPT-4o-mini para generación de respuestas. Balance entre costo y calidad.

**Chunking:** HybridChunker oficial de Docling. Divide respetando estructura semántica y contextualiza con headings/metadata.

**Gestión de dependencias:** uv por su velocidad y manejo determinista de versiones.

## Instalación

### Requisitos
- Python 3.12+
- Docker Desktop (para correr Qdrant localmente)
- uv package manager

### Setup

```bash
# Instalar dependencias
uv sync

# Configurar variables de entorno
cp .env.example .env
# Editar .env con las claves API necesarias

# Levantar Qdrant
docker-compose up -d

# Iniciar servidor de desarrollo
uv run uvicorn rag_app.main:app --reload
```

El servidor arranca en `http://localhost:8000`. Dashboard de Qdrant disponible en `http://localhost:6333/dashboard`.

## Pipeline de Extracción y Chunking

Usamos **Docling** para extraer y procesar PDFs científicos manteniendo su estructura completa.

### Script de orquestación

```bash
# Procesar un PDF (extracción + chunking)
uv run python src/rag_app/services/run_chunking.py data/uploads/paper.pdf

# Re-chunkear un documento ya extraído (sin OCR por defecto)
uv run python src/rag_app/services/run_chunking.py data/uploads/paper.pdf --force

# Activar OCR para PDFs escaneados
uv run python src/rag_app/services/run_chunking.py data/uploads/paper.pdf --ocr

# Re-chunkear todos los documentos en cache
uv run python src/rag_app/services/run_chunking.py --all --force
```

### ¿Qué extrae Docling?

- **document.json**: DoclingDocument completo serializado
- **markdown.md**: Conversión a markdown para lectura humana
- **figures/**: Imágenes PNG + metadata JSON de cada figura
- **tables/**: Tablas en Markdown/HTML + metadata JSON
- **chunks/chunks.json**: Todos los chunks con embeddings contextualizados
- **chunks/chunks_audit.md**: Reporte legible para revisar calidad del chunking
- **chunks/statistics.json**: Estadísticas de distribución de tokens

### Features del chunking

El **HybridChunker** oficial de Docling:
- Divide respetando estructura semántica (no por tamaño fijo)
- Usa tokenizer de `multilingual-e5-large` (mismo que embeddings)
- Contextualiza chunks con headings jerárquicos
- **Vincula chunks con sus figuras**: cada chunk sabe qué imágenes le pertenecen
- Corrige running headers mal clasificados por el layout model
- **OCR desactivado por defecto** para PDFs nativos digitales

### Auditar antes de embeddings

Revisa `data/cache/{documento}/chunks/chunks_audit.md` para ver:
- Texto exacto que se enviará al modelo de embeddings
- Headings jerárquicos de cada chunk
- Imágenes asociadas a cada chunk (para LLMs multimodales)
- Páginas cubiertas
- Distribución de tokens

### ¿Por qué Docling?

Los PDFs científicos son complicados: multi-columna, ecuaciones, tablas complejas, figuras referenciadas. Docling entiende la estructura del documento académico y extrae todo correctamente, preservando relaciones entre texto e imágenes.

## Desarrollo

### Testing básico

```bash
# Verificar que el servidor responde
curl http://localhost:8000/health

# Probar pipeline completo de extracción + chunking
uv run python src/rag_app/services/run_chunking.py data/uploads/paper.pdf

# Revisar chunks generados
cat data/cache/paper/chunks/chunks_audit.md
```

### Estado actual

**Completado:**
- ✅ Configuración base del proyecto
- ✅ Provider de embeddings funcional
- ✅ Integración con Qdrant
- ✅ Extracción estructurada de PDFs con Docling
- ✅ Chunking inteligente con HybridChunker
- ✅ Vinculación de chunks con imágenes para RAG multimodal
- ✅ Corrección de running headers mal clasificados
- ✅ OCR configurable (desactivado por defecto)
- ✅ Pipeline de embeddings + indexación (chunks.json → Qdrant, idempotente)
- ✅ Búsqueda híbrida (BM25 + vectorial + RRF) con filtrado por metadata (página, headings)

**En desarrollo:**
- ⏳ Pipeline de ingesta (endpoint para subir papers)
- ⏳ Integración con LLM
- ⏳ Endpoint de chat
- ⏳ Reranking con cross-encoder (evaluar si hace falta tras medir calidad del RRF)

## Decisiones Técnicas

**¿Por qué Qdrant sobre pgvector?** Qdrant está diseñado específicamente para búsqueda vectorial y ofrece búsqueda híbrida nativa. Para un proyecto futuro con ~5-10k chunks de audio/video, la optimización de Qdrant para este caso de uso específico justifica la complejidad de un servicio adicional.

**¿Por qué chunking inteligente?** Los PDFs científicos tienen estructura clara (abstract, introducción, métodos, resultados). Partir el documento respetando esta estructura en lugar de usar tamaño fijo crea chunks más coherentes que mejoran la calidad del retrieval.

**¿Por qué e5-large sobre modelos más pequeños?** Con 1024 dimensiones captura mejor la semántica de textos técnicos complejos. La diferencia en calidad (~5-10% más accuracy) justifica el costo computacional para un sistema de producción.
