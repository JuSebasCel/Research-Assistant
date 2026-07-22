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

**Chunking:** Estrategia inteligente usando la estructura del documento extraída por Docling.

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

## Extracción de PDFs

Usamos **Docling** para extraer información estructurada de los papers científicos. A diferencia de extractores simples que solo sacan texto plano, Docling preserva la estructura completa del documento.

### ¿Qué extrae?

- Jerarquía del documento (títulos, secciones, subsecciones)
- Párrafos con su contexto estructural
- Tablas en formato Markdown/HTML
- Figuras con sus captions
- Referencias entre elementos
- Coordenadas espaciales (bounding boxes)
- Metadata de páginas

Todo se guarda en `data/cache/` para no tener que reprocesar el PDF.

### Auditar la extracción

Antes de mandar un documento al RAG, puedes revisar que la extracción salió bien:

```bash
uv run python scripts/inspect_docling.py data/uploads/paper.pdf
```

Esto genera un reporte en `data/audit/` con:
- El markdown extraído (lo más importante para revisar)
- Estadísticas del documento
- Todas las figuras y tablas extraídas
- Checklist de qué revisar manualmente

### ¿Por qué Docling?

Los PDFs científicos son complicados: multi-columna, ecuaciones, tablas complejas, figuras referenciadas. Docling entiende la estructura del documento académico y extrae todo correctamente, no solo el texto.

## Desarrollo

### Testing básico

```bash
# Verificar que el servidor responde
curl http://localhost:8000/health

# Probar extracción de PDF
uv run python scripts/inspect_docling.py data/uploads/paper.pdf

# Probar provider de embeddings
uv run python scripts/test_embeddings.py
```

### Estado actual

**Completado:**
- ✅ Configuración base del proyecto
- ✅ Provider de embeddings funcional
- ✅ Integración con Qdrant
- ✅ Extracción estructurada de PDFs con Docling

**En desarrollo:**
- 🔄 Chunking inteligente usando estructura del documento
- ⏳ Pipeline de ingesta (endpoint para subir papers)
- ⏳ Búsqueda híbrida (BM25 + vectorial + RRF)
- ⏳ Integración con LLM
- ⏳ Endpoint de chat

## Decisiones Técnicas

**¿Por qué Qdrant sobre pgvector?** Qdrant está diseñado específicamente para búsqueda vectorial y ofrece búsqueda híbrida nativa. Para un proyecto futuro con ~5-10k chunks de audio/video, la optimización de Qdrant para este caso de uso específico justifica la complejidad de un servicio adicional.

**¿Por qué chunking inteligente?** Los PDFs científicos tienen estructura clara (abstract, introducción, métodos, resultados). Partir el documento respetando esta estructura en lugar de usar tamaño fijo crea chunks más coherentes que mejoran la calidad del retrieval.

**¿Por qué e5-large sobre modelos más pequeños?** Con 1024 dimensiones captura mejor la semántica de textos técnicos complejos. La diferencia en calidad (~5-10% más accuracy) justifica el costo computacional para un sistema de producción.
