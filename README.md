# Research Assistant

RAG (Retrieval-Augmented Generation) para papers científicos: subís PDFs, el sistema los procesa manteniendo su estructura, y le preguntás en lenguaje natural con respuestas citadas y trazables al documento y página exactos.

## ¿Qué es?

Un backend en FastAPI + un frontend en React que juntos arman el flujo completo: extracción estructurada de PDFs (Docling), búsqueda híbrida (BM25 + embeddings densos, fusionados con RRF nativo de Qdrant), y generación de respuestas con Gemini, siempre citando de dónde salió cada afirmación.

No hay login ni multi-tenant: pensado para correr en tu propia máquina, sobre tu propia colección de papers.

## Stack técnico

| Capa | Elección | Por qué |
| --- | --- | --- |
| Extracción de PDF | [Docling](https://github.com/docling-project/docling) | Entiende estructura académica real (multi-columna, tablas, figuras referenciadas), no solo texto plano |
| Embeddings densos | `intfloat/multilingual-e5-large` | Multilingüe (ES/EN), corre localmente, sin dependencia de una API externa |
| Embeddings sparse | `Qdrant/bm25` (fastembed) | Señal léxica complementaria a la semántica |
| Vector store | [Qdrant](https://qdrant.tech/) | Named vectors + fusión RRF nativa entre dense y sparse |
| LLM | Google Gemini | Tier gratuito real (sin tarjeta), a diferencia de un crédito único que expira |
| Backend | FastAPI | Streaming (SSE) nativo para chat e ingesta |
| Frontend | React 19 + Vite + Tailwind v4 | — |

## Estructura del proyecto

```
src/rag_app/
├── api/            # Routers de FastAPI (chat, documents, ingest, settings)
├── core/           # Config centralizada, logging, formato SSE compartido
├── models/         # Schemas Pydantic
├── providers/      # Integraciones externas (Gemini, embeddings, Qdrant client)
├── repositories/   # Operaciones CRUD sobre Qdrant
└── services/       # Orquestación: extracción, chunking, indexación, chat

frontend/src/
├── components/     # Sidebar, ChatView, MessageSources, SettingsPanel, Markdown
├── hooks/          # useConversations (localStorage), useIngest (upload + SSE)
└── lib/            # Cliente HTTP, parseo de SSE compartido, formato de fechas
```

## Correr en local

Necesitás **Python 3.12+** con [uv](https://docs.astral.sh/uv/), **Node.js 22+**, y Docker (para Qdrant).

```bash
# Qdrant
docker-compose up -d

# Backend
uv sync
cp .env.example .env
# completar GEMINI_API_KEY en .env (gratis: https://aistudio.google.com/apikey)
uv run uvicorn rag_app.main:app --reload

# Frontend (otra terminal)
cd frontend
npm install
npm run dev
```

Frontend en `http://localhost:5173`, backend en `http://localhost:8000`, dashboard de Qdrant en `http://localhost:6333/dashboard`.

### Scripts

| Comando | Qué hace |
| --- | --- |
| `uv run uvicorn rag_app.main:app --reload` | Servidor de desarrollo del backend |
| `uv run pytest` | Suite de tests (marca `integration` requiere `GEMINI_API_KEY` real) |
| `uv run ruff check src/ tests/` | Lint del backend |
| `npm run dev` | Servidor de desarrollo del frontend |
| `npm run build` | Build de producción (`tsc -b && vite build`) |
| `npm run lint` | Lint del frontend (oxlint) |

Alternativa a la interfaz: `src/rag_app/services/run_chunking.py` y `run_indexing.py` procesan e indexan PDFs por línea de comandos (`--help` para ver opciones).

## Cómo funciona

1. **Extracción** (Docling): PDF → árbol estructurado con jerarquía, tablas, figuras y su ubicación exacta en página.
2. **Chunking** (HybridChunker): respeta la estructura semántica del documento, no corta por tamaño fijo. Cada chunk queda vinculado a sus figuras.
3. **Indexación**: cada chunk se embebe con dos señales (densa + sparse) y se sube a Qdrant con UUIDs determinísticos, así reindexar el mismo documento actualiza en vez de duplicar.
4. **Búsqueda**: fusión RRF de ambas señales. Sin filtrar a un documento, la búsqueda hace fan-out por cada documento indexado para no dejar afuera papers relevantes que no ganan el ranking global.
5. **Generación**: Gemini responde solo con lo que está en los fragmentos recuperados, citando documento y página; si no encuentra nada relevante, lo dice en vez de inventar.

## Features

- Chat con streaming, citas obligatorias, y contexto multimodal (envía las figuras asociadas a cada fragmento)
- Búsqueda híbrida con fan-out multi-documento y filtrado por documento o carpeta
- Ingesta de PDFs desde la interfaz, con progreso por etapas vía SSE
- Metadata de documentos editable (nombre visible, carpeta, drag-and-drop entre carpetas) sin tocar la clave real de indexación
- Fuentes por mensaje con link directo al PDF original en la página citada
- Múltiples conversaciones persistidas en localStorage (renombrar, pinear, borrar)
- Cada persona puede traer su propia API key de Gemini en vez de compartir la cuota del servidor

## Decisiones técnicas

**¿Por qué Qdrant y no pgvector?** Qdrant está construido específicamente para búsqueda vectorial, con fusión híbrida (dense + sparse) nativa server-side — evita tener que implementar RRF a mano.

**¿Por qué chunking respetando estructura en vez de tamaño fijo?** Los papers científicos tienen secciones claras (abstract, métodos, resultados); partir por esa estructura produce chunks más coherentes que mejoran el retrieval.

**¿Por qué el umbral de confianza es tan bajo?** El score RRF es ranking relativo dentro de la colección, no similitud absoluta — preguntas fuera de tema y preguntas relevantes caen en rangos de score casi idénticos. La defensa real contra respuestas inventadas es la instrucción del sistema al LLM, no el umbral.

**¿Por qué fan-out en vez de una sola búsqueda global?** Confirmado con datos reales: si un documento domina el ranking, los demás pueden quedar completamente afuera de los resultados aunque también sean relevantes.
