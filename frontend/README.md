# Frontend — RAG Scientific Papers

Interfaz web para el chat sobre los papers indexados: hilo de conversación con streaming, markdown/LaTeX real, panel de fuentes con imágenes de las figuras citadas.

React + TypeScript + Vite + Tailwind CSS. Paleta cálida inspirada en Claude.ai.

## Desarrollo

Requiere el backend (`rag_app.main:app`, ver README raíz del proyecto) corriendo en `http://localhost:8000` — este frontend le pega directo vía CORS, no hay proxy.

```bash
npm install
npm run dev
```

Abre `http://localhost:5173`.

## Estructura

```
src/
├── App.tsx                  → layout: sidebar + chat + panel de fuentes
├── components/
│   ├── Sidebar.tsx           → lista de documentos indexados (GET /documents)
│   ├── ChatView.tsx          → hilo de conversación + input
│   ├── Markdown.tsx          → render de markdown + LaTeX (KaTeX) para las respuestas
│   ├── SourcesPanel.tsx      → citas del último mensaje, con imágenes de figuras
│   └── ui/                   → primitivas (Card, Pill)
├── hooks/
│   └── useChatSession.ts     → estado del hilo + parseo manual de streaming SSE
│                                (POST /chat no es compatible con EventSource nativo)
└── lib/
    └── api.ts                → tipos compartidos con el backend + base URL
```
