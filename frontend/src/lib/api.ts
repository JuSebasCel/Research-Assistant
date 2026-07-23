export const API_BASE = "http://localhost:8000";

export interface Citation {
  document_name: string;
  chunk_id: string;
  pages: number[];
  image_urls: string[];
}

export type ChatEvent =
  | { type: "chunk"; text: string }
  | { type: "done"; citations: Citation[] }
  | { type: "no_results" }
  | { type: "error"; error: string };

export type IngestStage = "uploading" | "extracting" | "chunking" | "indexing";

export type IngestEvent =
  | { type: "stage"; stage: IngestStage; message: string }
  | { type: "done"; document_name: string; chunks_indexed: number }
  | { type: "error"; error: string };

export interface ChatRequest {
  query: string;
  top_k?: number;
  document_filter?: string | null;
  page_filter?: number | null;
  heading_contains?: string | null;
}

export async function fetchDocuments(): Promise<string[]> {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) {
    throw new Error(`No se pudo cargar la lista de documentos (${res.status})`);
  }
  return res.json();
}
