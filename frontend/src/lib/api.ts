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
  document_filters?: string[] | null;
  page_filter?: number | null;
  heading_contains?: string | null;
}

export interface DocumentInfo {
  document_name: string;
  display_name: string;
  folder: string | null;
}

export async function fetchDocuments(): Promise<DocumentInfo[]> {
  const res = await fetch(`${API_BASE}/documents`);
  if (!res.ok) {
    throw new Error(`No se pudo cargar la lista de documentos (${res.status})`);
  }
  return res.json();
}

export async function updateDocument(
  documentName: string,
  update: { display_name?: string; folder?: string | null }
): Promise<DocumentInfo> {
  const res = await fetch(`${API_BASE}/documents/${encodeURIComponent(documentName)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(update),
  });
  if (!res.ok) {
    throw new Error(`No se pudo actualizar el documento (${res.status})`);
  }
  return res.json();
}

export interface GeminiKeyStatus {
  has_custom_key: boolean;
  key_hint: string | null;
}

export async function fetchGeminiKeyStatus(): Promise<GeminiKeyStatus> {
  const res = await fetch(`${API_BASE}/settings/gemini-key`);
  if (!res.ok) throw new Error(`No se pudo consultar la clave (${res.status})`);
  return res.json();
}

export async function saveGeminiKey(apiKey: string): Promise<GeminiKeyStatus> {
  const res = await fetch(`${API_BASE}/settings/gemini-key`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ api_key: apiKey }),
  });
  if (!res.ok) throw new Error(`No se pudo guardar la clave (${res.status})`);
  return res.json();
}

export async function clearGeminiKey(): Promise<GeminiKeyStatus> {
  const res = await fetch(`${API_BASE}/settings/gemini-key`, { method: "DELETE" });
  if (!res.ok) throw new Error(`No se pudo quitar la clave (${res.status})`);
  return res.json();
}
