import { useCallback, useState } from "react";
import { API_BASE, type IngestEvent } from "../lib/api";
import { parseSSEStream } from "../lib/sse";

export type IngestStatus = "idle" | "uploading" | "done" | "error";

export interface UseIngestResult {
  status: IngestStatus;
  stageMessage: string | null;
  error: string | null;
  upload: (file: File) => Promise<void>;
}

/**
 * Sube un PDF a POST /documents/upload y sigue el progreso por etapas vía
 * SSE (no hay porcentaje continuo real: extracción y chunking son llamadas
 * bloqueantes de Docling, así que el progreso avanza a saltos).
 */
export function useIngest(onDone?: () => void): UseIngestResult {
  const [status, setStatus] = useState<IngestStatus>("idle");
  const [stageMessage, setStageMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const upload = useCallback(
    async (file: File) => {
      setStatus("uploading");
      setStageMessage("Subiendo...");
      setError(null);

      try {
        const formData = new FormData();
        formData.append("file", file);

        const res = await fetch(`${API_BASE}/documents/upload`, {
          method: "POST",
          body: formData,
        });

        if (!res.ok || !res.body) {
          throw new Error(`El servidor respondió ${res.status}`);
        }

        await parseSSEStream<IngestEvent>(res.body, (event) => {
          if (event.type === "stage") {
            setStageMessage(event.message);
          } else if (event.type === "done") {
            setStatus("done");
            setStageMessage(null);
            onDone?.();
          } else if (event.type === "error") {
            setStatus("error");
            setError(event.error);
          }
        });
      } catch (err) {
        setStatus("error");
        setError(err instanceof Error ? err.message : String(err));
      }
    },
    [onDone]
  );

  return { status, stageMessage, error, upload };
}
