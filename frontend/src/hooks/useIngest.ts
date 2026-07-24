import { useCallback, useState } from "react";
import { API_BASE, type IngestEvent } from "../lib/api";
import { parseSSEStream } from "../lib/sse";

export type IngestStatus = "idle" | "uploading" | "done" | "error";

export interface UseIngestResult {
  status: IngestStatus;
  stageMessage: string | null;
  error: string | null;
  upload: (files: File[]) => Promise<void>;
}

async function uploadOne(file: File, onStage: (message: string) => void): Promise<void> {
  const formData = new FormData();
  formData.append("file", file);

  const res = await fetch(`${API_BASE}/documents/upload`, { method: "POST", body: formData });
  if (!res.ok || !res.body) {
    throw new Error(`El servidor respondió ${res.status}`);
  }

  let failed: string | null = null;
  await parseSSEStream<IngestEvent>(res.body, (event) => {
    if (event.type === "stage") {
      onStage(event.message);
    } else if (event.type === "error") {
      failed = event.error;
    }
  });
  if (failed) throw new Error(failed);
}

// Archivos en secuencia, no en paralelo: el backend es un solo proceso
// CPU-bound en el embedding, subir en paralelo solo generaría contención.
export function useIngest(onDone?: () => void): UseIngestResult {
  const [status, setStatus] = useState<IngestStatus>("idle");
  const [stageMessage, setStageMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const upload = useCallback(
    async (files: File[]) => {
      if (files.length === 0) return;

      setStatus("uploading");
      setError(null);

      const failures: string[] = [];
      for (let i = 0; i < files.length; i++) {
        const file = files[i];
        const prefix = files.length > 1 ? `Subiendo ${i + 1}/${files.length}: ${file.name} — ` : "";
        setStageMessage(`${prefix}Subiendo...`);

        try {
          await uploadOne(file, (message) => setStageMessage(`${prefix}${message}`));
        } catch (err) {
          failures.push(`${file.name}: ${err instanceof Error ? err.message : String(err)}`);
        }
      }

      setStageMessage(null);
      if (failures.length > 0) {
        setStatus("error");
        setError(failures.join(" · "));
      } else {
        setStatus("done");
        onDone?.();
      }
    },
    [onDone]
  );

  return { status, stageMessage, error, upload };
}
