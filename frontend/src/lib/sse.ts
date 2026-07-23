/**
 * Parseo de streams SSE compartido entre chat e ingesta.
 *
 * EventSource nativo no sirve para requests POST (solo soporta GET), así
 * que el parseo se hace a mano sobre un fetch() + ReadableStream. Extraído
 * de useConversations porque useIngest necesita exactamente la misma lógica
 * de framing (líneas "data: {...}" separadas por líneas en blanco).
 */
export async function parseSSEStream<T>(
  body: ReadableStream<Uint8Array>,
  onEvent: (event: T) => void
): Promise<void> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const chunks = buffer.split("\n\n");
    buffer = chunks.pop() ?? "";

    for (const chunk of chunks) {
      const line = chunk.trim();
      if (!line.startsWith("data:")) continue;

      const payload = line.slice("data:".length).trim();
      if (!payload) continue;

      onEvent(JSON.parse(payload) as T);
    }
  }
}
